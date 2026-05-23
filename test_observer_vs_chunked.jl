###############################################################################
# A/B test: new DMRGObserver-style early-stop vs old chunked-restart early-stop.
# Same problem (L, boundary, Hamiltonian), same init seed, same energy_tol,
# same maxdim/noise/cutoff schedule.  Measures wall time + sweep count actually
# used + final energy + final |ΔE|.
#
# Usage:  julia --project=. test_observer_vs_chunked.jl
###############################################################################
using ITensors, ITensorMPS, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

const L         = 10
const NSWEEPS   = 160
const ENERGY_TOL= 1e-7
const MINSWEEPS = 20
const CHUNK     = 10
const SEED      = 1
const MAXDIM    = 300

# Sweep schedule that matches what the production code uses.
function _make_sweeps(nsweeps_in; maxdim_final=MAXDIM)
    sw = Sweeps(nsweeps_in)
    setmaxdim!(sw, 10, 20, 40, 80, 100, 200, 300, 400, 600, min(800, maxdim_final),
                  min(1000, maxdim_final), min(1200, maxdim_final),
                  min(1600, maxdim_final), min(2000, maxdim_final))
    setcutoff!(sw, 1e-11)
    setnoise!(sw, 1e-7, 1e-8, 1e-9, 0.0)
    return sw
end

# Sweep-counting observer: identical behavior to _EarlyStopObserver but tracks
# the final sweep number where it stopped.
mutable struct _CountingObserver <: AbstractObserver
    energy_tol::Float64
    minsweeps::Int
    last_energy::Float64
    last_sweep::Int
    last_dE::Float64
end
_CountingObserver(tol, mins) = _CountingObserver(tol, mins, Inf, 0, NaN)
ITensorMPS.measure!(o::_CountingObserver; kwargs...) = nothing
function ITensorMPS.checkdone!(o::_CountingObserver; kwargs...)
    sweep = get(kwargs, :sweep, 0)
    e     = get(kwargs, :energy, NaN)
    dE    = abs(e - o.last_energy)
    o.last_energy = e
    o.last_sweep  = sweep
    o.last_dE     = dE
    return sweep >= o.minsweeps && isfinite(dE) && dE < o.energy_tol
end

# OLD chunked-restart approach: call dmrg() repeatedly in `chunk`-sweep blocks,
# stop when |ΔE| between blocks is below tol after at least minsweeps total.
function dmrg_chunked(H, prev_psis, psi_init; nsweeps, chunk=10, energy_tol,
                     minsweeps, weight)
    psi = psi_init
    E   = Inf
    swept = 0
    last_dE = NaN
    while swept < nsweeps
        nsw_this = min(chunk, nsweeps - swept)
        sw = _make_sweeps(nsw_this)
        E_new, psi = weight > 0 ?
            dmrg(H, prev_psis, psi, sw; weight=weight, outputlevel=0) :
            dmrg(H, psi, sw; outputlevel=0)
        swept += nsw_this
        last_dE = abs(E_new - E)
        E = E_new
        if swept >= minsweeps && isfinite(last_dE) && last_dE < energy_tol
            break
        end
    end
    return E, psi, swept, last_dE
end

# NEW observer approach.
function dmrg_observer(H, prev_psis, psi_init; nsweeps, energy_tol, minsweeps, weight)
    obs = _CountingObserver(energy_tol, minsweeps)
    sw  = _make_sweeps(nsweeps)
    E, psi = weight > 0 ?
        dmrg(H, prev_psis, psi_init, sw; weight=weight, observer=obs, outputlevel=0) :
        dmrg(H, psi_init, sw; observer=obs, outputlevel=0)
    return E, psi, obs.last_sweep, obs.last_dE
end

# --- Run the A/B test ---
@info "A/B test: chunked vs observer at L=$L, nsweeps_max=$NSWEEPS, tol=$ENERGY_TOL"
p = Data.Params(; m0=0.1, eta=0.5, alpha=1.0, Lphys=L, L=L)
H, sites = build_H(p; boundary=:PBC, gauge_law=:z2,
                   z2_obc_boundary=:drop, bg_left=-1, bg_right=-1)

println("\n===== JIT WARMUP (excluded from timing) =====")
let psi_warm = random_mps(sites; linkdims=2)
    sw_warm = _make_sweeps(2)
    obs_w = _CountingObserver(ENERGY_TOL, MINSWEEPS)
    dmrg(H, psi_warm, sw_warm; observer=obs_w, outputlevel=0)
    dmrg(H, psi_warm, sw_warm; outputlevel=0)  # chunked path also calls plain dmrg
end
println("warmup done")

println("\n===== GROUND STATE =====")

# Run 1: observer (new)
Random.seed!(SEED)
psi0_init = random_mps(sites; linkdims=2)
@printf("[observer] starting...\n"); flush(stdout)
t1 = @elapsed (E_obs, psi_obs, sweeps_obs, dE_obs) = dmrg_observer(H, MPS[], psi0_init;
    nsweeps=NSWEEPS, energy_tol=ENERGY_TOL, minsweeps=MINSWEEPS,
    weight=0.0)
@printf("[observer]   E = %.10f   sweeps_done = %d   final |ΔE| = %.2e   wall = %.1fs\n",
        E_obs, sweeps_obs, dE_obs, t1)

# Run 2: chunked (old)
Random.seed!(SEED)
psi0_init2 = random_mps(sites; linkdims=2)
@printf("[chunked]  starting...\n"); flush(stdout)
t2 = @elapsed (E_ch, psi_ch, sweeps_ch, dE_ch) = dmrg_chunked(H, MPS[], psi0_init2;
    nsweeps=NSWEEPS, chunk=CHUNK, energy_tol=ENERGY_TOL, minsweeps=MINSWEEPS,
    weight=0.0)
@printf("[chunked]    E = %.10f   sweeps_done = %d   final |ΔE| = %.2e   wall = %.1fs\n",
        E_ch, sweeps_ch, dE_ch, t2)

println()
println("---")
@printf("Wall time observer = %.1fs, chunked = %.1fs  → ratio chunked/observer = %.2fx\n",
        t1, t2, t2/t1)
@printf("Sweep count observer = %d, chunked = %d\n", sweeps_obs, sweeps_ch)
@printf("ΔE between methods: %.2e\n", abs(E_obs - E_ch))

# --- Same with a few excited states ---
function run_excited_loop(psi_obs, psi_ch, E_obs0, E_ch0, t1, t2, sweeps_obs, sweeps_ch)
    prev_obs   = [psi_obs]
    prev_ch    = [psi_ch]
    t_obs_tot, t_ch_tot = t1, t2
    sw_obs_tot, sw_ch_tot = sweeps_obs, sweeps_ch
    for k in 1:3
        Random.seed!(SEED + k)
        psi_init = random_mps(sites; linkdims=2)
        t_o = @elapsed (E_o, ψ_o, sw_o, dE_o) = dmrg_observer(H, prev_obs, psi_init;
            nsweeps=NSWEEPS, energy_tol=ENERGY_TOL, minsweeps=MINSWEEPS,
            weight=200.0)
        push!(prev_obs, ψ_o)
        Random.seed!(SEED + k)
        psi_init2 = random_mps(sites; linkdims=2)
        t_c = @elapsed (E_c, ψ_c, sw_c, dE_c) = dmrg_chunked(H, prev_ch, psi_init2;
            nsweeps=NSWEEPS, chunk=CHUNK, energy_tol=ENERGY_TOL, minsweeps=MINSWEEPS,
            weight=200.0)
        push!(prev_ch, ψ_c)
        t_obs_tot += t_o; t_ch_tot += t_c
        sw_obs_tot += sw_o; sw_ch_tot += sw_c
        @printf("k=%d: observer (E=%.6f,sw=%3d,|ΔE|=%.1e,t=%.1fs)  chunked (E=%.6f,sw=%3d,|ΔE|=%.1e,t=%.1fs)  ΔE_methods=%.2e\n",
                k, E_o, sw_o, dE_o, t_o, E_c, sw_c, dE_c, t_c, abs(E_o - E_c))
    end
    return t_obs_tot, t_ch_tot, sw_obs_tot, sw_ch_tot
end
println("\n===== EXCITED STATES (k=1,2,3) =====")
t_obs_tot, t_ch_tot, sw_obs_tot, sw_ch_tot =
    run_excited_loop(psi_obs, psi_ch, E_obs, E_ch, t1, t2, sweeps_obs, sweeps_ch)

println("\n--- SUMMARY (ground + 3 excited) ---")
@printf("Total wall    observer = %.1fs   chunked = %.1fs   ratio chunked/obs = %.2fx\n",
        t_obs_tot, t_ch_tot, t_ch_tot/t_obs_tot)
@printf("Total sweeps  observer = %d   chunked = %d\n", sw_obs_tot, sw_ch_tot)
