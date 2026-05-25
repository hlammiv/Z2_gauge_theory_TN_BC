###############################################################################
# Test warm-start MPS save/load at L=6 for both gauge and claude.
###############################################################################
using Printf

const STATES_G = joinpath(@__DIR__, "data", "states_gauge_test")
const STATES_C = joinpath(@__DIR__, "data", "states_claude_test")
const L_TEST = 6
const NSW    = 80
const KMAX   = 10

# ---------- GAUGE ----------
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))
println("\n===== gauge cold start =====")
t_cold_g = @elapsed convergence_overlap_pzero(;
    Larr=[L_TEST], k_max=KMAX, m0=0.1, eta=0.5,
    bg_left=-1, bg_right=-1, nsweeps=NSW, seed=1,
    save_states_dir=STATES_G, fname_suffix="_warm_cold")
@printf("\n>>> gauge cold elapsed: %.1fs\n", t_cold_g)

println("\n===== gauge warm start =====")
t_warm_g = @elapsed convergence_overlap_pzero(;
    Larr=[L_TEST], k_max=KMAX, m0=0.1, eta=0.5,
    bg_left=-1, bg_right=-1, nsweeps=NSW, seed=1,
    warm_start_dir=STATES_G, fname_suffix="_warm_warm")
@printf("\n>>> gauge warm elapsed: %.1fs  (speedup = %.2fx)\n", t_warm_g, t_cold_g/t_warm_g)

# ---------- CLAUDE ----------
include(joinpath(@__DIR__, "z2-claude.jl"))
println("\n\n===== claude cold start =====")
t_cold_c = @elapsed convergence_overlap_pzero(;
    N_list=[L_TEST], K=KMAX, m=0.1, η=0.5,
    bg_left=-1, bg_right=-1, nsweeps_gs=NSW, nsweeps_ex=NSW, seed=1,
    save_states_dir=STATES_C, fname_suffix="_warm_cold")
@printf("\n>>> claude cold elapsed: %.1fs\n", t_cold_c)

println("\n===== claude warm start =====")
t_warm_c = @elapsed convergence_overlap_pzero(;
    N_list=[L_TEST], K=KMAX, m=0.1, η=0.5,
    bg_left=-1, bg_right=-1, nsweeps_gs=NSW, nsweeps_ex=NSW, seed=1,
    warm_start_dir=STATES_C, fname_suffix="_warm_warm")
@printf("\n>>> claude warm elapsed: %.1fs  (speedup = %.2fx)\n", t_warm_c, t_cold_c/t_warm_c)
