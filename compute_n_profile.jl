###############################################################################
# Compute ground-state matter-site <Z_n> profiles for L ∈ {4,6,8,10,12,14,16,18}
# and boundaries {OBC truncate_xz, PBC drop}.
#
# In KS staggered convention with mass term -(m0/2)(-1)^n Z_n:
#   <Z_n> ≈ (-1)^n in the half-filled vacuum.
#   particle occupation: <N_n> = (1 - (-1)^n <Z_n>) / 2  ∈ [0,1]
#   chiral condensate:   <ψ̄ψ>_n = (-1)^n <Z_n> / 2  (per-site staggered sign)
#
# Convention:
#   OBC :open_site: matter qubits at indices 1, 3, ..., 2L-1
#   PBC:            matter qubits at indices 2, 4, ..., 2L
###############################################################################
using Printf, CSV, DataFrames, ITensors, ITensorMPS, HDF5
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

const OUT_CSV = joinpath(@__DIR__, "data", "n_profile_z2_gauge.csv")
const L_LIST  = [4, 6, 8, 10, 12, 14, 16, 18]
const NSWEEPS_GS = 200
const MAXDIM     = 300
# Meson maxdim is allowed larger since O|Ω⟩ may have larger bond dim
const MAXDIM_MESON = 600

function matter_qubit_index(n::Int, boundary::Symbol)
    boundary == :open_site ? 2n - 1 : 2n
end

function ground_state(L::Int, boundary::Symbol, z2_obc::Symbol; nsweeps_gs=NSWEEPS_GS)
    p = Data.Params(L=L, alpha=1.0, eta=0.5, m0=0.1)
    H, sites = redirect_stdout(devnull) do
        make_z2_gauge_theory_hamiltonian(p; boundary=boundary,
                                          gauge_law=:z2,
                                          lambda_gauss=20.0,
                                          z2_obc_boundary=z2_obc,
                                          bg_left=-1, bg_right=-1)
    end
    N = length(sites)
    psi0 = randomMPS(sites; linkdims=10)
    sweeps = Sweeps(nsweeps_gs)
    setmaxdim!(sweeps, 20, 40, 80, 150, 250, MAXDIM)
    setcutoff!(sweeps, 1e-11)
    setnoise!(sweeps, 1e-6, 1e-7, 1e-8, 1e-9, 0.0)
    _, psi = dmrg(H, psi0, sweeps; outputlevel=0)
    return psi, sites
end

function measure_Z_profile(psi, sites, L::Int, boundary::Symbol)
    Zs = Float64[]
    for n in 1:L
        q = matter_qubit_index(n, boundary)
        op_Z = op("Z", sites[q])
        orthogonalize!(psi, q)
        psi_q = psi[q]
        z = scalar(dag(prime(psi_q, "Site")) * op_Z * psi_q)
        push!(Zs, real(z))
    end
    return Zs
end

# Build the p=0 meson MPO, apply to ground state, normalize.
function meson_p0_state(psi_gs, sites, L::Int, boundary::Symbol)
    O = redirect_stdout(devnull) do
        make_O_pzero_mpo(sites, L, boundary)
    end
    psi_meson = apply(O, psi_gs; maxdim=MAXDIM_MESON, cutoff=1e-12)
    nrm = norm(psi_meson)
    if nrm < 1e-10
        error("|O|Ω⟩| = $nrm, meson operator annihilates vacuum at L=$L $boundary")
    end
    psi_meson ./= nrm
    return psi_meson, nrm
end

function main()
    rows = NamedTuple[]
    for L in L_LIST
        for (boundary, z2_obc) in [(:open_site, :truncate_xz), (:PBC, :drop)]
            @printf("[n_profile] L=%d %s %s ...\n", L, boundary, z2_obc)
            t = @elapsed begin
                psi_gs, sites = ground_state(L, boundary, z2_obc)
                Zs_gs = measure_Z_profile(psi_gs, sites, L, boundary)
                psi_meson, meson_norm = meson_p0_state(psi_gs, sites, L, boundary)
                Zs_me = measure_Z_profile(psi_meson, sites, L, boundary)
            end
            @printf("  done in %.1fs  (|O|Ω⟩|=%.4f)\n", t, meson_norm)
            for (n, z_gs, z_me) in zip(1:L, Zs_gs, Zs_me)
                push!(rows, (
                    L = L,
                    boundary = String(boundary),
                    z2_obc = String(z2_obc),
                    matter_site = n,
                    state = "ground",
                    Z_expect = z_gs,
                ))
                push!(rows, (
                    L = L,
                    boundary = String(boundary),
                    z2_obc = String(z2_obc),
                    matter_site = n,
                    state = "meson_p0",
                    Z_expect = z_me,
                ))
            end
            CSV.write(OUT_CSV, DataFrame(rows))
        end
    end
    @printf("\nWrote %s with %d rows\n", OUT_CSV, length(rows))
end

main()
