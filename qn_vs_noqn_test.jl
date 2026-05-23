###############################################################################
# Compare QN-aware DMRG (opt 4) vs non-QN DMRG (opt 1+3) at fixed L for the
# same physical setup. Times the per-state cost and computes ̄gap.
#
# Usage:  L_TEST=8 julia qn_vs_noqn_test.jl
###############################################################################

using ITensors, ITensorMPS, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

const L_TEST = parse(Int, get(ENV, "L_TEST", "8"))
const m0, eta, alpha = 0.1, 0.5, 1.0
const k_qn   = 20
const k_noqn = 30

println("\n##### QN vs non-QN benchmark at L=$L_TEST  (OBC :truncate_xz, bg=(-1,-1)) #####")

# ---------- shared init pattern ----------
# Staggered mass term -m0/2·(-1)^n·Z_n prefers Z_n = (-1)^{n+1}.
# That gives matter-parity ∏ Z_matter = (-1)^{L(L+1)/2 + L} = (-1)^{L(L+3)/2}.
# We compute the matching sector for our QN-aware init.
function staggered_init_strings(L::Int, sites)
    # OBC :open_site: matter at odd chain idx 2j-1 for j=1..L. Set n=1,3,5,... ("Dn")
    # to match Z = -1 favored by mass at odd n (since (-1)^{n+1} with n odd is +1...
    # wait, n=1: (-1)^2 = +1 → Z = +1 = Up).
    # Let's match exactly: Z_n = (-1)^{n+1}. n=1: Z=+1 (Up). n=2: Z=-1 (Dn). ...
    # So "Dn" goes at matter sites where n is even, i.e., j even (chain 3, 7, 11, ...).
    states = ["Up" for _ in 1:length(sites)]
    for j in 1:L
        if j % 2 == 0
            states[2j - 1] = "Dn"   # matter at chain idx 2j-1
        end
    end
    return states
end

# ---------- run QN-aware DMRG ----------
function run_qn(L)
    is_m = i -> isodd(i)
    N = 2L - 1
    sites = Index[]
    for i in 1:N
        if is_m(i)
            push!(sites, Index([QN("P", 0, 2) => 1, QN("P", 1, 2) => 1];
                               tags="S=1/2,Site,n=$i"))
        else
            push!(sites, Index([QN("P", 0, 2) => 2]; tags="S=1/2,Site,n=$i"))
        end
    end

    # Reuse the existing make_z2_gauge_theory_hamiltonian for OpSum logic, but
    # build MPO using QN-aware sites. We re-implement the OpSum construction
    # here to avoid threading sites through everything.
    os = OpSum()
    matter_site = j -> 2j - 1
    link_site   = j -> 2j
    lam = 20.0
    bg_left, bg_right = -1, -1

    for j in 1:(L-1)
        os += alpha/2, "X", link_site(j)
    end
    for j in 1:L
        os += -m0/2 * (j % 2 == 0 ? 1 : -1), "Z", matter_site(j)
    end
    for j in 1:(L-1)
        mi, bi, mj = matter_site(j), link_site(j), matter_site(j+1)
        os += eta/4, "X", mi, "Z", bi, "X", mj
        os += eta/4, "Y", mi, "Z", bi, "Y", mj
    end
    for j in 2:(L-1)
        mi, ll, lr = matter_site(j), link_site(j-1), link_site(j)
        os += -lam/2 * (-1)^j, "X", ll, "Z", mi, "X", lr
    end
    os += -lam/2 * (-1) * bg_left,    "Z", matter_site(1),       "X", link_site(1)
    os += -lam/2 * (-1)^L * bg_right, "X", link_site(L-1),       "Z", matter_site(L)

    H = MPO(os, sites)
    init = staggered_init_strings(L, sites)
    psi0_init = MPS(sites, init)
    @printf("[QN] init parity flux = %s\n", flux(psi0_init))

    sw = Sweeps(80)
    setmaxdim!(sw, 10, 20, 40, 80, 100, 200, 300, 300, 300, 300)
    setcutoff!(sw, 1e-11)
    setnoise!(sw, 1e-7, 1e-8, 1e-9, 0.0)

    Random.seed!(1)
    tstart = time()
    e0, ψ0 = dmrg(H, psi0_init, sw; outputlevel=0)
    energies = [e0]
    psis = [ψ0]
    for k in 1:k_qn
        ψinit = random_mps(sites, init; linkdims=2)
        ek, ψk = dmrg(H, copy.(psis), ψinit, sw; weight=200.0, outputlevel=0)
        push!(energies, ek)
        push!(psis, ψk)
    end
    t = time() - tstart

    O = make_O_pzero_mpo(sites, L, :open_site)
    overlaps = Float64[abs2(inner(psis[k+1]', O, psis[1])) for k in 0:k_qn]
    sum_w = sum(overlaps[2:end])
    barE  = sum(overlaps[k+1]*(energies[k+1]-energies[1]) for k in 1:k_qn) / sum_w
    @printf("[QN]  L=%d  time=%.1fs  k_max=%d  W_tot=%.5f  ̄gap=%.5f\n",
            L, t, k_qn, sum_w, barE)
    return (time=t, barE=barE, W=sum_w, energies=energies, overlaps=overlaps)
end

# ---------- run non-QN DMRG (opt 1+3 stack) ----------
function run_noqn(L)
    p = Data.Params(; m0=m0, eta=eta, alpha=alpha, Lphys=L, L=L)
    H, sites = build_H(p; boundary=:open_site, gauge_law=:z2,
                       z2_obc_boundary=:truncate_xz, bg_left=-1, bg_right=-1)

    Random.seed!(1)
    tstart = time()
    # IMPORTANT: fixed 80 sweeps, no early-stop, to match the QN path settings.
    energies, psis = solve_z2_higgs_k(H, sites; k_max=k_noqn,
                                      nsweeps=80, weight=200.0,
                                      maxdim_final=300, cutoff=1e-11,
                                      early_stop_tol=0.0)
    t = time() - tstart

    O = make_O_pzero_mpo(sites, L, :open_site)
    overlaps = Float64[abs2(inner(psis[k+1]', O, psis[1])) for k in 0:k_noqn]
    sum_w = sum(overlaps[2:end])
    barE  = sum(overlaps[k+1]*(energies[k+1]-energies[1]) for k in 1:k_noqn) / sum_w
    @printf("[noQN]L=%d  time=%.1fs  k_max=%d  W_tot=%.5f  ̄gap=%.5f\n",
            L, t, k_noqn, sum_w, barE)
    return (time=t, barE=barE, W=sum_w, energies=energies, overlaps=overlaps)
end

println("\n--- running QN ---")
qn_res = run_qn(L_TEST)
println("\n--- running non-QN (opt 1+3) ---")
noqn_res = run_noqn(L_TEST)

println("\n=========== SUMMARY ===========")
@printf("L=%d, OBC :truncate_xz bg=(-1,-1)\n", L_TEST)
@printf("QN     k_max=%d  time=%.1fs  ̄gap=%.5f  W=%.5f\n",
        k_qn, qn_res.time, qn_res.barE, qn_res.W)
@printf("non-QN k_max=%d  time=%.1fs  ̄gap=%.5f  W=%.5f\n",
        k_noqn, noqn_res.time, noqn_res.barE, noqn_res.W)
@printf("speedup_QN_vs_noQN = %.2fx\n", noqn_res.time / qn_res.time)
