###############################################################################
# DMRG-vs-exact state-by-state comparison at L=4
#
# Goal: see whether DMRG with our production settings (k_max=30, nsweeps_ex=80,
# weight=200, maxdim=400) accurately tracks the k-th eigenstate of H at L=4 —
# where exact diagonalization gives the ground truth for all 128 states.
#
# If DMRG fails to find the correct E_k for high k even at L=4, the L=14
# high-k convergence issues we observed are a general DMRG-state-targeting
# pathology and we should distrust those high-k states everywhere.
###############################################################################

using ITensors, ITensorMPS, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

const L      = 4
const k_max  = 30
const nsweeps_ex_list = (80, 160)
const weight_list     = (200.0, 500.0)
const seeds  = 1:3
const m0, eta, alpha = 0.1, 0.5, 1.0

function exact_energies(boundary, z2_obc)
    p = Data.Params(; m0=m0, eta=eta, alpha=alpha, Lphys=L, L=L)
    H_mpo, sites = build_H(p; boundary=boundary, gauge_law=:z2,
                           z2_obc_boundary=z2_obc, bg_left=-1, bg_right=-1)
    Hmat = mpo_to_array(H_mpo, sites)
    F = eigen(Hermitian(Hmat))
    return F.values, sites, H_mpo
end

function compare(boundary, z2_obc, label)
    println("\n========== ", label, " ==========")
    Eexact, sites, H_mpo = exact_energies(boundary, z2_obc)
    println("Hilbert dim = $(length(Eexact))")
    println("Exact E_0..E_$(k_max):")
    for k in 0:k_max
        @printf("  k=%-3d  E_k=%.6f  gap=%.6f\n", k, Eexact[k+1], Eexact[k+1]-Eexact[1])
    end

    for nsw in nsweeps_ex_list, w in weight_list, seed in seeds
        Random.seed!(seed)
        @printf("\n--- nsweeps_ex=%d  weight=%.0f  seed=%d ---\n", nsw, w, seed)
        # Replicate the production DMRG path: solve_z2_higgs_k with our settings.
        energies, _ = solve_z2_higgs_k(H_mpo, sites; k_max=k_max,
                                       nsweeps=nsw, weight=w,
                                       maxdim_final=400, cutoff=1e-11)
        @printf("  %-3s %-12s %-12s %-12s %-10s\n", "k", "E_DMRG", "E_exact", "ΔE", "rank")
        for k in 0:k_max
            E_d = energies[k+1]
            # Find which exact-k_e most closely matches DMRG's k.
            (_, rank) = findmin(abs.(Eexact .- E_d))
            E_e = Eexact[rank]
            rank_idx = rank - 1   # 0-indexed
            ΔE = E_d - E_e
            tag = rank_idx == k ? "" : "  ← out of order!"
            @printf("  %-3d %-12.6f %-12.6f %+10.3e   exact_k=%-4d%s\n",
                    k, E_d, E_e, ΔE, rank_idx, tag)
        end
    end
end

# Run for OBC :truncate_xz only — that's the boundary we care about for the FV
# study. PBC is similar in structure.
compare(:open_site, :truncate_xz, "OBC :truncate_xz, bg=(-1,-1)")
