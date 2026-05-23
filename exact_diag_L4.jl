###############################################################################
# Exact-diagonalization investigation of the spectral first moment ̄E at small L
#
# We:
#   1. Build the full Z₂ LGT Hamiltonian H as a dense matrix.
#   2. Exact-diagonalize via LAPACK.
#   3. Build O_p=0 as a dense matrix.
#   4. Compute the EXACT overlap |⟨ψ_k|O|ψ_0⟩|² over ALL eigenstates.
#   5. Tabulate W_total and ̄E for various truncations k_max.
#
# The L=4 case has 128 (OBC :open_site) or 256 (PBC) eigenstates - trivial for
# exact diagonalization. This is the gold standard to compare against DMRG with
# truncated k_max.
###############################################################################

using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

# Re-include the z2-gauge-theory.jl machinery: build_H, make_O_pzero_mpo, etc.
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

function exact_pzero_spectrum(L::Int, boundary::Symbol, z2_obc::Symbol;
                              m0::Float64=0.1, eta::Float64=0.5, alpha::Float64=1.0)
    p = Data.Params(; m0=m0, eta=eta, alpha=alpha, Lphys=L, L=L)
    H_mpo, sites = build_H(p; boundary=boundary, gauge_law=:z2,
                           z2_obc_boundary=z2_obc, bg_left=-1, bg_right=-1)
    Hmat = mpo_to_array(H_mpo, sites)
    @assert size(Hmat, 1) == size(Hmat, 2)
    N = size(Hmat, 1)
    @printf("\n[%s, z2_obc=%s]  Hilbert dim = %d\n", boundary, z2_obc, N)

    F = eigen(Hermitian(Hmat))
    energies = F.values         # ascending
    states   = F.vectors        # column k = eigenstate k (k zero-indexed: states[:,1] = ψ_0)

    O_mpo = make_O_pzero_mpo(sites, L, boundary)
    Omat  = mpo_to_array(O_mpo, sites)

    psi0  = states[:, 1]
    Pvec  = Omat * psi0
    Pnorm2 = real(dot(Pvec, Pvec))

    # ⟨ψ_k | P⟩ for all k, then squared.
    amps     = states' * Pvec
    overlaps = abs2.(amps)
    E0       = energies[1]

    return (energies = energies,
            overlaps = overlaps,
            E0       = E0,
            Pnorm2   = Pnorm2,
            N        = N)
end

function truncated_barE(energies, overlaps, E0, k_max::Int)
    upper = min(k_max, length(overlaps) - 1)
    sum_w  = 0.0
    sum_Ew = 0.0
    for k in 1:upper
        w = overlaps[k+1]
        sum_w  += w
        sum_Ew += (energies[k+1] - E0) * w
    end
    barE = sum_w > 0 ? sum_Ew / sum_w : NaN
    return barE, sum_w
end

# Allow L to be set from environment or default to 4.
const L_RUN = parse(Int, get(ENV, "L_EXACT", "4"))
println("\n##### EXACT DIAGONALIZATION at L=$(L_RUN) #####")

# Run for both boundaries
for (boundary, z2_obc, label) in [(:open_site, :truncate_xz, "OBC :truncate_xz, bg=(-1,-1)"),
                                   (:PBC,       :drop,         "PBC")]
    res = exact_pzero_spectrum(L_RUN, boundary, z2_obc;
                               m0=0.1, eta=0.5, alpha=1.0)
    full_W = sum(res.overlaps[2:end])  # all k≥1
    println("\n========== ", label, " ==========")
    @printf("  Hilbert dim = %d,  ‖P‖² = %.6f,  Σ_k|⟨k|O|0⟩|² = %.6f (should equal ‖P‖²)\n",
            res.N, res.Pnorm2, sum(res.overlaps))
    @printf("  W_full (Σ over k≥1) = %.6f,  W₀ = |⟨0|O|0⟩|² = %.6f\n",
            full_W, res.overlaps[1])

    println()
    @printf("  %-12s %-12s %-12s %-14s\n", "k_max", "̄gap", "W_total", "W/W_full")
    sample_ks = sort(unique([5, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, res.N - 1]))
    sample_ks = filter(k -> k <= res.N - 1, sample_ks)
    for k_max in sample_ks
        barE, W = truncated_barE(res.energies, res.overlaps, res.E0, k_max)
        @printf("  k_max=%-3d    %-10.5f   %-10.5f   %-10.4f\n",
                k_max, barE, W, W/full_W)
    end

    # Also report the largest individual overlaps for context.
    perm = sortperm(res.overlaps, rev=true)
    println("\n  Top 10 overlaps (k, gap, |⟨k|O|0⟩|², cum. W/W_full):")
    cum = 0.0
    for (rank, k_idx) in enumerate(perm[1:min(10, length(perm))])
        k = k_idx - 1
        w = res.overlaps[k_idx]
        gap = res.energies[k_idx] - res.E0
        cum += (k == 0 ? 0.0 : w)
        @printf("    rank %2d:  k=%-4d  gap=%9.5f  |⟨k|O|0⟩|²=%.6f  cum_W=%.4f\n",
                rank, k, gap, w, cum / full_W)
    end
end
