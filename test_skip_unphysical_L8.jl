using ITensors, ITensorMPS, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

const L = 8
p = Data.Params(; m0=0.1, eta=0.5, alpha=1.0, Lphys=L, L=L)
H, sites = build_H(p; boundary=:open_site, gauge_law=:z2,
                   z2_obc_boundary=:truncate_xz, bg_left=-1, bg_right=-1)

# Run BASELINE: no skipping, k_max=30
println("=== BASELINE (k_max=30, no skipping) ===")
Random.seed!(1)
t0 = time()
energies_baseline, psis_baseline = solve_z2_higgs_k(H, sites; k_max=30,
    nsweeps=80, weight=200.0, maxdim_final=300, cutoff=1e-11,
    early_stop_tol=1e-7)
t_baseline = time() - t0
e0 = energies_baseline[1]
gaps = [energies_baseline[k+1] - e0 for k in 0:30]
unphysical_count = sum(g > 10.0 for g in gaps)
@printf("\nBaseline elapsed: %.1fs\n", t_baseline)
@printf("Found %d total states, %d unphysical (gap > 10)\n",
        length(energies_baseline), unphysical_count)
println("\nGap distribution (k → gap to ground):")
for (k, g) in enumerate(gaps)
    tag = g > 10 ? " ← UNPHYSICAL" : ""
    @printf("  k=%-3d  gap=%9.5f%s\n", k-1, g, tag)
end

# Run SKIP MODE: physical_gap_threshold=10, max_attempts=50
println("\n\n=== SKIP MODE (k_max=30, physical_gap_threshold=10, max_attempts=50) ===")
Random.seed!(1)
t0 = time()
energies_skip, psis_skip = solve_z2_higgs_k(H, sites; k_max=30,
    nsweeps=80, weight=200.0, maxdim_final=300, cutoff=1e-11,
    early_stop_tol=1e-7,
    physical_gap_threshold=10.0, max_attempts=50)
t_skip = time() - t0
gaps_skip = [energies_skip[k+1] - energies_skip[1] for k in 0:(length(energies_skip)-1)]
@printf("\nSkip-mode elapsed: %.1fs (vs baseline %.1fs → %.2fx)\n",
        t_skip, t_baseline, t_baseline / t_skip)
@printf("Returned %d physical states\n", length(energies_skip))
println("\nGap distribution (k → gap):")
for (k, g) in enumerate(gaps_skip)
    @printf("  k=%-3d  gap=%9.5f\n", k-1, g)
end
