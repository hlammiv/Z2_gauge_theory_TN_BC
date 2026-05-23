# Z₂ gauge theory — Boundary conditions & circuit knitting

DMRG codes and analysis used in the paper *Periodic Boundary Conditions for
Lattice Field Theories with Circuit Knitting* (FERMILAB-PUB-XX-XXX-T).  The
target system is 1+1d Z₂ lattice gauge theory with Kogut–Susskind staggered
fermions; we compute the spectral first moment of the meson dispersion (̄M)
on OBC vs PBC chains to extract finite-volume corrections.

## DMRG implementations

Two *independently developed* DMRG drivers, kept epistemically separated so
they can validate each other:

- `z2-gauge-theory.jl` — main implementation (~2,100 lines).  Solver:
  `solve_z2_higgs_k` for ground + excited states via subspace projection,
  driver `convergence_overlap_pzero` for the full FV scan.
- `z2-claude.jl`  — parallel implementation (~1,600 lines).  Solver:
  `compute_excited_states`, same driver name `convergence_overlap_pzero`.

Both expose a `convergence_overlap_pzero(...)` entry point that writes per-(L,
boundary, z2_obc) CSV rows containing the spectral moment ̄M, per-state
overlaps with the zero-momentum meson operator, and per-state DMRG energy
variance σ_k.

Both use chunked early-stopping in DMRG.  The `_dmrg_maybe_early`
(gauge) / `_dmrg_chunked` (claude) helpers stop when |ΔE| between
`chunk_sweeps`-sized chunks falls below `early_stop_tol` (default 1e-7),
after at least `minsweeps` (default 20) sweeps.

`helper_functions.jl` — tiny shared helper (`mpo_to_array`) for exact
diagonalisation at small L.

## Validation / tests

- `exact_diag_L4.jl`        — exact diagonalisation driver, validates DMRG at small L.
- `dmrg_vs_exact_L4.jl`     — direct DMRG-vs-exact comparison at L=4.
- `test_skip_unphysical_L8.jl` — skip-state benchmark; confirmed no unphysical
  states leak through at our parameter point.
- `qn_test_L6.jl`, `qn_vs_noqn_test.jl` — Z₂ matter-parity QN conservation
  benchmark.  Conclusion: QN-aware DMRG is ~2× *slower* per state at our bond
  dimensions due to ITensors block-sparse overhead.
- `test_observer_vs_chunked.jl` — A/B test comparing per-sweep
  `DMRGObserver`-style early-stop against chunked-restart early-stop.
  Conclusion: chunked-restart is ~1.8× faster per sweep despite using more
  sweeps to reach the same |ΔE| tolerance — hence its use in production.

## Plotting

- `plot_fv_combined.py`     — production figure: ̄M(L) for OBC and PBC, plus
  OBC−PBC vs L with 5 extrapolation fits (1/L, 1/L², 1/L+1/L², 1/L^β,
  exp(−mL)), 1σ confidence bands, χ²/dof, AIC/BIC stats.
- `plot_nsw_convergence.py` — ̄M vs nsweeps for each L, both boundaries.  Used
  to verify DMRG sweep convergence.
- `plot_fv.py`, `plot_fv_seeds.py`, `plot_fv.jl` — older variants kept for
  reference.

## Reproducing a single run

```bash
julia --project=. -e '
  ENV["Z2GT_BLAS_THREADS"]="4"
  include("z2-gauge-theory.jl")
  convergence_overlap_pzero(; Larr=[10], k_max=30,
      m0=0.1, eta=0.5, bg_left=-1, bg_right=-1,
      nsweeps=320, seed=1, fname_suffix="_L10_demo")
'
```

Writes `data/z2_gauge_theory/convergence_overlap_pzero_L10_demo.csv`.  Same
arguments for `z2-claude.jl` except: `N_list=[L]` instead of `Larr=[L]`,
`K=k_max` instead of `k_max`, `m=m0` and `η=eta` instead of `m0`/`eta`,
`nsweeps_gs`/`nsweeps_ex` instead of `nsweeps`.

## Parameters used in the paper

Reference point: `m₀=0.1, η=0.5, α=1, bg=(-1,-1)`, `k_max=30`, `maxdim=300`,
`cutoff=1e-11`, `nsweeps=320` (`L=10,12,14,16`) / `nsweeps=160` (`L=8`).
4 random-init seeds averaged per (L, boundary, code).

## BLAS threading

The codes read `Z2GT_BLAS_THREADS` (gauge) / `Z2C_BLAS_THREADS` (claude) from
the environment at startup and set `BLAS.set_num_threads(...)` accordingly.
Default is `Sys.CPU_THREADS ÷ 2`.
