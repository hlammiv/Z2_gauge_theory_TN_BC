###############################################################################
# Diagnostic: re-cold-start L=14 PBC seed 1 with claude-matched noise schedule
# and larger initial bond dim. Compare per-state sigma_k vs original gauge run.
#
# Original gauge schedule: noise = [1e-7, 1e-8, 1e-9, 0.0],  linkdims_init = 2
# Patched schedule:        noise = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0],  linkdims = 10
#
# Expectation: sigma_k drops 5-30x toward claude's level at matched energies.
###############################################################################
using Printf
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

println("\n===== L=14 PBC seed 1 cold-start, patched noise + linkdims =====")
t = @elapsed convergence_overlap_pzero(;
    Larr=[14], boundaries=(:PBC,), k_max=30,
    m0=0.1, eta=0.5, bg_left=-1, bg_right=-1,
    nsweeps=640, seed=1,
    cold_noise=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0],
    cold_linkdims=10,
    fname_suffix="_L14_PBC_diag_noise_seed1")
@printf("\n>>> elapsed: %.1fs\n", t)
