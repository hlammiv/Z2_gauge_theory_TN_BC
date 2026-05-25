###############################################################################
# Test refinement gating at L=10 with saved nsw=640 states.
# Three runs:
#   A) Cold start (baseline, full DMRG on all states)
#   B) Warm-start, no gating (refines all states)
#   C) Warm-start with refinement gating (refines only high-overlap states)
# Compare wall time + final ̄M.
###############################################################################
using Printf

const STATES_G = joinpath(@__DIR__, "data", "states_L10_nsw640", "gauge")
const STATES_C = joinpath(@__DIR__, "data", "states_L10_nsw640", "claude")
const L_TEST = 10
const NSW    = 160  # short test
const KMAX   = 30

# Gauge
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

println("\n===== gauge: A) cold start =====")
t_A = @elapsed convergence_overlap_pzero(;
    Larr=[L_TEST], k_max=KMAX, m0=0.1, eta=0.5, bg_left=-1, bg_right=-1,
    nsweeps=NSW, seed=1, fname_suffix="_gating_A_cold")
@printf("\n>>> gauge A elapsed: %.1fs\n", t_A)

println("\n===== gauge: B) warm, no gating =====")
t_B = @elapsed convergence_overlap_pzero(;
    Larr=[L_TEST], k_max=KMAX, m0=0.1, eta=0.5, bg_left=-1, bg_right=-1,
    nsweeps=NSW, seed=1, warm_start_dir=STATES_G,
    fname_suffix="_gating_B_warm")
@printf("\n>>> gauge B elapsed: %.1fs\n", t_B)

println("\n===== gauge: C) warm + gating =====")
t_C = @elapsed convergence_overlap_pzero(;
    Larr=[L_TEST], k_max=KMAX, m0=0.1, eta=0.5, bg_left=-1, bg_right=-1,
    nsweeps=NSW, seed=1, warm_start_dir=STATES_G,
    refinement_gating=true, gating_overlap_threshold=0.001,
    fname_suffix="_gating_C_gated")
@printf("\n>>> gauge C elapsed: %.1fs\n", t_C)
@printf("Speedups vs cold:  warm = %.2fx, gated = %.2fx\n", t_A/t_B, t_A/t_C)

# Claude
include(joinpath(@__DIR__, "z2-claude.jl"))

println("\n\n===== claude: A) cold =====")
t_cA = @elapsed convergence_overlap_pzero(;
    N_list=[L_TEST], K=KMAX, m=0.1, η=0.5, bg_left=-1, bg_right=-1,
    nsweeps_gs=NSW, nsweeps_ex=NSW, seed=1, fname_suffix="_gating_A_cold")
@printf("\n>>> claude A elapsed: %.1fs\n", t_cA)

println("\n===== claude: B) warm, no gating =====")
t_cB = @elapsed convergence_overlap_pzero(;
    N_list=[L_TEST], K=KMAX, m=0.1, η=0.5, bg_left=-1, bg_right=-1,
    nsweeps_gs=NSW, nsweeps_ex=NSW, seed=1, warm_start_dir=STATES_C,
    fname_suffix="_gating_B_warm")
@printf("\n>>> claude B elapsed: %.1fs\n", t_cB)

println("\n===== claude: C) warm + gating =====")
t_cC = @elapsed convergence_overlap_pzero(;
    N_list=[L_TEST], K=KMAX, m=0.1, η=0.5, bg_left=-1, bg_right=-1,
    nsweeps_gs=NSW, nsweeps_ex=NSW, seed=1, warm_start_dir=STATES_C,
    refinement_gating=true, gating_overlap_threshold=0.001,
    fname_suffix="_gating_C_gated")
@printf("\n>>> claude C elapsed: %.1fs\n", t_cC)
@printf("Speedups vs cold:  warm = %.2fx, gated = %.2fx\n", t_cA/t_cB, t_cA/t_cC)
