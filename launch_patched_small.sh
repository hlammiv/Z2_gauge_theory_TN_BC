#!/bin/bash
# Re-cold-start L=4-12 (gauge, both boundaries, 4 seeds) with patched noise
# schedule + cold_linkdims=10.  Output to *_patched_seedN.csv so we don't
# touch existing data.
set -e
cd /home/hlamm/Desktop/QC/circuit_knitting
mkdir -p logs data/states_patched/gauge

# run_L L THREADS [SEQ_BCS]
# If THREADS>=4 OR SEQ_BCS=1: process boundaries sequentially (4 seeds parallel
# per BC, total 4×THREADS cores at peak).  Else: launch both BCs parallel
# (8 procs total, 8×THREADS cores at peak).
run_L () {
  local L=$1 THR=$2
  local seq_bcs=0
  if [ "$THR" -ge 4 ]; then seq_bcs=1; fi
  echo "$(date) — L=$L (threads/proc=$THR, seq_bcs=$seq_bcs)"

  do_bc () {
    local bc_tuple=$1 bdy_label=$2
    for seed in 1 2 3 4; do
      OPENBLAS_NUM_THREADS=$THR OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR \
        nohup julia --project=. --threads=$THR -e "include(\"z2-gauge-theory.jl\"); convergence_overlap_pzero(; Larr=[$L], boundaries=(:${bc_tuple}), k_max=30, m0=0.1, eta=0.5, bg_left=-1, bg_right=-1, nsweeps=640, seed=$seed, cold_noise=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0], cold_linkdims=10, save_states_dir=\"data/states_patched/gauge\", fname_suffix=\"_L${L}_${bdy_label}_patched_seed$seed\")" \
        > logs/patched_L${L}_${bdy_label}_seed${seed}.log 2>&1 &
    done
  }

  if [ "$seq_bcs" = "1" ]; then
    do_bc 'open_site,' 'open_site'; wait
    do_bc 'PBC,'       'PBC';        wait
  else
    do_bc 'open_site,' 'open_site'
    do_bc 'PBC,'       'PBC'
    wait
  fi
  echo "$(date) — L=$L done"
}

# L=4-8: 2 threads each × 8 procs = 16 cores. Tiny memory.
run_L  4 2
run_L  6 2
run_L  8 2
# L=10, 12: 4 threads each × 4 procs (BCs sequential) = 16 cores, less oversubscription.
run_L 10 4
run_L 12 4

echo "$(date) — all patched cold-starts L=4..12 complete"
