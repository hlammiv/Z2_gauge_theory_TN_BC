#!/usr/bin/env python3
"""Plot finite-volume convergence with energy-variance error bars."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/hlamm/Desktop/QC/circuit_knitting/data")

def load(csv_path):
    rows = list(csv.DictReader(csv_path.open()))
    def collect(bdy, obc):
        sel = [r for r in rows if r["boundary"] == bdy and (obc is None or r["z2_obc_boundary"] == obc)]
        sel.sort(key=lambda r: int(r["L"]))
        # Use spectral first moment: barE_gap with sigma_barE.
        return ([int(r["L"]) for r in sel],
                [float(r["barE_gap"])   for r in sel],
                [float(r["sigma_barE"]) for r in sel])
    return collect("open_site", "truncate_xz"), collect("PBC", None)

(L_obc_g, g_obc_g, s_obc_g), (L_pbc_g, g_pbc_g, s_pbc_g) = load(
    DATA / "z2_gauge_theory/convergence_overlap_pzero_bg-1-1_m0p1_eta0p5_kmax20_smoke.csv")
(L_obc_c, g_obc_c, s_obc_c), (L_pbc_c, g_pbc_c, s_pbc_c) = load(
    DATA / "z2_claude/convergence_overlap_pzero_bg-1-1_m0p1_eta0p5_kmax20_smoke.csv")

# For the difference panel: use z2-gauge values (both codes agree within σ).
L_obc, g_obc, s_obc = L_obc_g, g_obc_g, s_obc_g
L_pbc, g_pbc, s_pbc = L_pbc_g, g_pbc_g, s_pbc_g

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})

# Slight x-offset between the two implementations so points don't overlap visually.
def shift(Ls, dx): return [L + dx for L in Ls]
ax.errorbar(shift(L_obc_g, -0.08), g_obc_g, yerr=s_obc_g, fmt="o-", color="tab:blue",
            capsize=4, markersize=8, label="OBC z2-gauge-theory")
ax.errorbar(shift(L_obc_c, +0.08), g_obc_c, yerr=s_obc_c, fmt="o--", color="tab:cyan",
            capsize=4, markersize=7, mfc="none", label="OBC z2-claude")
ax.errorbar(shift(L_pbc_g, -0.08), g_pbc_g, yerr=s_pbc_g, fmt="s-", color="tab:red",
            capsize=4, markersize=8, label="PBC z2-gauge-theory")
ax.errorbar(shift(L_pbc_c, +0.08), g_pbc_c, yerr=s_pbc_c, fmt="s--", color="tab:orange",
            capsize=4, markersize=7, mfc="none", label="PBC z2-claude")
ax.set_ylabel("M  =  ̄gap (spectral first moment via O_p=0)", fontsize=12)
ax.set_title("Finite-volume convergence at (m₀=0.1, η=0.5, α=1, bg=(−1,−1))\n"
             "Spectral 1st moment, k_max=20, smoke DMRG (nsweeps_ex=20, maxdim=400)", fontsize=11)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)

diffs    = [a - b for a, b in zip(g_obc, g_pbc)]
err_diff = [(a*a + b*b)**0.5 for a, b in zip(s_obc, s_pbc)]
ax2.errorbar(L_obc, diffs, yerr=err_diff, fmt="d-", color="black", capsize=4, markersize=7)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xlabel("L  (matter sites)", fontsize=12)
ax2.set_ylabel("OBC − PBC", fontsize=12)
ax2.set_xticks([4, 6, 8, 10, 12, 14])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ("png", "pdf"):
    out = DATA / f"fv_convergence_m0p1_eta0p5_kmax20_smoke.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
