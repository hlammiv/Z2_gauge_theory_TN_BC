#!/usr/bin/env python3
"""Combine 4-seed runs into mean ± std plots."""
import csv
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/hlamm/Desktop/QC/circuit_knitting/data")

CODES = {
    "z2-gauge": (DATA / "z2_gauge_theory",
                 "convergence_overlap_pzero_bg-1-1_m0p1_eta0p5_k30_seed{seed}.csv"),
    "z2-claude": (DATA / "z2_claude",
                  "convergence_overlap_pzero_bg-1-1_m0p1_eta0p5_k30_seed{seed}.csv"),
}

def load_seeds(dir_, pattern, seeds=(1, 2, 3, 4)):
    """Return dict (L, boundary) -> list of (barE, sigma, W) across seeds."""
    out = {}
    for seed in seeds:
        path = dir_ / pattern.format(seed=seed)
        rows = list(csv.DictReader(path.open()))
        for r in rows:
            bdy = r["boundary"]
            obc = r["z2_obc_boundary"]
            if not ((bdy == "PBC") or (bdy == "open_site" and obc == "truncate_xz")):
                continue
            L = int(r["L"])
            key = (L, bdy)
            out.setdefault(key, []).append((float(r["barE_gap"]),
                                            float(r["sigma_barE"]),
                                            float(r["W_total"])))
    return out

def mean_std(vals):
    n = len(vals)
    if n == 0: return float("nan"), float("nan")
    m = sum(vals) / n
    var = sum((v - m)**2 for v in vals) / max(1, n - 1)
    return m, math.sqrt(var)

# Load both codes.
data = {code: load_seeds(d, p) for code, (d, p) in CODES.items()}

# For each (L, boundary, code), compute seed-mean and seed-std.
# Combine "uncertainty per seed" σ_per_seed and seed-to-seed σ_spread
# in quadrature to get total error.
print("L  bdy        code        mean      seed_std  σ̄_per_seed  total_σ  W_mean")
combined = {}
for code, d in data.items():
    for (L, bdy), entries in sorted(d.items()):
        gaps  = [e[0] for e in entries]
        sigs  = [e[1] for e in entries]
        Ws    = [e[2] for e in entries]
        m_gap, s_gap = mean_std(gaps)
        sig_mean = sum(sigs) / len(sigs)
        # SEM-of-mean from the seed average:
        sem_seed = s_gap / math.sqrt(len(gaps))
        # Combined error: SEM of mean (statistical) plus mean per-seed DMRG residual.
        sig_total = math.sqrt(sem_seed**2 + sig_mean**2)
        w_mean = sum(Ws) / len(Ws)
        combined.setdefault(code, {})[(L, bdy)] = (m_gap, sig_total, w_mean, s_gap, sem_seed, sig_mean)
        print(f"{L:2d}  {bdy:10s}  {code:11s} {m_gap:.5f}   {s_gap:.5f}   {sig_mean:.5f}    "
              f"{sig_total:.5f}   {w_mean:.4f}")

# Plot.
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})

def shift(L, dx): return L + dx
COLORS = {"OBC": {"z2-gauge": "tab:blue",  "z2-claude": "tab:cyan"},
          "PBC": {"z2-gauge": "tab:red",   "z2-claude": "tab:orange"}}
MARKER = {"OBC": "o", "PBC": "s"}
DASH   = {"z2-gauge": "-", "z2-claude": "--"}

for bdy_label, bdy_key in [("OBC", "open_site"), ("PBC", "PBC")]:
    for ci, code in enumerate(("z2-gauge", "z2-claude")):
        keys = sorted([k for k in combined[code].keys() if k[1] == bdy_key])
        Ls   = [k[0] for k in keys]
        Ls_s = [shift(L, -0.08 if ci == 0 else +0.08) for L in Ls]
        m    = [combined[code][k][0] for k in keys]
        e    = [combined[code][k][1] for k in keys]
        mfc  = None if ci == 0 else "none"
        ax.errorbar(Ls_s, m, yerr=e,
                    fmt=f"{MARKER[bdy_label]}{DASH[code]}",
                    color=COLORS[bdy_label][code],
                    capsize=4, markersize=8, mfc=mfc,
                    label=f"{bdy_label} {code}")

ax.set_ylabel(r"$\bar M = \bar{\rm gap}$  (spectral 1st moment, O$_{p=0}$)", fontsize=12)
ax.set_title(r"FV at $(m_0=0.1,\ \eta=0.5,\ \alpha=1,\ bg=(-1,-1))$" + "\n"
             r"$k_{max}=30$, 4-seed mean $\pm$ combined error",
             fontsize=11)
ax.legend(loc="lower right", fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

# Difference panel: OBC - PBC, averaged across codes per-L.
Ls_diff, diffs, errs = [], [], []
all_L = sorted({k[0] for c in combined.values() for k in c.keys()})
for L in all_L:
    obc_vals, obc_errs, pbc_vals, pbc_errs = [], [], [], []
    for code in ("z2-gauge", "z2-claude"):
        if (L, "open_site") in combined[code]:
            obc_vals.append(combined[code][(L, "open_site")][0])
            obc_errs.append(combined[code][(L, "open_site")][1])
        if (L, "PBC") in combined[code]:
            pbc_vals.append(combined[code][(L, "PBC")][0])
            pbc_errs.append(combined[code][(L, "PBC")][1])
    if obc_vals and pbc_vals:
        # Code-averaged mean (treating both codes as samples of same quantity).
        obc_mean = sum(obc_vals)/len(obc_vals)
        pbc_mean = sum(pbc_vals)/len(pbc_vals)
        # Conservative: sum-in-quadrature of mean errors + inter-code spread.
        obc_spread = max(obc_vals) - min(obc_vals) if len(obc_vals) > 1 else 0
        pbc_spread = max(pbc_vals) - min(pbc_vals) if len(pbc_vals) > 1 else 0
        obc_err = math.sqrt((sum(e**2 for e in obc_errs)/len(obc_errs)) + (obc_spread/2)**2)
        pbc_err = math.sqrt((sum(e**2 for e in pbc_errs)/len(pbc_errs)) + (pbc_spread/2)**2)
        Ls_diff.append(L)
        diffs.append(obc_mean - pbc_mean)
        errs.append(math.sqrt(obc_err**2 + pbc_err**2))

ax2.errorbar(Ls_diff, diffs, yerr=errs, fmt="d-", color="black", capsize=4, markersize=7)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xlabel("L  (matter sites)", fontsize=12)
ax2.set_ylabel("OBC − PBC", fontsize=12)
ax2.set_xticks(all_L)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ("png", "pdf"):
    out = DATA / f"fv_convergence_k30_4seeds.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
