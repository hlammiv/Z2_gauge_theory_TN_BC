#!/usr/bin/env python3
"""
Plot gap_lo(nsw) trajectories per L for OBC and PBC, overlaid.
Shows directly which L are converged and which are still climbing.
The PBC plateau target (~1.466) and OBC small-L approach are visible.
"""
import re, csv, glob, math, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("/home/hlamm/Desktop/QC/circuit_knitting/data")
KMAX = 30
BAND_CUT = 2.0

def compute_gap_lo(row):
    num, den = 0.0, 0.0
    for k in range(1, KMAX+1):
        gk = float(row.get(f"gap_k{k}", 0.0))
        wk = float(row.get(f"overlap_k{k}", 0.0))
        if gk < BAND_CUT and wk > 0:
            num += wk*gk; den += wk
    return num/den if den > 0 else None

raw = {}  # (L, bdy) -> {nsw: [glo values across seeds]}
for f in glob.glob(str(DATA / "**" / "z2_gauge_theory" / "convergence_overlap_pzero_*.csv"),
                   recursive=True):
    bn = Path(f).name
    mL = re.search(r"L(\d+)[_D]", bn)
    if not mL: continue
    L = int(mL.group(1))
    m_nsw = re.search(r"(?:nsw|eff)(\d+)", bn)
    if not m_nsw: continue
    nsw = int(m_nsw.group(1))
    m_seed = re.search(r"seed(\d+)", bn)
    seed = int(m_seed.group(1)) if m_seed else 0
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if int(r["L"]) != L: continue
            if r["boundary"] == "PBC":
                bdy = "PBC"
            elif r["boundary"] == "open_site" and r["z2_obc_boundary"] == "truncate_xz":
                bdy = "OBC"
            else: continue
            if L == 18 and bdy == "OBC" and seed == 3: continue
            glo = compute_gap_lo(r)
            if glo is None: continue
            raw.setdefault((L,bdy), {}).setdefault(nsw, []).append(glo)

# Aggregate per (L, bdy, nsw) → mean ± SEM
agg = {}
for key, byn in raw.items():
    agg[key] = []
    for nsw in sorted(byn):
        vs = byn[nsw]
        m = sum(vs)/len(vs)
        sd = math.sqrt(sum((x-m)**2 for x in vs)/(len(vs)-1)) if len(vs)>1 else 0
        sem = sd/math.sqrt(len(vs)) if len(vs)>1 else 0
        agg[key].append((nsw, m, sem, len(vs)))

# Plot
fig, (axO, axP) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

Ls = sorted({k[0] for k in agg if k[0] >= 8})
cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(Ls)))

for ax, bdy, title in [(axO, "OBC", "OBC truncate_xz"), (axP, "PBC", "PBC drop")]:
    for i, L in enumerate(Ls):
        if (L, bdy) not in agg: continue
        series = agg[(L, bdy)]
        ns = [s[0] for s in series]; vs = [s[1] for s in series]; es = [s[2] for s in series]
        ax.errorbar(ns, vs, yerr=es, fmt="o-", color=cmap[i],
                    label=f"L={L} (#nsw={len(ns)})",
                    capsize=3, markersize=6, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("nsw (per-run budget, warm-started)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

axO.set_ylabel(r"$\overline{\rm gap}_{\rm lo}$  (meson-band 1st moment)")
# Reference: PBC asymptote
for ax in (axO, axP):
    ax.axhline(1.466, color="gray", ls=":", lw=1, alpha=0.6)
    ax.text(ax.get_xlim()[1]*0.5, 1.4665, "PBC plateau ~1.466",
            color="gray", fontsize=8, va="bottom", ha="right")

fig.suptitle(r"gap$_{\rm lo}$ convergence vs nsw, gauge only "
             r"($m_0=0.1,\ \eta=0.5,\ \alpha=1$, bg$=(-1,-1)$)", fontsize=12)
fig.tight_layout()
out_png = DATA / "gaplo_trajectories.png"
out_pdf = DATA / "gaplo_trajectories.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}")

# Print summary
print(f"\n{'L':>3} {'bdy':>3} {'best_nsw':>9} {'gap_lo':>9}±{'sem':>6}  {'last drift':>10}")
for key in sorted(agg):
    s = agg[key]
    last = s[-1]
    prev = s[-2] if len(s) >= 2 else None
    drift = (last[1] - prev[1]) if prev else float("nan")
    print(f"{key[0]:>3} {key[1]:>3} {last[0]:>9} {last[1]:>9.5f}±{last[2]:.5f}  {drift:>+10.5f}")
