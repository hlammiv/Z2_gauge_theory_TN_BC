#!/usr/bin/env python3
"""Plot ̄E vs nsweeps for each L, separately for OBC and PBC.

Combines all available nsw data:
  L=8:  nsw=80 (1 seed, prod) + nsw=160 (4 seeds)
  L=10: nsw=80 (1 seed, prod) + nsw=160 (4 seeds) + nsw=320 (4 seeds)
  L=12: nsw=80 (4 seeds) + nsw=160 (4 seeds) + nsw=320 (4 seeds)
  L=14: nsw=80 (4 seeds) + nsw=160 (4 seeds) + nsw=320 (4 seeds)
  L=16: nsw=80 (4 seeds, split run) + nsw=160 (4 seeds) + nsw=320 (4 seeds)
"""
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data"

# Data layout per (L, nsw, code): dir + glob pattern.
# Use None for "single-seed prod CSV file".
DATASETS = {
    (8,  80):  {"gauge": ("prod_file", DATA/"z2_gauge_theory/convergence_overlap_pzero_prod_maxdim400_nsw80_seed1.csv"),
                "claude":("prod_file", DATA/"z2_claude/convergence_overlap_pzero_prod_maxdim400_nsw80_seed1.csv")},
    (8,  160): {"gauge": ("dir", DATA/"L8_nsw160/z2_gauge_theory"),
                "claude":("dir", DATA/"L8_nsw160/z2_claude")},
    (10, 80):  {"gauge": ("prod_file", DATA/"z2_gauge_theory/convergence_overlap_pzero_prod_maxdim400_nsw80_seed1.csv"),
                "claude":("prod_file", DATA/"z2_claude/convergence_overlap_pzero_prod_maxdim400_nsw80_seed1.csv")},
    (10, 160): {"gauge": ("dir", DATA/"L10_nsw160/z2_gauge_theory"),
                "claude":("dir", DATA/"L10_nsw160/z2_claude")},
    (10, 320): {"gauge": ("dir", DATA/"L10_nsw320/z2_gauge_theory"),
                "claude":("dir", DATA/"L10_nsw320/z2_claude")},
    (12, 80):  {"gauge": ("dir", DATA/"lenore_prod_final/z2_gauge_theory"),
                "claude":("dir", DATA/"lenore_prod_final/z2_claude")},
    (12, 160): {"gauge": ("dir", DATA/"L12_nsw160/z2_gauge_theory"),
                "claude":("dir", DATA/"L12_nsw160/z2_claude")},
    (12, 320): {"gauge": ("dir", DATA/"L12_nsw320/z2_gauge_theory"),
                "claude":("dir", DATA/"L12_nsw320/z2_claude")},
    (14, 80):  {"gauge": ("dir", DATA/"lenore_prod_final/z2_gauge_theory"),
                "claude":("dir", DATA/"lenore_prod_final/z2_claude")},
    (14, 160): {"gauge": ("dir", DATA/"L14_nsw160/z2_gauge_theory"),
                "claude":("dir", DATA/"L14_nsw160/z2_claude")},
    (14, 320): {"gauge": ("dir", DATA/"L14_nsw320/z2_gauge_theory"),
                "claude":("dir", DATA/"L14_nsw320/z2_claude")},
    (16, 80):  {"gauge": ("dir", DATA/"L16/z2_gauge_theory"),
                "claude":("dir", DATA/"L16/z2_claude")},
    (16, 160): {"gauge": ("dir", DATA/"L16_nsw160/z2_gauge_theory"),
                "claude":("dir", DATA/"L16_nsw160/z2_claude")},
    (16, 320): {"gauge": ("dir", DATA/"L16_nsw320/z2_gauge_theory"),
                "claude":("dir", DATA/"L16_nsw320/z2_claude")},
}

def is_obc(r):
    return r["boundary"] == "open_site" and r["z2_obc_boundary"] == "truncate_xz"
def is_pbc(r):
    return r["boundary"] == "PBC"

def load_csv_for_L(path, L_target):
    out = {"OBC": [], "PBC": []}
    for r in csv.DictReader(path.open()):
        try:
            L = int(r["L"])
        except (ValueError, KeyError):
            continue
        if L != L_target: continue
        rec = (float(r["barE_gap"]), float(r["sigma_barE"]))
        if is_obc(r):
            out["OBC"].append(rec)
        elif is_pbc(r):
            out["PBC"].append(rec)
    return out

def load_dataset(spec, L_target):
    """Returns {boundary: list of (mean, sigma) per seed}."""
    kind, path = spec
    raw = {"OBC": [], "PBC": []}
    if kind == "prod_file":
        raw = load_csv_for_L(path, L_target)
    elif kind == "dir":
        for f in sorted(path.glob("*.csv")):
            this = load_csv_for_L(f, L_target)
            for k, vs in this.items():
                raw[k].extend(vs)
    return raw

def aggregate(raw):
    """Per-boundary: combine seeds → (mean, sigma_combined)."""
    out = {}
    for k, vals in raw.items():
        if not vals: continue
        gs = [v[0] for v in vals]; ss = [v[1] for v in vals]
        n = len(gs); m = sum(gs)/n
        seed_std = math.sqrt(sum((x-m)**2 for x in gs)/(n-1)) if n > 1 else 0.0
        sem = seed_std/math.sqrt(n) if n > 1 else 0.0
        sper_mean = sum(ss)/n
        comb = math.sqrt(sem**2 + (sper_mean/math.sqrt(n))**2) if n > 1 else sper_mean
        out[k] = (m, comb, n)
    return out

# Pull everything: results[(L, nsw, code)][bdy] = (mean, sigma, nseeds)
results = {}
for (L, nsw), specs in DATASETS.items():
    for code in ("gauge", "claude"):
        spec = specs.get(code)
        if spec is None: continue
        raw = load_dataset(spec, L)
        agg = aggregate(raw)
        if agg:
            results[(L, nsw, code)] = agg

# Build figure.
fig, (ax_obc, ax_pbc) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
Ls = sorted({L for (L, _, _) in results if L >= 12})
cmap = plt.colormaps.get_cmap("viridis")
colors = {L: cmap(i / max(1, len(Ls)-1)) for i, L in enumerate(Ls)}

for L in Ls:
    for code, fmt, mfc_open in [("gauge", "o-", False), ("claude", "s--", True)]:
        nsws = sorted({nsw for (LL, nsw, cc) in results if LL == L and cc == code})
        for ax, bdy in [(ax_obc, "OBC"), (ax_pbc, "PBC")]:
            xs, ys, es = [], [], []
            for nsw in nsws:
                key = (L, nsw, code)
                if bdy in results[key]:
                    m, s, n = results[key][bdy]
                    xs.append(nsw); ys.append(m); es.append(s)
            if not xs: continue
            label = f"L={L} ({code})"
            mfc = "none" if mfc_open else colors[L]
            ax.errorbar(xs, ys, yerr=es, fmt=fmt, color=colors[L],
                        capsize=3, markersize=6, mfc=mfc,
                        linewidth=1.4, label=label)

for ax, ttl in [(ax_obc, "OBC  (open, :truncate_xz, bg=-1,-1)"),
                (ax_pbc, "PBC  (:drop)")]:
    ax.set_ylabel(r"$\bar M$  (spectral 1st moment)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(ttl, fontsize=10.5)
    ax.legend(fontsize=8, ncol=2, loc="best")

ax_pbc.set_xlabel("nsweeps", fontsize=11)
ax_pbc.set_xscale("log")
ax_pbc.set_xticks([80, 160, 320])
ax_pbc.set_xticklabels(["80", "160", "320"])

fig.suptitle(r"$\bar M$ vs nsweeps  ($m_0=0.1$, $\eta=0.5$, $\alpha=1$, $k_{\max}=30$, $\chi_{\max}=300$)",
             fontsize=11)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = DATA / f"nsw_convergence.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

# Console summary.
print()
print(f"{'L':>3} {'nsw':>4} {'code':>7} {'n':>3}   {'OBC':>20s}   {'PBC':>20s}")
for (L, nsw, code) in sorted(results.keys()):
    obc = results[(L, nsw, code)].get("OBC")
    pbc = results[(L, nsw, code)].get("PBC")
    obc_s = f"{obc[0]:.5f}±{obc[1]:.4f}" if obc else "    --"
    pbc_s = f"{pbc[0]:.5f}±{pbc[1]:.4f}" if pbc else "    --"
    n_s = f"{obc[2] if obc else (pbc[2] if pbc else 0)}"
    print(f"{L:>3} {nsw:>4} {code:>7} {n_s:>3}   {obc_s:>20s}   {pbc_s:>20s}")
