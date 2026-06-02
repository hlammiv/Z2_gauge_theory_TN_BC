#!/usr/bin/env python3
"""
Extrapolate gauge gap_lo to nsw→∞ per (L, boundary) and plot vs L.

For each (L, boundary):
  1) Scan all gauge CSVs, compute gap_lo per (nsw, seed)
  2) Average seeds → gap_lo(nsw)
  3) Fit gap_lo(nsw) = M∞ - a/nsw   AND   M∞ - a·exp(-nsw/τ)
  4) Take M∞ as the median across viable models; sigma from model spread

L without a usable nsw series falls back to the single (highest-nsw) point.
"""
import re, csv, glob, math, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

DATA = Path("/home/hlamm/Desktop/QC/circuit_knitting/data")
KMAX = 30
BAND_CUT = 2.0

def compute_gap_lo(row):
    num, den, var = 0.0, 0.0, 0.0
    for k in range(1, KMAX+1):
        gk = float(row.get(f"gap_k{k}", 0.0))
        wk = float(row.get(f"overlap_k{k}", 0.0))
        sk = float(row.get(f"sigma_k{k}", 0.0))
        if gk < BAND_CUT and wk > 0:
            num += wk*gk; den += wk; var += (wk*sk)**2
    if den <= 0: return None, None
    return num/den, (var**0.5)/den

# Scan all gauge CSVs for L, nsw, boundary, seed
def scan_gauge():
    out = {}  # (L, bdy) -> {nsw: [(gap_lo, sigma)]}
    for f in glob.glob(str(DATA / "**" / "z2_gauge_theory" / "convergence_overlap_pzero_*.csv"),
                       recursive=True):
        bn = Path(f).name
        mL = re.search(r"L(\d+)[_D]", bn)
        if not mL: continue
        L = int(mL.group(1))
        m_nsw = re.search(r"(?:nsw|eff)(\d+)", bn)
        if not m_nsw: continue
        nsw = int(m_nsw.group(1))
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if int(r["L"]) != L: continue
                if r["boundary"] == "PBC":
                    bdy = "PBC"
                elif r["boundary"] == "open_site" and r["z2_obc_boundary"] == "truncate_xz":
                    bdy = "OBC"
                else:
                    continue
                # drop L=18 seed-3 OBC basin artifact
                m_seed = re.search(r"seed(\d+)", bn)
                seed = int(m_seed.group(1)) if m_seed else 0
                if L == 18 and bdy == "OBC" and seed == 3: continue
                glo, sig = compute_gap_lo(r)
                if glo is None: continue
                out.setdefault((L, bdy), {}).setdefault(nsw, []).append((glo, sig))
    return out

def f_inv(n, M, a):      return M - a/n
def f_invp(n, M, a, p):  return M - a/n**p
def f_exp(n, M, a, tau): return M - a*np.exp(-n/tau)

# Physical bound on gap_lo: it can't realistically exceed ~1.48 in the meson band.
M_HARD_MAX = 1.475

def fit_models(nsws, vals, errs):
    out = {}
    n = np.array(nsws, float); y = np.array(vals, float); e = np.array(errs, float)
    e = np.where(e>0, e, 1e-5)
    span = max(y.max()-y.min(), 1e-5)
    M_lo, M_hi = y[-1] - 0.005, M_HARD_MAX
    def keep(popt, pcov, name):
        M, sig = popt[0], math.sqrt(max(pcov[0,0], 0))
        if not math.isfinite(M) or not math.isfinite(sig): return
        if M >= M_HARD_MAX - 1e-4: return  # hit the wall
        if sig > 0.05: return              # unconstrained
        out[name] = (M, sig)
    try:
        popt, pcov = curve_fit(f_inv, n, y, p0=[y[-1]+span, span*n[0]],
                               sigma=e, maxfev=200000,
                               bounds=([M_lo,0],[M_hi,10]))
        keep(popt, pcov, "inv")
    except: pass
    if len(n) >= 4:
        try:
            popt, pcov = curve_fit(f_invp, n, y, p0=[y[-1]+span, span, 0.8],
                                   sigma=e, maxfev=200000,
                                   bounds=([M_lo,0,0.1],[M_hi,1e6,3]))
            keep(popt, pcov, "invp")
        except: pass
        try:
            popt, pcov = curve_fit(f_exp, n, y, p0=[y[-1]+span, span, n[len(n)//2]],
                                   sigma=e, maxfev=200000,
                                   bounds=([M_lo,0,10],[M_hi,10,1e5]))
            keep(popt, pcov, "exp")
        except: pass
    return out

data = scan_gauge()

# For each (L, bdy), aggregate seeds per nsw, then fit nsw→∞
extrap = {}  # (L, bdy) -> (gap_lo_inf, sigma, n_nsw, source)
print(f"\n{'L':>3} {'bdy':>3} {'#nsw':>4} {'highest nsw':>11} {'last gaplo':>10}  "
      f"{'inv_M':>8} {'invpM':>8} {'exp_M':>8}  {'M_inf':>9}±{'sig':>6}")
for key in sorted(data):
    L, bdy = key
    series = data[key]
    nsws = sorted(series)
    means, stds = [], []
    for nsw in nsws:
        vs = [v[0] for v in series[nsw]]
        m = sum(vs)/len(vs)
        s = (sum((x-m)**2 for x in vs)/(len(vs)-1))**0.5 if len(vs)>1 else 0.0
        means.append(m); stds.append(s if s>0 else 1e-5)
    # quick convergence check: last 2-doubling drift very small → just use highest nsw
    last_drift = abs(means[-1] - means[-2]) if len(means) >= 2 else 1.0
    converged = (last_drift < 0.0005)

    fits = fit_models(nsws, means, stds) if len(nsws) >= 3 else {}
    Ms = [v[0] for v in fits.values()]

    if converged and len(nsws) >= 2:
        # PBC-style: already converged within ±0.0005, take last value
        sig = max(stds[-1], 0.0005)
        extrap[key] = (means[-1], sig, len(nsws), "conv")
        tag = "(conv)"
    elif Ms:
        Minf = float(np.median(Ms))
        # sigma: model-spread half-width + max single-fit sigma
        spread = (max(Ms)-min(Ms))/2 if len(Ms) > 1 else 0.0
        fit_sig = max((v[1] for v in fits.values()), default=0.0)
        sig = (fit_sig**2 + spread**2)**0.5
        extrap[key] = (Minf, sig, len(nsws), "fit")
        tag = "(fit)"
    else:
        # fallback: highest-nsw with seed-spread sigma
        extrap[key] = (means[-1], max(stds[-1], 0.001), len(nsws), "raw")
        tag = "(raw)"

    inv_M = fits.get("inv",(float("nan"),0))[0]
    invp_M = fits.get("invp",(float("nan"),0))[0]
    exp_M = fits.get("exp",(float("nan"),0))[0]
    print(f"{L:>3} {bdy:>3} {len(nsws):>4} {nsws[-1]:>11} {means[-1]:>10.5f}  "
          f"{inv_M:>8.5f} {invp_M:>8.5f} {exp_M:>8.5f}  {extrap[key][0]:>9.5f}±{extrap[key][1]:.5f} {tag}")

# Plot
fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True,
                        gridspec_kw={"height_ratios":[2.2, 1]})
ax, ax2 = axs

for bdy, color, marker in [("OBC","tab:blue","o"), ("PBC","tab:red","s")]:
    Ls = sorted({k[0] for k in extrap if k[1]==bdy})
    vs = [extrap[(L,bdy)][0] for L in Ls]
    es = [extrap[(L,bdy)][1] for L in Ls]
    ax.errorbar(Ls, vs, yerr=es, fmt=f"{marker}-", color=color, capsize=4,
                markersize=8, label=f"{bdy} (gauge, nsw→∞ extrap)")

ax.set_ylabel(r"$\overline{\rm gap}_{\rm lo}^{(\rm nsw\to\infty)}$", fontsize=12)
ax.set_title(r"Extrapolated gap$_{\rm lo}$ vs $L$ "
             r"($m_0=0.1,\ \eta=0.5,\ \alpha=1,\ bg=(-1,-1)$, gauge only)", fontsize=11)
ax.grid(True, alpha=0.3); ax.legend()

# OBC − PBC
Ls_diff, diffs, errs = [], [], []
for L in sorted({k[0] for k in extrap}):
    if (L,"OBC") in extrap and (L,"PBC") in extrap:
        o, so = extrap[(L,"OBC")][:2]; p, sp = extrap[(L,"PBC")][:2]
        Ls_diff.append(L); diffs.append(o-p); errs.append((so**2+sp**2)**0.5)
ax2.errorbar(Ls_diff, diffs, yerr=errs, fmt="d-", color="black", capsize=4, markersize=7)
ax2.axhline(0, color="gray", ls="--", lw=0.8)
ax2.set_xlabel("L  (matter sites)", fontsize=12)
ax2.set_ylabel("OBC − PBC", fontsize=12)
ax2.grid(True, alpha=0.3)

# 1/L fit on OBC-PBC
Larr = np.array(Ls_diff, float); darr = np.array(diffs, float); earr = np.array(errs, float)
if len(Larr) >= 3:
    try:
        popt, pcov = curve_fit(lambda L,a,b: a + b/L, Larr, darr,
                               sigma=earr, absolute_sigma=False, maxfev=50000)
        Lplot = np.linspace(3, 22, 200)
        ax2.plot(Lplot, popt[0]+popt[1]/Lplot, "g-", lw=1.2,
                 label=rf"$a+b/L$:  $a_\infty={popt[0]:+.4f}\pm{np.sqrt(pcov[0,0]):.4f}$")
        ax2.legend(fontsize=9)
    except Exception as e:
        print("fit failed:", e)

plt.tight_layout()
out_png = DATA / "fv_convergence_gaplo_extrap.png"
out_pdf = DATA / "fv_convergence_gaplo_extrap.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\nwrote {out_png}")
