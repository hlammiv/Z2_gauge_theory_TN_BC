#!/usr/bin/env python3
"""Combine production single-seed (L=4..10) with 4-seed average (L=12,14)."""
import csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/hlamm/Desktop/QC/circuit_knitting/data")

# Production single-seed CSV paths
PROD = {
    "z2-gauge":  DATA/"z2_gauge_theory/convergence_overlap_pzero_prod_maxdim400_nsw80_seed1.csv",
    "z2-claude": DATA/"z2_claude/convergence_overlap_pzero_prod_maxdim400_nsw80_seed1.csv",
}

# 4-seed lenore CSVs at L=12, 14 (final production with opt 1+3 settings)
LEN = {
    "z2-gauge":  DATA/"lenore_prod_final/z2_gauge_theory",
    "z2-claude": DATA/"lenore_prod_final/z2_claude",
}
# L=16 from 4 seeds on lenore at nsweeps=320 (best convergence)
LEN16 = {
    "z2-gauge":  DATA/"L16_nsw320/z2_gauge_theory",
    "z2-claude": DATA/"L16_nsw320/z2_claude",
}
# L=8 from 4 seeds (2 local + 2 lenore) at nsweeps=160
LEN8 = {
    "z2-gauge":  DATA/"L8_nsw160/z2_gauge_theory",
    "z2-claude": DATA/"L8_nsw160/z2_claude",
}
# L=10 from 4 seeds on lenore at nsweeps=320 (tighter convergence test)
LEN10 = {
    "z2-gauge":  DATA/"L10_nsw320/z2_gauge_theory",
    "z2-claude": DATA/"L10_nsw320/z2_claude",
}
# L=12 from 4 seeds on lenore at nsweeps=320 (both codes)
LEN12 = {
    "z2-gauge":  DATA/"L12_nsw320/z2_gauge_theory",
    "z2-claude": DATA/"L12_nsw320/z2_claude",
}
# L=14 from 4 seeds on lenore at nsweeps=320 (both codes)
LEN14 = {
    "z2-gauge":  DATA/"L14_nsw320/z2_gauge_theory",
    "z2-claude": DATA/"L14_nsw320/z2_claude",
}

def load_prod(path, L_list=(4, 6)):
    """Return {(L, bdy): (gap, sigma)}."""
    out = {}
    for r in csv.DictReader(path.open()):
        L = int(r["L"])
        if L not in L_list: continue
        if r["boundary"] == "PBC" or (r["boundary"]=="open_site" and r["z2_obc_boundary"]=="truncate_xz"):
            out[(L, r["boundary"])] = (float(r["barE_gap"]), float(r["sigma_barE"]))
    return out

def load_4seed(dirpath, L_list=(14,)):
    """Return {(L, bdy): (mean, total_sigma)} averaging across 4 seeds."""
    raw = {}
    for f in sorted(dirpath.glob("*.csv")):
        for r in csv.DictReader(f.open()):
            L = int(r["L"])
            if L not in L_list: continue
            if r["boundary"] == "PBC" or (r["boundary"]=="open_site" and r["z2_obc_boundary"]=="truncate_xz"):
                key = (L, r["boundary"])
                raw.setdefault(key, []).append((float(r["barE_gap"]), float(r["sigma_barE"])))
    out = {}
    for key, vals in raw.items():
        gs = [v[0] for v in vals]; ss = [v[1] for v in vals]
        n = len(gs)
        m = sum(gs)/n
        seed_std = math.sqrt(sum((x-m)**2 for x in gs)/(n-1)) if n>1 else 0
        sem = seed_std/math.sqrt(n)
        sper_mean = sum(ss)/n
        # combined: SEM-of-mean from spread + per-seed DMRG residual / √N
        comb = math.sqrt(sem**2 + (sper_mean/math.sqrt(n))**2)
        out[key] = (m, comb)
    return out

# Combine
data = {code: {} for code in PROD}
for code in PROD:
    data[code].update(load_prod(PROD[code]))
    data[code].update(load_4seed(LEN8[code],  L_list=(8,)))
    data[code].update(load_4seed(LEN10[code], L_list=(10,)))
    data[code].update(load_4seed(LEN12[code], L_list=(12,)))
    data[code].update(load_4seed(LEN14[code], L_list=(14,)))
    data[code].update(load_4seed(LEN16[code], L_list=(16,)))

# Code-averaged numbers w/ inter-code spread folded in as systematic.
combined = {}
all_keys = sorted(set().union(*[d.keys() for d in data.values()]))
for (L, bdy) in all_keys:
    g_val, g_sig = data["z2-gauge"][(L, bdy)]
    c_val, c_sig = data["z2-claude"][(L, bdy)]
    mean = (g_val + c_val) / 2
    sig_stat = math.sqrt(g_sig**2 + c_sig**2) / 2
    sig_sys  = abs(g_val - c_val) / 2
    sig_tot  = math.sqrt(sig_stat**2 + sig_sys**2)
    combined[(L, bdy)] = (mean, sig_tot)

# Plot.
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})

# Per-code curves (no inter-code combining yet) for plotting.
def filt_code(code, bdy):
    keys = sorted([k for k in data[code] if k[1] == bdy])
    return [k[0] for k in keys], [data[code][k][0] for k in keys], [data[code][k][1] for k in keys]

def shift(L, dx): return [x + dx for x in L]

# z2-gauge (solid markers, slight -x offset)
L_g_obc, g_g_obc, s_g_obc = filt_code("z2-gauge", "open_site")
L_g_pbc, g_g_pbc, s_g_pbc = filt_code("z2-gauge", "PBC")
ax.errorbar(shift(L_g_obc, -0.08), g_g_obc, yerr=s_g_obc, fmt="o-", color="tab:blue",
            capsize=4, markersize=8, label="OBC z2-gauge-theory")
ax.errorbar(shift(L_g_pbc, -0.08), g_g_pbc, yerr=s_g_pbc, fmt="s-", color="tab:red",
            capsize=4, markersize=8, label="PBC z2-gauge-theory")

# z2-claude (open markers, +x offset, dashed)
L_c_obc, g_c_obc, s_c_obc = filt_code("z2-claude", "open_site")
L_c_pbc, g_c_pbc, s_c_pbc = filt_code("z2-claude", "PBC")
ax.errorbar(shift(L_c_obc, +0.08), g_c_obc, yerr=s_c_obc, fmt="o--", color="tab:cyan",
            capsize=4, markersize=7, mfc="none", label="OBC z2-claude")
ax.errorbar(shift(L_c_pbc, +0.08), g_c_pbc, yerr=s_c_pbc, fmt="s--", color="tab:orange",
            capsize=4, markersize=7, mfc="none", label="PBC z2-claude")
ax.set_ylabel(r"$\bar M = \bar{\rm gap}$  (spectral 1st moment, $O_{p=0}$)", fontsize=12)
ax.set_title(r"FV at $(m_0=0.1,\ \eta=0.5,\ \alpha=1,\ bg=(-1,-1))$" + "\n"
             r"$k_{max}=30$, maxdim=400, nsweeps=320 ($L=10,12,14,16$) / 160 ($L=8$); 4 seeds at $L \geq 8$",
             fontsize=11)
ax.legend(loc="center right", fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

# Difference panel
Ls_diff, diffs, errs = [], [], []
for L in sorted({k[0] for k in combined}):
    if (L, "open_site") in combined and (L, "PBC") in combined:
        o, so = combined[(L, "open_site")]; p, sp = combined[(L, "PBC")]
        Ls_diff.append(L); diffs.append(o-p); errs.append(math.sqrt(so**2 + sp**2))

ax2.errorbar(Ls_diff, diffs, yerr=errs, fmt="d-", color="black", capsize=4, markersize=7)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xlabel("L  (matter sites)", fontsize=12)
ax2.set_ylabel("OBC − PBC", fontsize=12)
ax2.set_xticks(Ls_diff)
ax2.grid(True, alpha=0.3)

# Extrapolation fits to L → ∞.  Include ALL L=4..16 points.
# L=4 is exact diag with σ=0 → apply a σ-floor of 1e-3 so it enters χ²
# with a sane weight.  Same floor applied uniformly (only affects L=4 since
# L=6 already has σ ~ 4e-3).
SIGMA_FLOOR = 1e-3
_arr = [(L, d, max(e, SIGMA_FLOOR)) for (L, d, e) in zip(Ls_diff, diffs, errs)]
fit_Ls   = np.array([t[0] for t in _arr], dtype=float)
fit_d    = np.array([t[1] for t in _arr])
fit_e    = np.array([t[2] for t in _arr])
# Show fit curves across the full plot range (L=4..18).
Lplot    = np.linspace(4, 18, 200)

def _f_invL(L, a, b):           return a + b/L
def _f_invL2(L, a, b):          return a + b/L**2
def _f_invL_plus_invL2(L, a, b, c): return a + b/L + c/L**2
def _f_invL_beta(L, a, b, beta):    return a + b/L**beta
def _f_exp(L, a, b, m):         return a + b*np.exp(-m*L)

fit_specs = [
    ("$a + b/L$",            _f_invL,            [0.0, -0.3],         "tab:purple"),
    ("$a + b/L^2$",          _f_invL2,           [0.0, -2.0],         "tab:olive"),
    ("$a + b/L + c/L^2$",    _f_invL_plus_invL2, [0.0, -0.3, -0.5],   "tab:green"),
    ("$a + b/L^\\beta$",     _f_invL_beta,       [0.0, -0.3, 1.0],    "tab:pink"),
    ("$a + b e^{-mL}$",      _f_exp,             [0.0, -0.3, 0.2],    "tab:brown"),
]

def _numerical_jac(fn, x, popt, eps_frac=1e-6):
    """∂f/∂p evaluated at popt, returns J of shape (len(x), len(popt))."""
    J = np.zeros((len(x), len(popt)))
    for j, p in enumerate(popt):
        h = max(abs(p) * eps_frac, 1e-10)
        pj_plus  = popt.copy(); pj_plus[j]  = p + h
        pj_minus = popt.copy(); pj_minus[j] = p - h
        J[:, j] = (fn(x, *pj_plus) - fn(x, *pj_minus)) / (2*h)
    return J

fit_summaries = []
for label, fn, p0, color in fit_specs:
    try:
        popt, pcov = curve_fit(fn, fit_Ls, fit_d, sigma=fit_e, p0=p0,
                               absolute_sigma=True, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        # Goodness-of-fit stats
        resid = (fit_d - fn(fit_Ls, *popt)) / fit_e
        chi2_val = float((resid**2).sum())
        npar = len(popt)
        ndata = len(fit_Ls)
        dof = max(1, ndata - npar)
        p_val = float(chi2_dist.sf(chi2_val, dof)) if dof > 0 else float("nan")
        # AIC / BIC for Gaussian errors with known σ (constant log-likelihood term
        # drops out, so we use the relative form 2k + χ² / k log(N) + χ²).
        aic = 2*npar + chi2_val
        bic = npar*math.log(ndata) + chi2_val
        # Fit curve and 1σ confidence band via parameter-covariance propagation.
        ycurve = fn(Lplot, *popt)
        J = _numerical_jac(fn, Lplot, popt)
        var_y = np.einsum("ij,jk,ik->i", J, pcov, J)
        sigma_y = np.sqrt(np.clip(var_y, 0, None))
        ax2.plot(Lplot, ycurve, "-", color=color, linewidth=1.4, alpha=0.85,
                 label=f"{label}  $a_\\infty={popt[0]:+.4f}\\pm{perr[0]:.4f}$  "
                       f"$\\chi^2/{dof}={chi2_val:.2f}$  p={p_val:.2f}")
        ax2.fill_between(Lplot, ycurve - sigma_y, ycurve + sigma_y,
                         color=color, alpha=0.15, linewidth=0)
        fit_summaries.append({
            "label": label, "popt": popt, "perr": perr,
            "chi2": chi2_val, "dof": dof, "p": p_val, "aic": aic, "bic": bic,
            "ndata": ndata, "npar": npar,
        })
    except Exception as e:
        print(f"  fit {label} failed: {e}")

ax2.legend(loc="lower right", fontsize=7.5)
ax2.set_xlim(3.5, 18.5)

print()
print("Extrapolation fits to L → ∞ (L ≥ 8 only):")
print(f"  {'model':22s}  {'a∞':>10s}  {'σ_a':>10s}  {'χ²':>6s}  dof  {'p':>5s}    AIC     BIC")
aic_min = min(s["aic"] for s in fit_summaries) if fit_summaries else 0
bic_min = min(s["bic"] for s in fit_summaries) if fit_summaries else 0
for s in fit_summaries:
    print(f"  {s['label']:22s}  {s['popt'][0]:+10.5f}  {s['perr'][0]:10.5f}  "
          f"{s['chi2']:6.2f}  {s['dof']:>3d}  {s['p']:5.2f}  "
          f"ΔAIC={s['aic']-aic_min:5.2f}  ΔBIC={s['bic']-bic_min:5.2f}")
# Annotate sigma values
for L, d, e in zip(Ls_diff, diffs, errs):
    sig = abs(d) / e
    ax2.annotate(f"{sig:.1f}σ", (L, d), textcoords="offset points", xytext=(8, -10),
                 fontsize=9, color="dimgray")

plt.tight_layout()
for ext in ("png", "pdf"):
    out = DATA / f"fv_convergence_final.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

print()
print("Summary:")
for L in sorted({k[0] for k in combined}):
    o, so = combined[(L, "open_site")]; p, sp = combined[(L, "PBC")]
    d = o-p; e = math.sqrt(so**2+sp**2); sig = abs(d)/e
    print(f"  L={L:>2}  OBC={o:.5f}±{so:.4f}  PBC={p:.5f}±{sp:.4f}  diff={d:+.4f}±{e:.4f}  ({sig:.1f}σ)")
