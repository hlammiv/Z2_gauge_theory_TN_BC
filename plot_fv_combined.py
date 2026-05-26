#!/usr/bin/env python3
"""Combine production single-seed (L=4..10) with 4-seed average (L=12,14)."""
import csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist, f as f_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Empirical convergence drift per (L, boundary), measured from the most recent
# nsweeps doubling we have.  DMRG variational bias is one-sided (energies and
# gaps drift UP with more nsweeps), so we treat this as a *one-sided* systematic:
#   σ_+ on ̄M = drift size  (̄M could be this much higher with full convergence)
#   σ_- on ̄M = 0           (we never observe downward drift)
# Other systematics (k_max, maxdim) are smaller and lumped into an L-dependent
# floor:
#   L=4  -> 0      (exact diagonalisation; no DMRG/k_max/maxdim systematics)
#   L=6  -> 0.001  (DMRG present but bond dim and k_max truncation tiny)
#   L≥8  -> 0.003  (real maxdim+k_max systematics)
def _sys_floor(L):
    if L <= 4: return 0.0
    if L <= 6: return 0.001
    return 0.003
EMPIRICAL_DRIFT_UP = {
    # L → {boundary → drift size}.  Most-recent nsweeps doubling per L:
    #   L=8:  nsw=160 → 640
    #   L=10: nsw=320 → 640
    #   L=12,14: nsw=160 → 320 (nsw=640 still running)
    #   L=16: nsw=320 → 640
    8:  {"OBC": 0.00002, "PBC": 0.00076},  # nsw=160→640
    10: {"OBC": 0.00052, "PBC": 0.00142},  # nsw=320→640
    12: {"OBC": 0.00125, "PBC": 0.00367},  # nsw=320→640
    14: {"OBC": 0.00151, "PBC": 0.00220},  # nsw=320→640
    16: {"OBC": 0.0051,  "PBC": 0.0063},   # nsw=320→640
}

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
# L=16 from 4 seeds on lenore at nsweeps=640 (most converged)
LEN16 = {
    "z2-gauge":  DATA/"L16_nsw640/z2_gauge_theory",
    "z2-claude": DATA/"L16_nsw640/z2_claude",
}
# L=6 from 4 seeds local at nsweeps=640 (tighter than single-seed prod)
LEN6 = {
    "z2-gauge":  DATA/"L6_nsw640/z2_gauge_theory",
    "z2-claude": DATA/"L6_nsw640/z2_claude",
}
# L=8 from 4 seeds local at nsweeps=640 (most converged)
LEN8 = {
    "z2-gauge":  DATA/"L8_nsw640/z2_gauge_theory",
    "z2-claude": DATA/"L8_nsw640/z2_claude",
}
# L=10 from 4 seeds local at nsweeps=640 (most converged + warm-start states saved)
LEN10 = {
    "z2-gauge":  DATA/"L10_nsw640/z2_gauge_theory",
    "z2-claude": DATA/"L10_nsw640/z2_claude",
}
# L=12: 4 seeds × 2 codes at nsweeps=640 (fully done)
LEN12 = {
    "z2-gauge":  DATA/"L12_nsw640/z2_gauge_theory",
    "z2-claude": DATA/"L12_nsw640/z2_claude",
}
# L=14: 4 seeds × 2 codes at nsweeps=640 (fully done)
LEN14 = {
    "z2-gauge":  DATA/"L14_nsw640/z2_gauge_theory",
    "z2-claude": DATA/"L14_nsw640/z2_claude",
}

def load_prod(path, L_list=(4,)):
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
    data[code].update(load_4seed(LEN6[code],  L_list=(6,)))
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
             r"$k_{max}=30$, maxdim=300, nsweeps=640 ($L=6$–$16$); 4 seeds at $L \geq 8$",
             fontsize=11)
ax.legend(loc="center right", fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

# Difference panel
Ls_diff, diffs, errs = [], [], []
for L in sorted({k[0] for k in combined}):
    if (L, "open_site") in combined and (L, "PBC") in combined:
        o, so = combined[(L, "open_site")]; p, sp = combined[(L, "PBC")]
        Ls_diff.append(L); diffs.append(o-p); errs.append(math.sqrt(so**2 + sp**2))

# Asymmetric error bars on (OBC − PBC), propagated from per-boundary drifts:
#   σ_+ (toward zero, less negative): driven by σ_+(OBC) — if OBC could shift UP
#   σ_- (more negative):              driven by σ_+(PBC) — if PBC could shift UP
# stat σ on the difference is added in quadrature.  SYS_FLOOR (per boundary)
# captures residual maxdim/k_max systematics not measured directly.
err_plus  = []   # toward zero
err_minus = []   # more negative
for L, _d, e_stat in zip(Ls_diff, diffs, errs):
    drift = EMPIRICAL_DRIFT_UP.get(L, {"OBC": 0.0, "PBC": 0.0})
    floor_L = _sys_floor(L)
    sigp_OBC = math.sqrt(drift["OBC"]**2 + floor_L**2)
    sigp_PBC = math.sqrt(drift["PBC"]**2 + floor_L**2)
    err_plus.append(math.sqrt(e_stat**2 + sigp_OBC**2))
    err_minus.append(math.sqrt(e_stat**2 + sigp_PBC**2))
yerr_asym = [err_minus, err_plus]   # matplotlib wants [lower, upper]

ax2.errorbar(Ls_diff, diffs, yerr=yerr_asym, fmt="none", color="gray",
             capsize=6, alpha=0.5, linewidth=1.5,
             label="stat ⊕ asym sys (per-bdy drift)")
ax2.errorbar(Ls_diff, diffs, yerr=errs, fmt="d-", color="black", capsize=4,
             markersize=7, label="stat. only")
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
# For curve_fit, use the LARGER side of the asymmetric error as a conservative
# symmetric σ.  This keeps the fit machinery simple and is honest for points
# where the two sides differ.
_arr = []
for L, d, e_stat, ep, em in zip(Ls_diff, diffs, errs, err_plus, err_minus):
    e_use = max(max(ep, em), SIGMA_FLOOR)
    _arr.append((L, d, e_use))
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

def _fmt_value_sigma(value, sigma):
    """Format as 'value ± σ' with σ rounded to 1 sig fig and value rounded
    to match its decimal place."""
    if not (math.isfinite(sigma) and sigma > 0):
        return f"{value:+.4f}", "0"
    exp = math.floor(math.log10(sigma))
    decimals = max(0, -exp)
    return f"{value:+.{decimals}f}", f"{sigma:.{decimals}f}"

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
        a_str, s_str = _fmt_value_sigma(popt[0], perr[0])
        ax2.plot(Lplot, ycurve, "-", color=color, linewidth=1.4, alpha=0.85,
                 label=f"{label}  $a_\\infty={a_str}\\pm{s_str}$")
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

# ------------------------------------------------------------------------
# Monte-Carlo refit with asymmetric (split-normal) per-point errors.
# For each MC trial, sample each data point from its split-normal distribution
# (σ_+ for upper, σ_- for lower), refit each model, collect a∞.
# After N trials, report the median + 16/84 percentile envelope on a∞.
# Fit weights inside each trial use the symmetric geometric mean σ.
# ------------------------------------------------------------------------
N_MC = 2000
rng = np.random.default_rng(20260525)
fit_d_arr  = np.array(diffs)
fit_ep_arr = np.array(err_plus)
fit_em_arr = np.array(err_minus)
# Symmetric σ to use as curve_fit weights inside the MC loop
fit_e_sym  = np.sqrt(fit_ep_arr * fit_em_arr)
fit_e_sym  = np.maximum(fit_e_sym, SIGMA_FLOOR)

def _split_normal_sample(mu, sigma_minus, sigma_plus, rng):
    """One sample per element of mu (shape N)."""
    p_plus = sigma_plus / (sigma_plus + sigma_minus)
    u  = rng.random(len(mu))
    an = np.abs(rng.standard_normal(len(mu)))
    return np.where(u < p_plus,
                    mu + an * sigma_plus,
                    mu - an * sigma_minus)

mc_results = {s["label"]: [] for s in fit_summaries}
for trial in range(N_MC):
    d_trial = _split_normal_sample(fit_d_arr, fit_em_arr, fit_ep_arr, rng)
    for label, fn, p0, _color in fit_specs:
        if label not in mc_results: continue
        try:
            popt_mc, _ = curve_fit(fn, fit_Ls, d_trial, sigma=fit_e_sym,
                                   p0=p0, absolute_sigma=True, maxfev=10000)
            mc_results[label].append(popt_mc[0])
        except Exception:
            pass

print()
print(f"Monte-Carlo refit ({N_MC} trials, asymmetric per-point errors):")
print(f"  {'model':22s}  {'median':>10s}  {'+1σ':>10s}  {'-1σ':>10s}   n_ok")
for s in fit_summaries:
    label = s["label"]
    samples = np.array(mc_results.get(label, []))
    if len(samples) < 100:
        print(f"  {label:22s}  too few good fits ({len(samples)})")
        continue
    median = float(np.median(samples))
    lo, hi = np.percentile(samples, [16, 84])
    sig_plus  = hi - median
    sig_minus = median - lo
    print(f"  {label:22s}  {median:+10.5f}  {sig_plus:+10.5f}  {sig_minus:+10.5f}   {len(samples):>4d}")

# F-test for nested models: 1/L (k=2) ⊂ 1/L+1/L² (k=3).
# Tests whether the extra parameter in the more-complex model is statistically
# warranted given the reduction in χ².
_by_label = {s["label"]: s for s in fit_summaries}
sA = _by_label.get("$a + b/L$")
sB = _by_label.get("$a + b/L + c/L^2$")
if sA is not None and sB is not None:
    dchi2 = sA["chi2"] - sB["chi2"]
    dpar  = sB["npar"] - sA["npar"]
    dof_B = sB["dof"]
    if dpar > 0 and dof_B > 0 and dchi2 > 0:
        F = (dchi2 / dpar) / (sB["chi2"] / dof_B)
        p_F = float(f_dist.sf(F, dpar, dof_B))
        verdict = "extra parameter NOT warranted" if p_F > 0.05 else "extra parameter warranted"
        print(f"\nF-test (nested): 1/L  vs  1/L + 1/L²")
        print(f"  Δχ²={dchi2:.3f}, Δk={dpar}, F={F:.3f}, p_F={p_F:.3f}  → {verdict}")
    else:
        print(f"\nF-test skipped (Δχ²={dchi2:.3f}, Δk={dpar})")

# Annotate sigma values (use the asymmetric σ in the direction of zero,
# i.e. how many σ_+ away from zero is the data point).
for L, d, ep in zip(Ls_diff, diffs, err_plus):
    sig = abs(d) / ep
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
