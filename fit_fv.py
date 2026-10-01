#!/usr/bin/env python3
"""Finite-volume hypothesis tests for the paper's Sec. III.

For the weighted gap Dbar and the summed weight W (see fv_data.py) this script
asks which functional form describes how the open-boundary (OBC) results
approach the periodic-boundary (PBC) value as the number of sites N_s grows:

    y_OBC(N_s) = a + g(N_s)

with g a power series in 1/N_s that starts at first order, one that starts at
second order, a single power N_s^-beta, or an exponential.  Each form is fitted
to N_s = N_min..18; N_s = 24 is held out and predicted.  The limit a is either
tied to the PBC value ("common limit") or left free.

Outputs (under --paper-dir): generated/fv_fit_results.json, three generated
LaTeX tables, and figures/fv_gaplo.pdf.

    python3 fit_fv.py
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

import fv_data

ROOT = Path(__file__).resolve().parent
DEFAULT_PAPER = ROOT.parent / "circuit_knitting" / "paper"

OBSERVABLES = ("Dbar", "W")
OBS_TEX = {"Dbar": r"\overline{\Delta}", "W": "W"}
HELD_OUT = 24
FIT_SIZES = tuple(n for n in fv_data.SIZES if n < HELD_OUT)
N_MIN_VALUES = (4, 6, 8, 10, 12)
REFERENCE_SIZE = 10          # PBC size that defines the infinite-volume value
P_ACCEPT = 0.05

# name -> (LaTeX form of g, powers of 1/N_s if linear, nonlinear function, start values)
MODELS = {
    "P1": (r"b/N_s", (1,), None, None),
    "P2": (r"b/N_s+c/N_s^2", (1, 2), None, None),
    "P3": (r"b/N_s+c/N_s^2+d/N_s^3", (1, 2, 3), None, None),
    "Q2": (r"c/N_s^2+d/N_s^3", (2, 3), None, None),
    "Q3": (r"c/N_s^2+d/N_s^3+e/N_s^4", (2, 3, 4), None, None),
    "PW": (r"b/N_s^{\beta}", None,
           lambda n, b, beta: b / n ** beta, [(0.2, 1.0), (0.5, 1.5), (1.0, 2.0)]),
    "EX": (r"b\,e^{-\kappa N_s}", None,
           lambda n, b, kappa: b * np.exp(-kappa * n), [(0.1, 0.1), (0.2, 0.3), (1.0, 1.0)]),
    "EM": (r"b\,e^{-M N_s}", None, None, None),   # rate pinned to the meson mass
}
SERIES_WITH_1_OVER_N = ("P1", "P2", "P3")


def _linear_fit(columns, y, err, prior):
    """Weighted linear least squares for y = a + sum_j theta_j columns_j."""
    design = np.column_stack([np.ones_like(y)] + columns)
    rows = design / err[:, None]
    target = y / err
    if prior is not None:
        extra = np.zeros(design.shape[1])
        extra[0] = 1.0 / prior[1]
        rows = np.vstack([rows, extra])
        target = np.append(target, prior[0] / prior[1])
    # Column scaling keeps the normal equations well conditioned.
    scale = np.linalg.norm(rows, axis=0)
    solution, *_ = np.linalg.lstsq(rows / scale, target, rcond=None)
    params = solution / scale
    cov = np.linalg.pinv((rows / scale).T @ (rows / scale)) / np.outer(scale, scale)
    chi2 = float(np.sum((rows @ params - target) ** 2))
    return params, cov, chi2


def _nonlinear_fit(func, starts, sizes, y, err, prior, a_start, sign):
    """Least squares for y = a + func(N, *theta), best of several starts."""
    def residuals(p):
        with np.errstate(over="ignore", invalid="ignore"):
            res = (y - p[0] - func(sizes, *p[1:])) / err
        if prior is not None:
            res = np.append(res, (p[0] - prior[0]) / prior[1])
        return res

    best = None
    for start in starts:
        p0 = np.array([a_start, sign * start[0], *start[1:]])
        try:
            sol = least_squares(residuals, p0, x_scale="jac", xtol=1e-15, ftol=1e-15,
                                gtol=1e-15, max_nfev=20000)
        except (ValueError, FloatingPointError):
            continue
        chi2 = float(np.sum(sol.fun ** 2))
        if best is None or chi2 < best[2]:
            cov = np.linalg.pinv(sol.jac.T @ sol.jac)
            best = (sol.x, cov, chi2)
    if best is None:
        raise RuntimeError("nonlinear fit failed from every start value")
    return best


def fit(model, sizes, y, err, prior=None, mass=None):
    """Fit y = a + g(N_s).  prior = (value, sigma) ties a to the PBC value.

    Returns a dict with parameters [a, ...], their covariance, chi2, dof and a
    callable prediction.
    """
    sizes = np.asarray(sizes, float)
    y = np.asarray(y, float)
    err = np.asarray(err, float)
    _, powers, func, starts = MODELS[model]
    if model == "EM":
        columns = [np.exp(-mass * sizes)]
        params, cov, chi2 = _linear_fit(columns, y, err, prior)

        def g(n, theta):
            return theta[0] * np.exp(-mass * np.asarray(n, float))
    elif powers is not None:
        columns = [sizes ** -p for p in powers]
        params, cov, chi2 = _linear_fit(columns, y, err, prior)

        def g(n, theta):
            n = np.asarray(n, float)
            return sum(t * n ** -p for t, p in zip(theta, powers))
    else:
        a_start = prior[0] if prior is not None else y[-1]
        sign = 1.0 if y[0] > y[-1] else -1.0
        params, cov, chi2 = _nonlinear_fit(func, starts, sizes, y, err, prior, a_start, sign)

        def g(n, theta):
            return func(np.asarray(n, float), *theta)

    n_data = len(y) + (1 if prior is not None else 0)
    dof = n_data - len(params)

    def predict(n, with_limit=True):
        """Return (value, sigma) of the fitted curve at size(s) n.

        with_limit=False gives the size-dependent part g(N_s) alone.
        """
        n = np.atleast_1d(np.asarray(n, float))
        offset = 1.0 if with_limit else 0.0
        value = offset * params[0] + g(n, params[1:])
        jac = np.empty((len(n), len(params)))
        for j in range(len(params)):
            step = 1e-6 * max(abs(params[j]), 1e-6)
            shifted = params.copy()
            shifted[j] += step
            jac[:, j] = (offset * shifted[0] + g(n, shifted[1:]) - value) / step
        sigma = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", jac, cov, jac), 0, None))
        return value, sigma

    return {"model": model, "params": params, "cov": cov,
            "errors": np.sqrt(np.clip(np.diag(cov), 0, None)),
            "chi2": chi2, "dof": dof,
            "p": float(chi2_dist.sf(chi2, dof)) if dof > 0 else float("nan"),
            "predict": predict}


def reference(data, obs):
    """Infinite-volume value from PBC: the N_s = REFERENCE_SIZE result.

    Its uncertainty also covers the change from the next smaller size.
    """
    here = data[(REFERENCE_SIZE, "PBC")]
    below = data[(REFERENCE_SIZE - 2, "PBC")]
    return here[obs], math.hypot(here[f"{obs}_err"], here[obs] - below[obs])


def run_grid(data):
    """All fits: observable x {common, free} limit x model x N_min."""
    results = []
    for obs in OBSERVABLES:
        ref = reference(data, obs)
        mass = reference(data, "Dbar")[0]
        y_all = {n: data[(n, "open_site")][obs] for n in fv_data.SIZES}
        e_all = {n: data[(n, "open_site")][f"{obs}_err"] for n in fv_data.SIZES}
        for limit in ("common", "free"):
            for model in MODELS:
                for n_min in N_MIN_VALUES:
                    sizes = [n for n in FIT_SIZES if n >= n_min]
                    n_params = 1 + (1 if model == "EM" else len(MODELS[model][1] or (0, 0)))
                    n_data = len(sizes) + (1 if limit == "common" else 0)
                    if n_data <= n_params:
                        continue
                    try:
                        result = fit(model, sizes, [y_all[n] for n in sizes],
                                     [e_all[n] for n in sizes],
                                     prior=ref if limit == "common" else None, mass=mass)
                    except RuntimeError:
                        continue
                    value, sigma = result["predict"](HELD_OUT)
                    result.update({
                        "obs": obs, "limit": limit, "n_min": n_min, "n_points": len(sizes),
                        "held_out_pred": float(value[0]), "held_out_sigma": float(sigma[0]),
                        "held_out_pull": float((y_all[HELD_OUT] - value[0])
                                               / math.hypot(e_all[HELD_OUT], sigma[0])),
                        "limit_minus_pbc": float(result["params"][0] - ref[0]),
                        "limit_minus_pbc_err": float(math.hypot(result["errors"][0], ref[1])
                                                     if limit == "free" else result["errors"][0]),
                    })
                    results.append(result)
    return results


def select(results, obs, limit, model, n_min):
    for r in results:
        if (r["obs"], r["limit"], r["model"], r["n_min"]) == (obs, limit, model, n_min):
            return r
    return None


def smallest_acceptable(results, obs, limit, model):
    """Smallest N_min at which the model is not rejected, or None."""
    for n_min in N_MIN_VALUES:
        r = select(results, obs, limit, model, n_min)
        if r is not None and r["p"] >= P_ACCEPT:
            return r
    return None


def preferred(results, obs):
    """Lowest-order 1/N_s series describing all sizes from the smallest N_min.

    Sizes are dropped from below only when no series up to third order fits.
    """
    for n_min in N_MIN_VALUES:
        for model in SERIES_WITH_1_OVER_N:
            r = select(results, obs, "common", model, n_min)
            if r is not None and r["p"] >= P_ACCEPT:
                return r
    raise RuntimeError(f"no acceptable 1/N_s series for {obs}")


def effective_exponents(data, obs):
    """beta_eff from successive sizes: |y_OBC - y_PBC| ~ N_s^-beta_eff."""
    ref = reference(data, obs)[0]
    out = []
    for n1, n2 in zip(fv_data.SIZES[:-1], fv_data.SIZES[1:]):
        d1 = abs(data[(n1, "open_site")][obs] - ref)
        d2 = abs(data[(n2, "open_site")][obs] - ref)
        e1 = data[(n1, "open_site")][f"{obs}_err"] / d1
        e2 = data[(n2, "open_site")][f"{obs}_err"] / d2
        log_ratio = math.log(n2 / n1)
        out.append({"sizes": [n1, n2], "beta": math.log(d1 / d2) / log_ratio,
                    "err": math.hypot(e1, e2) / log_ratio})
    return out


# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------
def pm(value, err, sign=False):
    """Format value(err) with the uncertainty to two significant figures."""
    if not (err > 0 and math.isfinite(err)):
        return f"{value:+.6f}" if sign else f"{value:.6f}"
    decimals = 1 - math.floor(math.log10(err))
    if round(err * 10 ** decimals) >= 100:   # rounding pushed the error to three digits
        decimals -= 1
    decimals = max(0, decimals)
    digits = round(err * 10 ** decimals)
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return f"{text}({digits})"


def chi2_tex(chi2, dof):
    if chi2 >= 1000:
        exponent = math.floor(math.log10(chi2))
        return rf"${chi2 / 10 ** exponent:.1f}\times10^{{{exponent}}}/{dof}$"
    if chi2 >= 100:
        return rf"${chi2:.0f}/{dof}$"
    if chi2 >= 10:
        return rf"${chi2:.1f}/{dof}$"
    return rf"${chi2:.2f}/{dof}$"


def pull_tex(pull):
    if abs(pull) >= 100:
        return rf"${pull:+.0f}$"
    return rf"${pull:+.1f}$"


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
HEADER = "% Generated by finite_volume/fit_fv.py; do not edit by hand.\n"


def write_data_table(data, path):
    lines = [HEADER.rstrip(), r"\begin{table*}[t]", r"\centering",
             r"\begin{tabular}{r@{\qquad}l@{\qquad}l@{\qquad}l@{\qquad}l}\hline\hline",
             r"$N_s$ & $\overline{\Delta}_{\rm PBC}$ & $\overline{\Delta}_{\rm OBC}$ "
             r"& $W_{\rm PBC}$ & $W_{\rm OBC}$ \\ \hline"]
    for n in fv_data.SIZES:
        cells = []
        for obs in OBSERVABLES:
            for boundary in ("PBC", "open_site"):
                s = data[(n, boundary)]
                cells.append(pm(s[obs], s[f"{obs}_err"]))
        mark = r"$^{*}$" if n == HELD_OUT else ""
        lines.append(f"{n}{mark} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline\hline", r"\end{tabular}",
              r"\caption{Weighted gap $\overline{\Delta}$ and summed weight $W$ of the "
              r"single-meson band, Eqs.~\eqref{eq:Wdef} and~\eqref{eq:Dbardef}, for periodic "
              r"and open boundary conditions at $(m_0,\eta)=(0.1,0.5)$.  Uncertainties on "
              r"$\overline{\Delta}$ are dominated by bounds from the energy variance of the "
              r"DMRG states and by the change in the last refinement stage, and are not "
              r"statistical; those on $W$ combine the scatter among starting states, the difference "
              r"between the two implementations, and the change in the last refinement stage.  "
              r"The starred size is not used in any fit.}",
              r"\label{tab:fv-data}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def _fit_row_cells(results, obs, model):
    cells = []
    for n_min in (4, 6):
        r = select(results, obs, "common", model, n_min)
        cells.append(chi2_tex(r["chi2"], r["dof"]))
    best = smallest_acceptable(results, obs, "common", model)
    if best is None:
        cells += ["--", "--"]
    else:
        cells += [str(best["n_min"]), pull_tex(best["held_out_pull"])]
    return cells


def write_fit_table(results, path):
    columns = (r"$\chi^2/{\rm dof}$ & $\chi^2/{\rm dof}$ & $N_{\min}^{\rm acc}$ & pull")
    lines = [HEADER.rstrip(), r"\begin{table*}[t]", r"\centering",
             r"\begin{tabular}{l|cccc|cccc}\hline\hline",
             r" & \multicolumn{4}{c|}{$\overline{\Delta}$} & \multicolumn{4}{c}{$W$} \\",
             r" & $N_{\min}=4$ & $N_{\min}=6$ & & at $N_s=24$ "
             r"& $N_{\min}=4$ & $N_{\min}=6$ & & at $N_s=24$ \\",
             rf"$y_{{\rm OBC}}(N_s)-y_{{\infty}}$ & {columns} & {columns} \\ \hline"]
    for model in MODELS:
        cells = []
        for obs in OBSERVABLES:
            cells += _fit_row_cells(results, obs, model)
        lines.append(f"${MODELS[model][0]}$ & " + " & ".join(cells) + r" \\")
        if model in ("P3", "Q3"):
            lines.append(r"\hline")
    lines += [r"\hline\hline", r"\end{tabular}",
              r"\caption{Tests of the functional form with which the open-boundary results "
              r"approach the infinite-volume value $y_\infty$, taken from the periodic "
              r"lattice.  Each form is fitted to $N_{\min}\le N_s\le 18$.  For each observable "
              r"the first two columns give $\chi^2/{\rm dof}$ for $N_{\min}=4$ and $6$; "
              r"$N_{\min}^{\rm acc}$ is the smallest $N_{\min}$ at which the form is not "
              r"rejected at the 5\% level (a dash means it is rejected for every "
              r"$N_{\min}\le 12$); and the pull is the deviation of the measured $N_s=24$ "
              r"value from the prediction of that fit, in units of the combined uncertainty.  "
              r"In the last row the rate is fixed to the meson gap $M=\overline{\Delta}_\infty$.}",
              r"\label{tab:fv-fits}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def write_grid_table(results, path):
    lines = [HEADER.rstrip(), r"\begin{table*}[t]", r"\centering",
             r"\begin{tabular}{ll|ccc|ccc}\hline\hline",
             r" & & \multicolumn{3}{c|}{$\overline{\Delta}$} & \multicolumn{3}{c}{$W$} \\",
             r"form & $N_{\min}$ & $\chi^2/{\rm dof}$ & $y_\infty^{\rm OBC}-y_\infty^{\rm PBC}$ "
             r"& pull & $\chi^2/{\rm dof}$ & $y_\infty^{\rm OBC}-y_\infty^{\rm PBC}$ & pull "
             r"\\ \hline"]
    for model in MODELS:
        first = True
        for n_min in N_MIN_VALUES:
            cells = []
            for obs in OBSERVABLES:
                r = select(results, obs, "free", model, n_min)
                if r is None:
                    cells += ["", "", ""]
                    continue
                cells += [chi2_tex(r["chi2"], r["dof"]),
                          f"${pm(r['limit_minus_pbc'], r['limit_minus_pbc_err'], sign=True)}$",
                          pull_tex(r["held_out_pull"])]
            if not any(cells):
                continue
            label = f"${MODELS[model][0]}$" if first else ""
            first = False
            lines.append(f"{label} & {n_min} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{Fits of $y_{\rm OBC}(N_s)=y_\infty^{\rm OBC}+g(N_s)$ to "
              r"$N_{\min}\le N_s\le 18$ with the limit $y_\infty^{\rm OBC}$ left free, for each "
              r"form $g$ of Table~\ref{tab:fv-fits}.  The fitted limit is given relative to the "
              r"periodic value, and the pull compares the measured $N_s=24$ value with the "
              r"prediction of the fit.}",
              r"\label{tab:fv-grid}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def summary_dict(data, results):
    """Everything the paper text quotes, in one JSON-serialisable dict."""
    out = {"reference_size": REFERENCE_SIZE, "held_out": HELD_OUT,
           "p_accept": P_ACCEPT, "delta_cut": fv_data.DELTA_CUT,
           "data": {}, "reference": {}, "preferred": {}, "effective_exponent": {},
           "fits": []}
    for (n, boundary), s in sorted(data.items()):
        out["data"][f"{n}_{fv_data.LABEL[boundary]}"] = {
            k: v for k, v in s.items() if isinstance(v, (int, float)) or v is None}
    for obs in OBSERVABLES:
        ref = reference(data, obs)
        out["reference"][obs] = {"value": ref[0], "err": ref[1]}
        out["effective_exponent"][obs] = effective_exponents(data, obs)
        best = preferred(results, obs)
        free = select(results, obs, "free", best["model"], best["n_min"])
        out["preferred"][obs] = {
            "model": best["model"], "n_min": best["n_min"],
            "chi2": best["chi2"], "dof": best["dof"], "p": best["p"],
            "params": best["params"].tolist(), "errors": best["errors"].tolist(),
            "held_out_pred": best["held_out_pred"], "held_out_sigma": best["held_out_sigma"],
            "held_out_pull": best["held_out_pull"],
            "free_limit_minus_pbc": free["limit_minus_pbc"],
            "free_limit_minus_pbc_err": free["limit_minus_pbc_err"],
            "free_limit": float(free["params"][0]), "free_limit_err": float(free["errors"][0]),
            "free_chi2": free["chi2"], "free_dof": free["dof"]}
    # How large a lattice each boundary condition needs for a given accuracy
    # on the gap.  OBC from the preferred series; PBC from the data directly.
    ref = reference(data, "Dbar")[0]
    best = preferred(results, "Dbar")
    out["relative_shift_percent"] = {
        fv_data.LABEL[b]: {str(n): 100 * (data[(n, b)]["Dbar"] - ref) / ref
                           for n in fv_data.SIZES} for b in fv_data.BOUNDARIES}
    # Spread of the leading coefficient b over the fits that define the band.
    out["band"] = {}
    for obs in OBSERVABLES:
        members = band_fits(results, obs)
        out["band"][obs] = {
            "members": [{"model": m["model"], "n_min": m["n_min"],
                         "b": float(m["params"][1]), "b_err": float(m["errors"][1])}
                        for m in members],
            "b_low": float(min(m["params"][1] - m["errors"][1] for m in members)),
            "b_high": float(max(m["params"][1] + m["errors"][1] for m in members))}
    out["obc_size_for_accuracy"] = {}
    for target in (0.01, 0.001):
        n = 4
        while abs(best["predict"](n, with_limit=False)[0][0]) / ref > target:
            n += 2
        out["obc_size_for_accuracy"][str(target)] = n
    # Highest level of the OBC band: an individual level converges faster
    # (as 1/N_s^2) than the volume-averaged moments.
    out["obc_top_level"] = {
        str(n): {"gap": data[(n, "open_site")]["top_level"],
                 "n2_times_shift": n ** 2 * (data[(n, "open_site")]["top_level"] - ref),
                 "shift_percent": 100 * (data[(n, "open_site")]["top_level"] - ref) / ref}
        for n in fv_data.SIZES}
    for r in results:
        out["fits"].append({
            "obs": r["obs"], "limit": r["limit"], "model": r["model"], "n_min": r["n_min"],
            "chi2": r["chi2"], "dof": r["dof"], "p": r["p"],
            "params": r["params"].tolist(), "errors": r["errors"].tolist(),
            "held_out_pred": r["held_out_pred"], "held_out_sigma": r["held_out_sigma"],
            "held_out_pull": r["held_out_pull"],
            "limit_minus_pbc": r["limit_minus_pbc"],
            "limit_minus_pbc_err": r["limit_minus_pbc_err"]})
    return out


def band_fits(results, obs):
    """Fits whose spread defines the plotted uncertainty band.

    The preferred series, the series one order lower at the smallest N_min
    where it is acceptable, and the preferred order with the next N_min.  The
    band is the envelope of their one-sigma ranges, so it shows the effect of
    truncating the series and of the choice of N_min as well as the fit
    uncertainty.
    """
    best = preferred(results, obs)
    members = [best]
    order = SERIES_WITH_1_OVER_N.index(best["model"])
    if order > 0:
        lower = smallest_acceptable(results, obs, "common", SERIES_WITH_1_OVER_N[order - 1])
        if lower is not None:
            members.append(lower)
    later = select(results, obs, "common", best["model"], best["n_min"] + 2)
    if later is not None and later["p"] >= P_ACCEPT:
        members.append(later)
    return members


def band_envelope(results, obs, sizes, with_limit):
    """(low, high) envelope of the one-sigma ranges of band_fits at sizes.

    Each fit contributes only at sizes it describes, N_s >= its N_min.
    """
    sizes = np.asarray(sizes, float)
    low = np.full(sizes.shape, np.inf)
    high = np.full(sizes.shape, -np.inf)
    for member in band_fits(results, obs):
        value, sigma = member["predict"](sizes, with_limit=with_limit)
        valid = sizes >= member["n_min"]
        low = np.where(valid, np.minimum(low, value - sigma), low)
        high = np.where(valid, np.maximum(high, value + sigma), high)
    return low, high


def make_figure(data, results, path, scaled=True):
    """Draw the paper figure: three panels sharing the 1/N_s axis.

    Top and middle show the weighted gap and the summed weight for both
    boundary conditions.  The bottom panel shows the OBC-PBC difference of
    both: scaled=True multiplies it by N_s, so that the intercept is the
    1/N_s coefficient; scaled=False shows the plain difference, which goes to
    zero at infinite volume.  Color encodes the boundary condition throughout
    (vermillion PBC, blue OBC).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, vermillion = "#0072B2", "#D55E00"
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Latin Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
        "font.size": 16, "axes.labelsize": 19, "legend.fontsize": 16,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.linewidth": 0.8,
        "errorbar.capsize": 2.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })
    sizes = np.array(fv_data.SIZES, float)
    inverse = 1.0 / sizes
    # All panels share the 1/N_s axis, so the infinite-volume limit is the
    # left edge of each.  N_s itself is labelled along the top.
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(6.4, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.07})

    def draw_observable(axis, obs, ylabel, ylim, legend_loc):
        """One observable for both boundary conditions, with the OBC fit."""
        ref = reference(data, obs)[0]
        axis.axhline(ref, color=vermillion, linewidth=1.0, linestyle="--", zorder=1)
        best = preferred(results, obs)
        grid_inv = np.linspace(0.0, 1.0 / best["n_min"], 400)[1:]
        low, high = band_envelope(results, obs, 1.0 / grid_inv, with_limit=True)
        axis.fill_between(grid_inv, low, high, color=blue, alpha=0.3, linewidth=0)
        axis.plot(grid_inv, best["predict"](1.0 / grid_inv)[0], color=blue, linewidth=1.2,
                  zorder=2)
        # Filled: used in the OBC fit (all measured PBC points except the
        # held-out size are filled).
        masks = {"PBC": sizes < HELD_OUT,
                 "open_site": (sizes >= best["n_min"]) & (sizes < HELD_OUT)}
        for boundary, color, marker, label, offset in (
                ("PBC", vermillion, "s", "PBC", 0.0015),
                ("open_site", blue, "o", "OBC", -0.0015)):
            y = np.array([data[(int(n), boundary)][obs] for n in sizes])
            e = np.array([data[(int(n), boundary)][f"{obs}_err"] for n in sizes])
            filled = masks[boundary]
            axis.errorbar(inverse[filled] + offset, y[filled], yerr=e[filled], fmt=marker,
                          color=color, markersize=6.5, label=label, zorder=4)
            axis.errorbar(inverse[~filled] + offset, y[~filled], yerr=e[~filled], fmt=marker,
                          color=color, markersize=6.5, mfc="white", mew=1.3, zorder=4)
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.grid(alpha=0.15)
        if legend_loc:
            axis.legend(loc=legend_loc, frameon=False)

    draw_observable(top, "Dbar", r"$\overline{\Delta}$", (1.405, 1.482), "lower left")
    draw_observable(middle, "W", r"$W$", (0.658, 0.724), None)
    size_axis = top.secondary_xaxis("top")
    labelled = (4, 6, 8, 10, 12, 16, 24)
    size_axis.set_xticks([0.0] + [1.0 / n for n in labelled])
    size_axis.set_xticklabels([r"$\infty$"] + [str(n) for n in labelled])
    size_axis.set_xlabel(r"$N_s$")

    # Bottom: the OBC-PBC difference of both observables on one axis.  Both
    # are open-boundary results, so both are blue; the marker tells them apart.
    # W is drawn first so that the gap point at N_s = 4, which nearly coincides
    # with the W point there, stays visible on top of it.
    for obs, sign, marker, label, layer in (("W", 1.0, "D", r"$y=W$", 3),
                                            ("Dbar", -1.0, "o", r"$y=\overline{\Delta}$", 5)):
        ref = reference(data, obs)[0]
        y = np.array([data[(int(n), "open_site")][obs] for n in sizes])
        e = np.array([data[(int(n), "open_site")][f"{obs}_err"] for n in sizes])
        factor = sizes if scaled else np.ones_like(sizes)
        values, errors = sign * factor * (y - ref), factor * e
        best = preferred(results, obs)
        fitted = (sizes >= best["n_min"]) & (sizes < HELD_OUT)
        # Curve and band are drawn only where the fit applies, N_s >= N_min.
        grid_inv = np.linspace(0.0, 1.0 / best["n_min"], 400)[1:]
        grid_factor = 1.0 / grid_inv if scaled else 1.0
        low, high = band_envelope(results, obs, 1.0 / grid_inv, with_limit=False)
        edges = np.sort(np.vstack([sign * low * grid_factor, sign * high * grid_factor]), axis=0)
        bottom.fill_between(grid_inv, edges[0], edges[1], color=blue, alpha=0.3, linewidth=0)
        series, _ = best["predict"](1.0 / grid_inv, with_limit=False)
        bottom.plot(grid_inv, sign * series * grid_factor, color=blue, linewidth=1.2, zorder=2)
        bottom.errorbar(inverse[fitted], values[fitted], yerr=errors[fitted], fmt=marker,
                        color=blue, markersize=6, label=label, zorder=layer + 1)
        bottom.errorbar(inverse[~fitted], values[~fitted], yerr=errors[~fitted], fmt=marker,
                        color=blue, markersize=8 if obs == "W" else 6, mfc="white", mew=1.3,
                        zorder=layer)
    bottom.set_xlabel(r"$1/N_s$")
    bottom.set_xlim(0.0, 0.26)
    if scaled:
        bottom.set_ylabel(r"$N_s\,|y_{\rm OBC}-y_\infty|$")
        bottom.set_ylim(0.09, 0.27)
    else:
        bottom.set_ylabel(r"$|y_{\rm OBC}-y_\infty|$")
        bottom.set_ylim(0.0, 0.058)
    handles, labels = bottom.get_legend_handles_labels()
    bottom.legend(handles[::-1], labels[::-1], frameon=False,
                  loc="lower right" if scaled else "upper left")
    bottom.grid(alpha=0.15)

    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def print_report(data, results):
    for obs in OBSERVABLES:
        ref = reference(data, obs)
        print(f"\n=== {obs}: PBC reference {pm(*ref)} ===")
        for limit in ("common", "free"):
            print(f"  -- limit {limit} --")
            for model in MODELS:
                for n_min in N_MIN_VALUES:
                    r = select(results, obs, limit, model, n_min)
                    if r is None:
                        continue
                    params = " ".join(pm(v, e, sign=True)
                                      for v, e in zip(r["params"][1:], r["errors"][1:]))
                    print(f"  {model} Nmin={n_min:2d} chi2/dof={r['chi2']:12.3g}/{r['dof']} "
                          f"p={r['p']:.3f} limit-PBC={pm(r['limit_minus_pbc'], r['limit_minus_pbc_err'], sign=True)} "
                          f"[{params}] pull24={r['held_out_pull']:+.1f}")
        best = preferred(results, obs)
        print(f"  preferred: {best['model']} with N_min={best['n_min']}")
        print("  beta_eff:", " ".join(f"({x['sizes'][0]},{x['sizes'][1]}):{pm(x['beta'], x['err'])}"
                                      for x in effective_exponents(data, obs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    data = fv_data.load_all()
    results = run_grid(data)
    print_report(data, results)

    generated = args.paper_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "fv_fit_results.json").write_text(
        json.dumps(summary_dict(data, results), indent=1) + "\n")
    write_data_table(data, generated / "fv_data_table.tex")
    write_fit_table(results, generated / "fv_fit_results.tex")
    write_grid_table(results, generated / "fv_fit_grid.tex")
    if not args.no_figure:
        make_figure(data, results, args.paper_dir / "figures" / "fv_gaplo.pdf")
    print(f"\nwrote outputs under {args.paper_dir}")


if __name__ == "__main__":
    main()
