#!/usr/bin/env python3
"""
Extrapolate gauge PBC M̄ to infinite nsw per L.

Data = mean PBC M̄ at each nsw label (warm-started cumulative series).
x-axis = nominal nsw label (proxy for convergence depth).

Models fit per L:
  A) M_inf - a/n
  B) M_inf - a/n^p      (3-param)
  C) M_inf - a*exp(-n/tau)  (3-param)

Validation: L=10, L=12 are known to plateau at ~1.612 by nsw=640.
Fit them on their EARLY points only and check M_inf recovers ~1.612.
"""
import numpy as np
from scipy.optimize import curve_fit

# gauge PBC mean M̄ at each nsw (from convergence_overlap_pzero CSVs)
SERIES = {
    10: [(160, 1.60748), (320, 1.60947), (640, 1.61157)],
    12: [(160, 1.60589), (320, 1.60664), (640, 1.61163)],
    14: [(160, 1.60385), (320, 1.60684), (640, 1.60824), (1280, 1.60925)],
    16: [(160, 1.58792), (320, 1.60158), (640, 1.60811), (1280, 1.60888), (2560, 1.60915)],
    18: [(640, 1.60594), (1280, 1.60766), (1920, 1.60752), (2560, 1.60915), (5120, 1.61054)],
}
PLATEAU = 1.6116  # observed converged small-L PBC value

def f_pow1(n, Minf, a):       return Minf - a / n
def f_powp(n, Minf, a, p):    return Minf - a / n**p
def f_exp(n, Minf, a, tau):   return Minf - a * np.exp(-n / tau)

def fit(model, n, y, p0, bounds):
    try:
        popt, pcov = curve_fit(model, n, y, p0=p0, bounds=bounds, maxfev=200000)
        perr = np.sqrt(np.diag(pcov))
        resid = y - model(n, *popt)
        rms = np.sqrt(np.mean(resid**2))
        return popt, perr, rms
    except Exception as e:
        return None, None, str(e)

def report(L, pts, drop_last_for_validation=False):
    n = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    tag = f"L={L}"
    if drop_last_for_validation:
        # fit on all but the last (known) point, predict it
        n_fit, y_fit = n[:-1], y[:-1]
        known = y[-1]; known_n = n[-1]
        tag += f" [validate: fit n≤{int(n_fit[-1])}, predict n={int(known_n)}={known:.5f}]"
    else:
        n_fit, y_fit = n, y
    print(f"\n{tag}")
    span = y_fit.max() - y_fit.min() + 1e-9
    # A) 1/n
    popt, perr, rms = fit(f_pow1, n_fit, y_fit, [y_fit[-1]+span, span*n_fit[0]],
                          ([y_fit[-1]-0.01, 0], [1.65, 10]))
    if popt is not None:
        print(f"  A) M_inf - a/n      : M_inf = {popt[0]:.5f} ± {perr[0]:.5f}   rms={rms:.5f}")
    # B) 1/n^p
    if len(n_fit) >= 3:
        popt, perr, rms = fit(f_powp, n_fit, y_fit, [y_fit[-1]+span, span, 0.7],
                              ([y_fit[-1]-0.01, 0, 0.1], [1.65, 1e6, 3]))
        if popt is not None:
            print(f"  B) M_inf - a/n^p    : M_inf = {popt[0]:.5f} ± {perr[0]:.5f}   p={popt[2]:.2f}  rms={rms:.5f}")
    # C) exp
    if len(n_fit) >= 3:
        popt, perr, rms = fit(f_exp, n_fit, y_fit, [y_fit[-1]+span, span, n_fit[len(n_fit)//2]],
                              ([y_fit[-1]-0.01, 0, 10], [1.65, 10, 1e5]))
        if popt is not None:
            print(f"  C) M_inf - a e^-n/τ : M_inf = {popt[0]:.5f} ± {perr[0]:.5f}   τ={popt[2]:.0f}  rms={rms:.5f}")

print("="*70)
print("VALIDATION — fit early points, predict the known plateau")
print("="*70)
report(10, SERIES[10], drop_last_for_validation=True)
report(12, SERIES[12], drop_last_for_validation=True)

print("\n" + "="*70)
print("EXTRAPOLATION — large L, all points, M_inf = converged PBC M̄")
print("="*70)
for L in (14, 16, 18):
    report(L, SERIES[L])

print(f"\n(small-L converged plateau ≈ {PLATEAU})")
