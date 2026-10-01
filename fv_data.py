#!/usr/bin/env python3
"""Finite-volume data layer for the paper's Sec. III analysis.

For every lattice size N_s and boundary condition this module returns the two
meson-band spectral moments of the volume-averaged operator O:

    W        = sum_n w_n                    (zeroth moment, summed weight)
    Dbar     = sum_n w_n Delta_n / W        (first moment, weighted gap)

with w_n = |<E_n|O|E_0>|^2, Delta_n = E_n - E_0, restricted to Delta_n < DELTA_CUT.

All primary numbers come from the z2-gauge-theory.jl runs listed in FINAL.
The independent z2-claude.jl runs are loaded only as a cross-check.
"""
import csv
import math
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
GAUGE = DATA / "z2_gauge_theory"
PREFIX = "convergence_overlap_pzero"

# Upper edge of the single-meson band; the next band starts at Delta ~ 2.34.
DELTA_CUT = 2.0

SIZES = (4, 6, 8, 10, 12, 14, 16, 18, 24)
BOUNDARIES = ("open_site", "PBC")
LABEL = {"open_site": "OBC", "PBC": "PBC"}

# Most converged z2-gauge run for each (N_s, boundary): directory and glob.
FINAL = {
    (14, "open_site"): (GAUGE, f"{PREFIX}_L14_OBC_warm3280_seed*.csv"),
    (14, "PBC"): (GAUGE, f"{PREFIX}_L14_PBC_patched_warm_seed*.csv"),
    (16, "open_site"): (GAUGE, f"{PREFIX}_L16_OBC_warm2_k40_seed*.csv"),
    (16, "PBC"): (GAUGE, f"{PREFIX}_L16_PBC_patched_warm2560_seed*.csv"),
    (18, "open_site"): (DATA / "L18_nsw5120" / "z2_gauge_theory", "*.csv"),
    (18, "PBC"): (GAUGE, f"{PREFIX}_L18_PBC_patched_warm2560_seed*.csv"),
    (24, "open_site"): (GAUGE, f"{PREFIX}_L24_OBC_warm5_k40_seed*.csv"),
    (24, "PBC"): (GAUGE, f"{PREFIX}_L24_PBC_warm_k40_seed*.csv"),
}
for _L in (4, 6, 8, 10, 12):
    for _b in BOUNDARIES:
        FINAL[(_L, _b)] = (GAUGE, f"{PREFIX}_L{_L}_{_b}_patched_seed*.csv")

# Preceding refinement stage of the same run, where one exists.  Used to
# measure how far the result moved in the last stage.
PREVIOUS = {
    (14, "open_site"): (DATA / "L14_nsw1280_OBC" / "z2_gauge_theory", "*.csv"),
    (14, "PBC"): (GAUGE, f"{PREFIX}_L14_PBC_patched_nsw320_seed*.csv"),
    (16, "open_site"): (GAUGE, f"{PREFIX}_L16_OBC_warm_k40_seed*.csv"),
    (16, "PBC"): (GAUGE, f"{PREFIX}_L16_PBC_patched_warm1280_seed*.csv"),
    (18, "open_site"): (DATA / "L18_nsw2560" / "z2_gauge_theory", "*.csv"),
    (18, "PBC"): (GAUGE, f"{PREFIX}_L18_PBC_patched_warm1280_seed*.csv"),
    (24, "open_site"): (GAUGE, f"{PREFIX}_L24_OBC_warm4_k40_seed*.csv"),
}

# Exact diagonalisation at N_s = 4, 6 (written by exact_diag_small.jl).
EXACT = DATA / "exact_diag" / "exact_moments.csv"

# Independent implementation (z2-claude.jl), most converged run per size.
CLAUDE = {
    4: (DATA / "z2_claude", f"{PREFIX}_prod_maxdim400_nsw80_seed1.csv"),
    6: (DATA / "L6_nsw640" / "z2_claude", "*.csv"),
    8: (DATA / "L8_nsw640" / "z2_claude", "*.csv"),
    10: (DATA / "L10_nsw640" / "z2_claude", "*.csv"),
    12: (DATA / "L12_nsw640" / "z2_claude", "*.csv"),
    14: (DATA / "L14_nsw640" / "z2_claude", "*.csv"),
    16: (DATA / "L16_nsw640" / "z2_claude", "*.csv"),
    18: (DATA / "L18_nsw1920" / "z2_claude", "*.csv"),
}


def moments(row):
    """Band-restricted moments of one CSV row.

    Returns (Dbar, W, sigma_Dbar, n_states, top), where top is the gap of the
    highest state in the band.  sigma_Dbar propagates the
    per-state energy standard deviations sigma_n = sqrt(<H^2> - <H>^2), each of
    which bounds the distance of E_n from an exact eigenvalue.
    """
    n_states = 0
    while f"gap_k{n_states + 1}" in row:
        n_states += 1
    num = den = var = top = 0.0
    for n in range(1, n_states + 1):
        gap = float(row[f"gap_k{n}"])
        weight = float(row[f"overlap_k{n}"])
        sigma = float(row.get(f"sigma_k{n}") or 0.0)
        if gap < DELTA_CUT and weight > 0:
            num += weight * gap
            den += weight
            var += (weight * sigma) ** 2
        if gap < DELTA_CUT:
            top = max(top, gap)
    if den <= 0:
        return None
    return num / den, den, math.sqrt(var) / den, n_states, top


def _is_selected(row, size, boundary):
    if int(float(row["L"])) != size or row["boundary"] != boundary:
        return False
    return boundary == "PBC" or row["z2_obc_boundary"] == "truncate_xz"


def load_runs(directory, pattern, size, boundary):
    """Per-seed moments for one (N_s, boundary) from the files matching pattern."""
    runs = []
    for path in sorted(Path(directory).glob(pattern)):
        match = re.search(r"seed(\d+)", path.name)
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if not _is_selected(row, size, boundary):
                    continue
                result = moments(row)
                if result is not None:
                    runs.append({"seed": int(match.group(1)) if match else 0,
                                 "Dbar": result[0], "W": result[1],
                                 "sigma_var": result[2], "n_states": result[3],
                                 "top": result[4],
                                 "file": path.name})
    return runs


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def summarize(runs):
    """Seed-averaged moments with their measured error components."""
    n = len(runs)
    dbar = [r["Dbar"] for r in runs]
    weight = [r["W"] for r in runs]
    return {
        "n_seeds": n,
        "n_states": runs[0]["n_states"],
        "Dbar": _mean(dbar),
        "Dbar_seed_sem": _std(dbar) / math.sqrt(n),
        # Convergence error of the DMRG states.  It is common to all seeds, so
        # it is not reduced by averaging.
        "Dbar_var": _mean([r["sigma_var"] for r in runs]),
        "W": _mean(weight),
        "top_level": _mean([r["top"] for r in runs]),
        "W_seed_sem": _std(weight) / math.sqrt(n),
    }


def load_exact():
    """Return {(N_s, boundary): {"Dbar", "W"}} from exact diagonalisation.

    Only the boundary charges used in the paper, (q_L, q_R) = (-1, -1).
    """
    out = {}
    with EXACT.open() as handle:
        for row in csv.DictReader(handle):
            if (int(row["q_left"]), int(row["q_right"])) == (-1, -1):
                out[(int(row["L"]), row["boundary"])] = {
                    "Dbar": float(row["Dbar"]), "W": float(row["W"])}
    return out


def load_all():
    """Return {(N_s, boundary): summary} for the primary (z2-gauge) data.

    Error model.  Every component is measured, none is assigned by hand:
      Dbar: seed standard error (+) propagated energy standard deviation
            (+) shift in the last refinement stage (+) validation floor.
      W:    seed standard error (+) difference between the two codes
            (+) shift in the last refinement stage (+) validation floor.
    The validation floor is the largest deviation of the DMRG result from
    exact diagonalisation at N_s = 4, 6.  Sizes with a single seed take the
    seed scatter of the nearest smaller size that has several.
    """
    exact = load_exact()
    other = load_claude()
    out = {}
    for key in sorted(FINAL):
        size, boundary = key
        runs = load_runs(*FINAL[key], size, boundary)
        if not runs:
            raise FileNotFoundError(f"no final run for N_s={size} {boundary}")
        summary = summarize(runs)
        summary["Dbar_stage_shift"] = summary["W_stage_shift"] = 0.0
        if key in PREVIOUS:
            prev = summarize(load_runs(*PREVIOUS[key], size, boundary))
            summary["Dbar_stage_shift"] = summary["Dbar"] - prev["Dbar"]
            summary["W_stage_shift"] = summary["W"] - prev["W"]
        summary["Dbar_code_diff"] = summary["W_code_diff"] = None
        if key in other:
            summary["Dbar_code_diff"] = other[key]["Dbar"] - summary["Dbar"]
            summary["W_code_diff"] = other[key]["W"] - summary["W"]
        summary["Dbar_exact"] = summary["W_exact"] = None
        if key in exact:
            summary["Dbar_exact"] = exact[key]["Dbar"]
            summary["W_exact"] = exact[key]["W"]
        out[key] = summary

    floor = {q: max(abs(s[q] - s[f"{q}_exact"]) for s in out.values()
                    if s[f"{q}_exact"] is not None) for q in ("Dbar", "W")}
    for boundary in BOUNDARIES:
        last_multi = None
        for size in SIZES:
            s = out[(size, boundary)]
            if s["n_seeds"] > 1:
                last_multi = s
            elif last_multi is not None:
                scale = math.sqrt(last_multi["n_seeds"])
                s["Dbar_seed_sem"] = last_multi["Dbar_seed_sem"] * scale
                s["W_seed_sem"] = last_multi["W_seed_sem"] * scale
            s["Dbar_floor"], s["W_floor"] = floor["Dbar"], floor["W"]
            s["Dbar_err"] = math.sqrt(s["Dbar_seed_sem"] ** 2 + s["Dbar_var"] ** 2
                                      + s["Dbar_stage_shift"] ** 2 + floor["Dbar"] ** 2)
            s["W_err"] = math.sqrt(s["W_seed_sem"] ** 2 + (s["W_code_diff"] or 0.0) ** 2
                                   + s["W_stage_shift"] ** 2 + floor["W"] ** 2)
            # No second code and no earlier stage to compare with: do not claim
            # a smaller uncertainty than the next smaller size.
            if s["W_code_diff"] is None and (size, boundary) not in PREVIOUS:
                index = SIZES.index(size)
                if index > 0:
                    s["W_err"] = max(s["W_err"], out[(SIZES[index - 1], boundary)]["W_err"])
    return out


def load_claude():
    """Return {(N_s, boundary): summary} for the independent implementation."""
    out = {}
    for size, (directory, pattern) in CLAUDE.items():
        for boundary in BOUNDARIES:
            runs = load_runs(directory, pattern, size, boundary)
            if runs:
                out[(size, boundary)] = summarize(runs)
    return out


if __name__ == "__main__":
    data = load_all()
    print(f"{'N_s':>3} {'bc':>3} {'seeds':>5} {'n':>3} {'Dbar':>10} {'sem':>8} {'var':>8} "
          f"{'err':>8} {'code':>9} {'exact':>9} | {'W':>11} {'sem':>8} {'stage':>8} "
          f"{'code':>8} {'exact':>8} {'err':>8}")

    def fmt(value, width, spec):
        return f"{value:{spec}}".rjust(width) if value is not None else " " * width

    for (size, boundary), s in sorted(data.items()):
        d_exact = None if s["Dbar_exact"] is None else s["Dbar_exact"] - s["Dbar"]
        w_exact = None if s["W_exact"] is None else s["W_exact"] - s["W"]
        print(f"{size:>3} {LABEL[boundary]:>3} {s['n_seeds']:>5} {s['n_states']:>3} "
              f"{s['Dbar']:10.7f} {s['Dbar_seed_sem']:8.1e} {s['Dbar_var']:8.1e} "
              f"{s['Dbar_err']:8.1e} {fmt(s['Dbar_code_diff'], 9, '+.1e')} "
              f"{fmt(d_exact, 9, '+.1e')} | {s['W']:11.9f} {s['W_seed_sem']:8.1e} "
              f"{s['W_stage_shift']:+8.1e} {fmt(s['W_code_diff'], 8, '+.1e')} "
              f"{fmt(w_exact, 8, '+.1e')} {s['W_err']:8.1e}")
