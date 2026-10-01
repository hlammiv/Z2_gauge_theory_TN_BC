"""Checks for the Sec. III finite-volume analysis (run with pytest)."""
import math

import numpy as np
import pytest

import fit_fv
import fv_data


@pytest.fixture(scope="module")
def data():
    return fv_data.load_all()


@pytest.fixture(scope="module")
def results(data):
    return fit_fv.run_grid(data)


def test_every_size_has_both_moments(data):
    for size in fv_data.SIZES:
        for boundary in fv_data.BOUNDARIES:
            s = data[(size, boundary)]
            assert 1.40 < s["Dbar"] < 1.47
            assert 0.66 < s["W"] < 0.72
            assert s["Dbar_err"] > 0 and s["W_err"] > 0


def test_dmrg_reproduces_exact_diagonalisation(data):
    for size in (4, 6):
        for boundary in fv_data.BOUNDARIES:
            s = data[(size, boundary)]
            assert abs(s["Dbar"] - s["Dbar_exact"]) < 1e-6
            assert abs(s["W"] - s["W_exact"]) < 1e-6


def test_synthetic_series_is_recovered_and_rival_rejected():
    sizes = np.array(fit_fv.FIT_SIZES, float)
    truth = 1.5 - 0.2 / sizes - 0.1 / sizes ** 2
    err = np.full_like(sizes, 1e-5)
    good = fit_fv.fit("P2", sizes, truth, err)
    assert np.allclose(good["params"], [1.5, -0.2, -0.1], atol=1e-6)
    assert good["chi2"] < 1e-6
    rival = fit_fv.fit("Q2", sizes, truth, err, prior=(1.5, 1e-5))
    assert rival["p"] < 1e-6


def test_prior_ties_limit_to_reference():
    sizes = np.array(fit_fv.FIT_SIZES, float)
    truth = 1.5 - 0.2 / sizes
    fitted = fit_fv.fit("P1", sizes, truth, np.full_like(sizes, 1e-4), prior=(1.5, 1e-6))
    assert abs(fitted["params"][0] - 1.5) < 1e-6
    assert fitted["dof"] == len(sizes) + 1 - 2


def test_leading_correction_is_first_order(data, results):
    for obs in fit_fv.OBSERVABLES:
        best = fit_fv.preferred(results, obs)
        assert best["model"] in fit_fv.SERIES_WITH_1_OVER_N
        assert abs(best["params"][1]) > 10 * best["errors"][1]
        assert abs(best["held_out_pull"]) < 2
        # Forms with no 1/N_s term cannot describe all sizes.
        for rival in ("Q2", "Q3", "EX", "EM"):
            assert fit_fv.select(results, obs, "common", rival, 4)["p"] < fit_fv.P_ACCEPT
    # For the summed weight they fail for every N_min up to 10.
    for rival in ("Q2", "Q3", "EX", "EM"):
        for n_min in (4, 6, 8, 10):
            assert fit_fv.select(results, "W", "common", rival, n_min)["p"] < fit_fv.P_ACCEPT


def test_free_limit_agrees_with_periodic_value(results):
    for obs in fit_fv.OBSERVABLES:
        best = fit_fv.preferred(results, obs)
        free = fit_fv.select(results, obs, "free", best["model"], best["n_min"])
        assert abs(free["limit_minus_pbc"]) < 2 * free["limit_minus_pbc_err"]


@pytest.mark.parametrize("value, err, expected", [
    (1.46646, 0.00014, "1.46646(14)"),
    (-0.1739, 0.0032, "-0.1739(32)"),
    (58.9, 7.7, "58.9(77)"),
    (-201.0, 34.0, "-201(34)"),
    (0.5, 0.0996, "0.50(10)"),
])
def test_pm_format(value, err, expected):
    assert fit_fv.pm(value, err) == expected
    assert math.isfinite(float(expected.split("(")[0]))
