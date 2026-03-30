"""
Performance regression test for sap_model.py.

Captures reference outputs from fit_site on a diverse set of sites,
saves them to a .npz file, and compares post-refactor outputs at tight
floating-point tolerances.

Usage:
    python code/sap_perf_test.py --generate   # create reference .npz (run ONCE before refactor)
    python code/sap_perf_test.py --check      # compare current code against reference
    python code/sap_perf_test.py --self-test  # run sap_model self-tests (fast)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Ensure repo-root imports work
sys.path.insert(0, os.path.dirname(__file__))

import config
from sap_data import compute_intensity_strata, load_all
from sap_model import (
    SiteParams,
    compute_adaptive_weights,
    compute_mu,
    compute_s_matrix,
    fit_site,
    tweedie_deviance,
    tweedie_total_deviance,
    tweedie_variance,
)

REFERENCE_PATH = os.path.join(
    config.SONG_ANALYSIS_CACHE_DIR,
    "perf_test_reference.npz",
)

# Sites chosen to span all 4 intensity strata + varying zero counts
TEST_SITES = [0, 42, 100, 500, 1000]

# Tolerances: algebraic rewrites should be near-identical in IEEE 754
ATOL = 1e-10
RTOL = 1e-10


def _load_data():
    """Load SAP data (Phase 1) and compute derived quantities."""
    data, report = load_all(include_rna=True)
    strata = compute_intensity_strata(data.x_base)
    omega = compute_adaptive_weights(data.cvs, gamma=0.5)
    a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values
    y_all = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_all = data.x_base.values
    return data, strata, omega, a_bar, y_all, x_base_all


def _fit_test_sites(data, strata, omega, a_bar, y_all, x_base_all):
    """Fit each test site and collect results as arrays."""
    n_sites = data.n_sites_filtered
    sites = [s for s in TEST_SITES if s < n_sites]

    # Output arrays
    betas = []
    alpha_gens = []
    alpha_times = []
    rhos = []
    phis = []
    gamma0s = []
    gamma1s = []
    mus = []
    deviances = []
    converged_flags = []
    s_matrices = []

    for j in sites:
        y = y_all[j]
        x_base_j = x_base_all[j]
        r_slice_j = (data.r_tensor[:, :, j]
                     if data.r_tensor is not None
                     else np.zeros((6, 24)))

        params, conv = fit_site(
            j, y, data.a_obs, data.sample_meta,
            x_base_j, r_slice_j, p=1.5,
            lambda_q=0.1, lambda_rho=0.1,
            omega=omega, a_bar=a_bar, eta=2.0, stratum=strata[j],
        )

        mu = compute_mu(params, data.a_obs, data.sample_meta, x_base_j, r_slice_j)
        S = compute_s_matrix(params, data.sample_meta, x_base_j, r_slice_j)

        pos_mask = y > 0
        dev = tweedie_total_deviance(y[pos_mask], mu[pos_mask], 1.5) if np.any(pos_mask) else 0.0

        betas.append(params.beta)
        alpha_gens.append(params.alpha_gen)
        alpha_times.append(params.alpha_time)
        rhos.append(params.rho)
        phis.append(params.phi)
        gamma0s.append(params.gamma0)
        gamma1s.append(params.gamma1)
        mus.append(mu)
        deviances.append(dev)
        converged_flags.append(conv)
        s_matrices.append(S)

    return {
        "sites": np.array(sites),
        "betas": np.array(betas),
        "alpha_gens": np.array(alpha_gens),
        "alpha_times": np.array(alpha_times),
        "rhos": np.array(rhos),
        "phis": np.array(phis),
        "gamma0s": np.array(gamma0s),
        "gamma1s": np.array(gamma1s),
        "mus": np.array(mus),
        "deviances": np.array(deviances),
        "converged": np.array(converged_flags),
        "s_matrices": np.array(s_matrices),
    }


def generate_reference():
    """Run fits and save reference outputs."""
    print("Loading data...")
    data, strata, omega, a_bar, y_all, x_base_all = _load_data()

    print(f"Fitting {len([s for s in TEST_SITES if s < data.n_sites_filtered])} test sites...")
    t0 = time.perf_counter()
    results = _fit_test_sites(data, strata, omega, a_bar, y_all, x_base_all)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s")

    # Also capture Tweedie utility outputs for extra coverage
    y_test = np.array([0.0, 0.0, 1.5, 3.2, 0.8, 5.0, 0.1, 10.0])
    mu_test = np.array([0.5, 0.3, 1.2, 3.0, 1.0, 4.5, 0.2, 9.5])
    results["tw_deviance_15"] = tweedie_deviance(y_test, mu_test, 1.5)
    results["tw_variance_15"] = tweedie_variance(mu_test, 1.5)

    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    np.savez_compressed(REFERENCE_PATH, **results)
    print(f"Reference saved to {REFERENCE_PATH}")
    print(f"  Sites: {results['sites']}")
    print(f"  Converged: {results['converged']}")


def check_against_reference():
    """Compare current code outputs against saved reference."""
    if not os.path.exists(REFERENCE_PATH):
        print(f"ERROR: Reference file not found: {REFERENCE_PATH}")
        print("Run with --generate first.")
        sys.exit(1)

    ref = np.load(REFERENCE_PATH)

    print("Loading data...")
    data, strata, omega, a_bar, y_all, x_base_all = _load_data()

    print("Fitting test sites with current code...")
    t0 = time.perf_counter()
    results = _fit_test_sites(data, strata, omega, a_bar, y_all, x_base_all)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s")

    # Also recompute Tweedie utilities
    y_test = np.array([0.0, 0.0, 1.5, 3.2, 0.8, 5.0, 0.1, 10.0])
    mu_test = np.array([0.5, 0.3, 1.2, 3.0, 1.0, 4.5, 0.2, 9.5])
    results["tw_deviance_15"] = tweedie_deviance(y_test, mu_test, 1.5)
    results["tw_variance_15"] = tweedie_variance(mu_test, 1.5)

    # Compare
    all_passed = True
    fields = [
        "betas", "alpha_gens", "alpha_times", "rhos", "phis",
        "gamma0s", "gamma1s", "mus", "deviances", "s_matrices",
        "tw_deviance_15", "tw_variance_15",
    ]

    print()
    for field in fields:
        ref_val = ref[field]
        cur_val = results[field]
        max_abs_diff = float(np.max(np.abs(ref_val - cur_val)))
        max_rel_diff = float(np.max(
            np.abs(ref_val - cur_val) / (np.abs(ref_val) + 1e-30)
        ))
        passed = np.allclose(ref_val, cur_val, atol=ATOL, rtol=RTOL)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {field:20s}  max_abs={max_abs_diff:.2e}  max_rel={max_rel_diff:.2e}")

    # Converged flags: exact match
    conv_match = np.array_equal(ref["converged"], results["converged"])
    status = "PASS" if conv_match else "FAIL"
    if not conv_match:
        all_passed = False
    print(f"  [{status}] {'converged':20s}  exact={conv_match}")

    print()
    if all_passed:
        print("ALL CHECKS PASSED — numerical equivalence confirmed.")
    else:
        print("SOME CHECKS FAILED — refactor introduced numerical differences.")
        sys.exit(1)


def run_self_tests():
    """Proxy to sap_model self-tests."""
    from sap_model import _run_self_tests
    _run_self_tests()


def main():
    parser = argparse.ArgumentParser(description="SAP model performance regression tests")
    parser.add_argument("--generate", action="store_true",
                        help="Generate reference .npz from current (pre-refactor) code")
    parser.add_argument("--check", action="store_true",
                        help="Compare current code against saved reference")
    parser.add_argument("--self-test", action="store_true",
                        help="Run sap_model Tweedie self-tests")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
    elif args.generate:
        generate_reference()
    elif args.check:
        check_against_reference()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
