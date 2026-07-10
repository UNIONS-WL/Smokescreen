"""Backend smoke test: blind a synthetic SACC vector through a synthetic callable.

Run against an *installed* Smokescreen (no repo imports beyond the package).
Asserts, end-to-end and CCL-free:

1. the blinded vector differs from the input,
2. the same seed reproduces the identical blinded vector,
3. a different seed produces a different blinded vector.
"""
import copy

import numpy as np
import sacc

from smokescreen import ConcealDataVector

N = 10
FIDUCIAL = {"sigma8": 0.8, "Omega_c": 0.25}
SHIFTS = {"sigma8": 0.1, "Omega_c": (-0.05, 0.05)}


def make_sacc(n=N):
    """A small in-memory SACC: one tracer pair, n cl_ee rows, dense covariance."""
    s = sacc.Sacc()
    s.add_tracer("misc", "src")
    for i in range(n):
        s.add_data_point("galaxy_shear_cl_ee", ("src", "src"),
                         float(i + 1), ell=10 * (i + 1))
    s.add_covariance(np.diag(np.arange(1, n + 1, dtype=float)) + 0.01)
    return s


def theory_fn(params):
    """Synthetic theory: smooth analytic function of the angular bins. No CCL."""
    ell = 10 * np.arange(1, N + 1, dtype=float)
    return params["sigma8"] * np.log(ell) + params["Omega_c"] * np.sqrt(ell)


def blind(seed):
    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, copy.deepcopy(make_sacc()),
                            seed=seed, theory_fn=theory_fn)
    cdv.calculate_concealing_factor()
    return np.asarray(cdv.apply_concealing_to_likelihood_datavec())


original = make_sacc().mean

assert not np.allclose(blind(2112), original), "blinded vector equals input"
assert np.array_equal(blind(2112), blind(2112)), "same seed did not reproduce"
assert not np.array_equal(blind(2112), blind(42)), "different seeds agree"

print("backend-smoke OK: changed, seed-reproducible, seed-sensitive")
