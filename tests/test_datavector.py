"""Blinding-mechanics tests for ConcealDataVector.

All mechanics tests inject a SYNTHETIC ``theory_fn`` (a pure, deterministic
closure over the cosmo params) so they never import pyccl or firecrown. The one
default-CCL-backend test imports pyccl locally.
"""
import numpy as np
import pytest
import sacc

from smokescreen.datavector import ConcealDataVector, concealing_factor
from smokescreen.param_shifts import DRAW_SCHEME, draw_param_shifts, seed_commitment


FIDUCIAL = {"sigma8": 0.8, "Omega_c": 0.25}
SHIFTS = {"sigma8": 0.1, "Omega_c": (-0.05, 0.05)}
SEED = 2112


# --- fixtures ---------------------------------------------------------------

def _make_sacc(n=5):
    """A minimal SACC with n cl_ee rows and a dense covariance."""
    s = sacc.Sacc()
    s.add_tracer("misc", "src")
    for i in range(n):
        s.add_data_point("galaxy_shear_cl_ee", ("src", "src"),
                         float(i + 1), ell=10 * (i + 1))
    cov = np.diag(np.arange(1, n + 1, dtype=float)) + 0.01
    s.add_covariance(cov)
    return s


def _synthetic_theory_fn(n):
    """Pure length-n theory vector, deterministic in the cosmo params."""
    base = np.arange(1, n + 1, dtype=float)
    return lambda p: base * p["sigma8"] + p["Omega_c"]


@pytest.fixture
def sacc_data():
    return _make_sacc(5)


@pytest.fixture
def theory_fn():
    return _synthetic_theory_fn(5)


# --- pure vector core -------------------------------------------------------

def test_concealing_factor_is_theory_difference(theory_fn):
    factor = concealing_factor(FIDUCIAL, SHIFTS, seed=SEED, theory_fn=theory_fn)

    deltas = draw_param_shifts(SHIFTS, SEED, shift_distr="flat")
    concealed = {k: FIDUCIAL[k] + deltas.get(k, 0.0) for k in FIDUCIAL}
    expected = theory_fn(concealed) - theory_fn(FIDUCIAL)
    np.testing.assert_array_equal(factor, expected)


def test_concealing_factor_mult_is_theory_ratio(theory_fn):
    factor = concealing_factor(FIDUCIAL, SHIFTS, seed=SEED, theory_fn=theory_fn,
                               factor_type="mult")

    deltas = draw_param_shifts(SHIFTS, SEED, shift_distr="flat")
    concealed = {k: FIDUCIAL[k] + deltas.get(k, 0.0) for k in FIDUCIAL}
    expected = theory_fn(concealed) / theory_fn(FIDUCIAL)
    np.testing.assert_array_equal(factor, expected)


def test_concealing_factor_needs_no_sacc_and_no_backend(theory_fn):
    # the core takes a theory_fn explicitly and never reaches for a default
    # backend, so it stays free of pyccl and of any data container
    factor = concealing_factor(FIDUCIAL, SHIFTS, seed=SEED, theory_fn=theory_fn)
    assert factor.shape == (5,)
    with pytest.raises(TypeError):
        concealing_factor(FIDUCIAL, SHIFTS, seed=SEED)


def test_concealing_factor_rejects_unknown_shift_key(theory_fn):
    with pytest.raises(ValueError, match="omega_c"):
        concealing_factor(FIDUCIAL, {"omega_c": 0.05}, seed=SEED,
                          theory_fn=theory_fn)


def test_concealing_factor_rejects_unknown_factor_type(theory_fn):
    with pytest.raises(NotImplementedError):
        concealing_factor(FIDUCIAL, SHIFTS, seed=SEED, theory_fn=theory_fn,
                          factor_type="subtract")


def test_class_factor_matches_pure_core(sacc_data, theory_fn):
    # the class is an adapter: same seed, same shifts, same factor
    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                            theory_fn=theory_fn, debug=True)
    np.testing.assert_array_equal(
        cdv.calculate_concealing_factor(factor_type="add"),
        concealing_factor(FIDUCIAL, SHIFTS, seed=SEED, theory_fn=theory_fn),
    )


def test_class_evaluates_each_theory_point_once(sacc_data):
    calls = []
    base = np.arange(1, 6, dtype=float)

    def counting_theory_fn(p):
        calls.append(dict(p))
        return base * p["sigma8"] + p["Omega_c"]

    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                            theory_fn=counting_theory_fn)
    cdv.calculate_concealing_factor(factor_type="add")
    cdv.calculate_concealing_factor(factor_type="mult")
    # fiducial and hidden, once each — delegating the factor must not make the
    # adapter pay for repeated theory evaluations
    assert len(calls) == 2


# --- construction & length guard --------------------------------------------

def test_construction_succeeds(sacc_data, theory_fn):
    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                            theory_fn=theory_fn)
    assert isinstance(cdv, ConcealDataVector)


def test_length_guard_raises(sacc_data):
    wrong = lambda p: np.ones(3)  # noqa: E731  -- wrong length
    with pytest.raises(ValueError):
        ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED, theory_fn=wrong)


def test_shape_guard_rejects_column_vector(sacc_data):
    # (n, 1) has len() == n but would broadcast mean(n,) + factor(n,1) into an
    # (n, n) matrix — must be rejected, not silently corrupt the blinded SACC
    wrong = lambda p: np.ones((5, 1))  # noqa: E731
    with pytest.raises(ValueError, match="shape"):
        ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED, theory_fn=wrong)


def test_unknown_shift_key_raises(sacc_data, theory_fn):
    # a typo'd shift key must fail loudly at construction
    with pytest.raises(ValueError, match="omega_c"):
        ConcealDataVector(FIDUCIAL, {"omega_c": 0.05}, sacc_data,
                          seed=SEED, theory_fn=theory_fn)


def test_unknown_keyword_raises(sacc_data, theory_fn):
    # a swallowed `theory_fn` typo would silently fall back to the default CCL
    # backend — blinding with one theory and unblinding with another
    with pytest.raises(TypeError):
        ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                          theory_func=theory_fn)


def test_seed_is_required(sacc_data, theory_fn):
    # custody contract: no default seed — omitting it must be a TypeError
    with pytest.raises(TypeError):
        ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, theory_fn=theory_fn)


# --- concealing factor (add) bit-for-bit ------------------------------------

def test_concealing_factor_add_bit_for_bit(sacc_data, theory_fn):
    # debug mode returns the factor; compare bit-for-bit to an independent recompute
    cdv_dbg = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                                theory_fn=theory_fn, debug=True)
    factor = cdv_dbg.calculate_concealing_factor(factor_type="add")

    deltas = draw_param_shifts(SHIFTS, SEED, shift_distr="flat")
    concealed = {k: FIDUCIAL[k] + deltas.get(k, 0.0) for k in FIDUCIAL}
    expected = theory_fn(concealed) - theory_fn(FIDUCIAL)
    np.testing.assert_array_equal(factor, expected)


def test_apply_add_returns_mean_plus_factor(sacc_data, theory_fn):
    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                            theory_fn=theory_fn, debug=True)
    factor = cdv.calculate_concealing_factor(factor_type="add")
    blinded = cdv.apply_concealing_to_likelihood_datavec()
    np.testing.assert_array_equal(blinded, sacc_data.mean + factor)


# --- concealing factor (mult) -----------------------------------------------

def test_concealing_factor_mult(sacc_data, theory_fn):
    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                            theory_fn=theory_fn, debug=True)
    factor = cdv.calculate_concealing_factor(factor_type="mult")

    deltas = draw_param_shifts(SHIFTS, SEED, shift_distr="flat")
    concealed = {k: FIDUCIAL[k] + deltas.get(k, 0.0) for k in FIDUCIAL}
    expected = theory_fn(concealed) / theory_fn(FIDUCIAL)
    np.testing.assert_array_equal(factor, expected)

    blinded = cdv.apply_concealing_to_likelihood_datavec()
    np.testing.assert_array_equal(blinded, sacc_data.mean * factor)


# --- save path --------------------------------------------------------------

def _blinded_cdv(sacc_data, theory_fn):
    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                            theory_fn=theory_fn)
    cdv.calculate_concealing_factor(factor_type="add")
    cdv.apply_concealing_to_likelihood_datavec()
    return cdv


def test_save_fits_and_cov_unchanged(tmp_path, sacc_data, theory_fn, monkeypatch):
    monkeypatch.setattr("smokescreen.datavector.getpass.getuser",
                        lambda: "test_user")
    cov_before = sacc_data.covariance.dense.copy()
    cdv = _blinded_cdv(sacc_data, theory_fn)

    out = cdv.save_concealed_datavector(str(tmp_path), "root", return_sacc=True)
    assert isinstance(out, sacc.Sacc)
    np.testing.assert_array_equal(out.mean, cdv.concealed_data_vector)
    np.testing.assert_allclose(out.covariance.dense, cov_before)
    # input SACC covariance untouched too
    np.testing.assert_allclose(sacc_data.covariance.dense, cov_before)

    md = out.metadata
    assert md["concealed"] is True
    assert md["creator"] == "test_user"
    assert "info" in md

    assert (tmp_path / "root_concealed_data_vector.fits").exists()


def test_save_commits_to_seed_without_revealing_it(tmp_path, sacc_data, theory_fn):
    # the concealed file is what circulates while the analysis is blind; the
    # seed plus the (public) config would unblind it, so only a commitment goes in
    cdv = _blinded_cdv(sacc_data, theory_fn)
    md = cdv.save_concealed_datavector(str(tmp_path), "root",
                                       return_sacc=True).metadata

    assert "seed_smokescreen" not in md
    assert md["seed_commitment"] == seed_commitment(SEED)
    assert str(SEED) not in md["seed_commitment"]
    assert md["draw_scheme"] == DRAW_SCHEME


def test_save_stamp_seed_opt_in(tmp_path, sacc_data, theory_fn):
    # upstream's behaviour remains reachable, deliberately
    cdv = _blinded_cdv(sacc_data, theory_fn)
    md = cdv.save_concealed_datavector(str(tmp_path), "root", return_sacc=True,
                                       stamp_seed=True).metadata
    assert md["seed_smokescreen"] == SEED
    assert md["seed_commitment"] == seed_commitment(SEED)


def test_save_return_sacc_false(tmp_path, sacc_data, theory_fn, monkeypatch):
    monkeypatch.setattr("smokescreen.datavector.getpass.getuser",
                        lambda: "test_user")
    cdv = _blinded_cdv(sacc_data, theory_fn)
    out = cdv.save_concealed_datavector(str(tmp_path), "root", return_sacc=False)
    assert out is None


def test_save_custom_suffix(tmp_path, sacc_data, theory_fn, monkeypatch):
    monkeypatch.setattr("smokescreen.datavector.getpass.getuser",
                        lambda: "test_user")
    cdv = _blinded_cdv(sacc_data, theory_fn)
    cdv.save_concealed_datavector(str(tmp_path), "root", suffix="mysuffix")
    assert (tmp_path / "root_mysuffix.fits").exists()


def test_save_hdf5_format(tmp_path, sacc_data, theory_fn, monkeypatch):
    pytest.importorskip("h5py")  # sacc.save_hdf5 needs h5py
    monkeypatch.setattr("smokescreen.datavector.getpass.getuser",
                        lambda: "test_user")
    cdv = _blinded_cdv(sacc_data, theory_fn)
    cdv.save_concealed_datavector(str(tmp_path), "root", output_format="hdf5")
    assert (tmp_path / "root_concealed_data_vector.hdf5").exists()


# --- end-to-end synthetic (default backend never constructed) ---------------

def test_end_to_end_synthetic(tmp_path, sacc_data, theory_fn, monkeypatch):
    monkeypatch.setattr("smokescreen.datavector.getpass.getuser",
                        lambda: "test_user")
    cdv = ConcealDataVector(FIDUCIAL, SHIFTS, sacc_data, seed=SEED,
                            theory_fn=theory_fn)
    cdv.calculate_concealing_factor(factor_type="add")
    blinded = cdv.apply_concealing_to_likelihood_datavec()
    out = cdv.save_concealed_datavector(str(tmp_path), "root", return_sacc=True)
    assert not np.array_equal(blinded, sacc_data.mean)  # actually blinded
    # Import discipline (pyccl never entering sys.modules on the synthetic path)
    # is covered by the subprocess suite in test_import_discipline.py.
    assert isinstance(out, sacc.Sacc)


# --- default CCL backend ----------------------------------------------------

def _make_cosmic_shear_sacc():
    """Small cosmic-shear SACC with 2 NZ tracers, xi± and cl_ee rows."""
    s = sacc.Sacc()
    z = np.linspace(0.05, 2.0, 50)
    for name, zmean in [("src0", 0.5), ("src1", 1.0)]:
        nz = np.exp(-0.5 * ((z - zmean) / 0.2) ** 2)
        s.add_tracer("NZ", name, z, nz)
    thetas = np.array([5.0, 20.0, 60.0, 120.0])  # arcmin
    ells = [50, 200, 800]
    pairs = [("src0", "src0"), ("src0", "src1"), ("src1", "src1")]
    for dt in ("galaxy_shear_xi_plus", "galaxy_shear_xi_minus"):
        for p in pairs:
            for th in thetas:
                s.add_data_point(dt, p, 1e-6, theta=th)
    for p in pairs:
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", p, 1e-9, ell=ell)
    n = len(s.mean)
    s.add_covariance(np.eye(n) * 1e-12)
    return s


def test_default_ccl_backend(tmp_path, monkeypatch):
    pytest.importorskip("pyccl")
    monkeypatch.setattr("smokescreen.datavector.getpass.getuser",
                        lambda: "test_user")
    s = _make_cosmic_shear_sacc()
    cov_before = s.covariance.dense.copy()
    fiducial = {"sigma8": 0.81, "Omega_c": 0.26, "Omega_b": 0.045,
                "h": 0.67, "n_s": 0.96}
    shifts = {"sigma8": 0.05, "Omega_c": (-0.03, 0.03)}

    cdv = ConcealDataVector(fiducial, shifts, s, seed=SEED, theory_fn=None)
    assert np.all(np.isfinite(cdv.theory_vec_fid))
    # the Cl branch (np.interp over the ell grid) must produce real power
    cl_idx = s.indices("galaxy_shear_cl_ee")
    assert np.all(cdv.theory_vec_fid[cl_idx] > 0)
    cdv.calculate_concealing_factor(factor_type="add")
    blinded = cdv.apply_concealing_to_likelihood_datavec()
    assert not np.array_equal(blinded, s.mean)

    out = cdv.save_concealed_datavector(str(tmp_path), "cs", return_sacc=True)
    np.testing.assert_allclose(out.covariance.dense, cov_before)
