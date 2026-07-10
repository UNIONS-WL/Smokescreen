import numpy as np
import pytest

from smokescreen.param_shifts import (
    draw_param_shifts,
    draw_flat_param_shifts,
    _normalize_seed,
)


# --- reproducibility & structure --------------------------------------------

def test_fixed_seed_reproduces_fixed_shift():
    shifts = {"Omega_c": 0.05, "sigma8": (-0.1, 0.1)}
    a = draw_param_shifts(shifts, 2112)
    b = draw_param_shifts(shifts, 2112)
    assert a == b
    assert set(a) == set(shifts)


def test_key_order_independence():
    forward = {"Omega_c": 0.05, "sigma8": 0.1, "h": (-0.02, 0.03)}
    reversed_dict = {"h": (-0.02, 0.03), "sigma8": 0.1, "Omega_c": 0.05}
    a = draw_param_shifts(forward, 2112)
    b = draw_param_shifts(reversed_dict, 2112)
    assert a == b  # per-key deltas identical regardless of insertion order


def test_key_subset_independence():
    # a key's delta depends only on (key, seed, shift_distr) — adding or
    # removing OTHER keys must not change it (staged-blinding reproducibility)
    full = draw_param_shifts({"sigma8": 0.05, "Omega_c": 0.03, "h": 0.01}, 42)
    alone = draw_param_shifts({"sigma8": 0.05}, 42)
    assert full["sigma8"] == alone["sigma8"]


def test_golden_values():
    # Pin the cross-run/cross-machine reproducibility contract: these exact
    # values are how historical blinding shifts are re-derived at unblinding.
    # Any RNG/normalization/derivation change MUST fail this test.
    assert _normalize_seed("my_secret_seed") == 3734715429001406524
    assert draw_param_shifts({"Omega_c": 0.05}, 2112) == {
        "Omega_c": pytest.approx(-0.034665567512199846, abs=0, rel=1e-15)
    }
    assert draw_param_shifts({"Omega_c": 0.05}, 2112, shift_distr="gaussian") == {
        "Omega_c": pytest.approx(0.008622042117174187, abs=0, rel=1e-15)
    }


def test_draw_does_not_perturb_global_np_random():
    shifts = {"Omega_c": 0.05, "sigma8": 0.1}
    np.random.seed(0)
    a = np.random.random()
    np.random.seed(0)
    draw_param_shifts(shifts, 2112)
    b = np.random.random()
    assert a == b  # local RNG never touched the global state


# --- delta semantics --------------------------------------------------------

def test_flat_float_is_delta_within_symmetric_envelope():
    h = 0.05
    shifts = {"Omega_c": h}
    for seed in range(20):
        delta = draw_param_shifts(shifts, seed)["Omega_c"]
        assert -h <= delta <= h


def test_flat_tuple_is_delta_within_box():
    lo, hi = -0.03, 0.07
    shifts = {"Omega_c": (lo, hi)}
    for seed in range(20):
        delta = draw_param_shifts(shifts, seed)["Omega_c"]
        assert lo <= delta <= hi


def test_flat_list_envelope_equivalent_to_tuple():
    # [lo, hi] is what YAML/JSON configs naturally produce
    as_list = draw_param_shifts({"Omega_c": [-0.03, 0.07]}, 2112)
    as_tuple = draw_param_shifts({"Omega_c": (-0.03, 0.07)}, 2112)
    assert as_list == as_tuple


# --- seed normalization -----------------------------------------------------

def test_string_seed_matches_normalized_int():
    shifts = {"Omega_c": 0.05, "sigma8": (-0.1, 0.1)}
    str_seed = "my_secret_seed"
    int_seed = _normalize_seed(str_seed)
    assert draw_param_shifts(shifts, str_seed) == draw_param_shifts(shifts, int_seed)


# --- distributions ----------------------------------------------------------

def test_gaussian_float():
    shifts = {"Omega_c": 0.05}
    result = draw_param_shifts(shifts, 2112, shift_distr="gaussian")
    assert isinstance(result["Omega_c"], float)
    # gaussian must actually be a different distribution than flat
    flat = draw_param_shifts(shifts, 2112, shift_distr="flat")
    assert result["Omega_c"] != flat["Omega_c"]


def test_gaussian_escapes_flat_envelope():
    # gaussian is unbounded; flat is confined to [-h, h] — a cheap
    # discriminator that the gaussian branch is not silently flat
    h = 0.05
    assert any(
        abs(draw_param_shifts({"Omega_c": h}, s, shift_distr="gaussian")["Omega_c"]) > h
        for s in range(200)
    )


def test_gaussian_tuple_raises():
    shifts = {"Omega_c": (-0.05, 0.05)}
    with pytest.raises(ValueError):
        draw_param_shifts(shifts, 2112, shift_distr="gaussian")


def test_flat_tuple_wrong_length_raises():
    shifts = {"Omega_c": (0.0, 0.05, 0.1)}
    with pytest.raises(ValueError):
        draw_param_shifts(shifts, 2112)


def test_unknown_distribution_raises():
    with pytest.raises(NotImplementedError):
        draw_param_shifts({"Omega_c": 0.05}, 2112, shift_distr="lorentzian")


# --- back-compat alias ------------------------------------------------------

def test_draw_flat_param_shifts_alias():
    shifts = {"Omega_c": 0.05, "sigma8": (-0.1, 0.1)}
    assert draw_flat_param_shifts(shifts, 2112) == draw_param_shifts(
        shifts, 2112, shift_distr="flat"
    )
