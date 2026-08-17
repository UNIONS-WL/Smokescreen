# author: Arthur Loureiro <arthur.loureiro@fysik.su.se>
# license: BSD 3-Clause
'''
Parameter Shifts (:mod:`smokescreen.param_shifts`)
===================================================

.. currentmodule:: smokescreen.param_shifts

The :mod:`smokescreen.param_shifts` module draws the hidden parameter
shifts that drive blinding. The shift for a given parameter is a *delta*
to be added to the fiducial value, drawn from a local, seeded RNG in a
way that does not depend on the iteration order of the shifts mapping.


Parameter Shifts
-----------------

.. autodata:: DRAW_SCHEME

.. autofunction:: draw_param_shifts
'''
import hashlib
from collections.abc import Mapping

import numpy as np


DRAW_SCHEME = 2
"""Version of the shift-draw semantics implemented by :func:`draw_param_shifts`.

Two installs that agree on ``(seed, shifts_dict, shift_distr)`` produce the same
deltas only if they also agree on ``DRAW_SCHEME``. Downstream custody schemes
bind this number into their blind commitment, so an install with different draw
semantics fails loudly at unblind instead of silently subtracting the wrong
shift.

Schemes:

1. One global ``numpy.random.default_rng(seed)`` stream consumed sequentially
   over the sorted keys (upstream DESC Smokescreen). A key's delta depends on
   how many sorted-earlier keys are present.
2. A per-key RNG, ``numpy.random.default_rng([seed, hash(key)])`` (this fork).
   A key's delta depends only on ``(seed, key, shift_distr)``.

Bump this whenever the drawn deltas for a fixed ``(seed, shifts_dict,
shift_distr)`` change for any reason.
"""


def _normalize_seed(seed):
    """
    Reduce a seed to an integer suitable for ``numpy.random.default_rng``.

    An ``int`` is returned as-is. A ``str`` is reduced deterministically to a
    64-bit integer via SHA-256 of its UTF-8 bytes; the mapping from a given
    string to its integer seed is fixed and reproducible across runs and
    machines.

    Parameters
    ----------
    seed : int or str
        Seed value.

    Returns
    -------
    int
        Normalized integer seed.
    """
    if isinstance(seed, str):
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    raise TypeError(f"seed must be int or str, got {type(seed).__name__}")


def draw_param_shifts(shifts_dict, seed, shift_distr="flat"):
    """
    Draw a delta for each named parameter from a local RNG seeded by ``seed``.

    Each returned value is a shift to be *added* to the fiducial value, not an
    absolute parameter value. The draw is per-key independent: the delta for a
    given key depends only on ``(key, seed, shift_distr)`` --- not on the
    iteration order of ``shifts_dict``, and not on which *other* keys are
    present. Each key gets its own RNG derived from ``(seed, key)``, so adding
    or removing a parameter from the shifts envelope never changes any other
    parameter's drawn delta. These semantics are versioned by
    :data:`DRAW_SCHEME`.

    Parameters
    ----------
    shifts_dict : Mapping[str, float | tuple[float, float]]
        Mapping of parameter name to its shift envelope, interpreted as a delta
        about zero. A ``float`` ``h`` is the symmetric delta envelope
        ``(-h, +h)``; a ``(lo, hi)`` pair (tuple or list, e.g. as loaded from
        YAML/JSON) is a delta box, the drawn delta lies in ``[lo, hi]``
        (typically straddling zero).
    seed : int or str
        Seed for the random number generator. A ``str`` is normalized to an
        ``int`` (see :func:`_normalize_seed`).
    shift_distr : str
        ``"flat"`` draws uniformly over the delta envelope. ``"gaussian"``
        draws from a zero-mean Gaussian with ``sigma`` equal to the ``float``
        half-width and accepts only the symmetric ``float`` form.

    Returns
    -------
    dict[str, float]
        Mapping ``{param_name: drawn_delta}``, keys a subset of
        ``shifts_dict``'s keys, values plain floats. Overlay these on the
        fiducial parameters: ``concealed[k] = fiducial[k] + delta[k]``.
    """
    if shift_distr not in ("flat", "gaussian"):
        raise NotImplementedError("Only flat and gaussian shifts are implemented")

    base_seed = _normalize_seed(seed)

    shifts = {}
    for key, envelope in shifts_dict.items():
        # Per-key derived RNG: the draw for a key depends only on (seed, key),
        # never on the other keys in the mapping or on iteration order.
        rng = np.random.default_rng([base_seed, _normalize_seed(key)])
        if shift_distr == "flat":
            if isinstance(envelope, (tuple, list)):
                if len(envelope) != 2:
                    raise ValueError(
                        f"Shift envelope for {key!r} must have length 2 "
                        f"(lo, hi); got {envelope!r}"
                    )
                lo, hi = envelope
            else:
                lo, hi = -envelope, envelope
            shifts[key] = float(rng.uniform(lo, hi))
        else:  # gaussian
            if isinstance(envelope, (tuple, list)):
                raise ValueError(
                    f"Gaussian shift for {key} requires a float half-width, "
                    f"got {envelope!r}"
                )
            shifts[key] = float(rng.normal(0.0, envelope))
    return shifts


# Backwards-compatibility alias: the draw returns a plain mapping regardless of
# whether callers reach for it by the historical or the current name.
def draw_flat_param_shifts(shift_dict: Mapping, seed):
    """Deprecated alias for :func:`draw_param_shifts` with ``shift_distr="flat"``."""
    return draw_param_shifts(shift_dict, seed, shift_distr="flat")
