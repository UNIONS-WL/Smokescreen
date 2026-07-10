# author: Arthur Loureiro <arthur.loureiro@fysik.su.se>
# license: BSD 3-Clause
'''
Default CCL cosmic-shear backend (:mod:`smokescreen.backends.ccl`)
=================================================================

.. currentmodule:: smokescreen.backends.ccl

Builds a default ``theory_fn`` that computes a cosmic-shear theory vector from
CCL for a standard weak-lensing SACC file. ``pyccl`` is imported here, so it
enters ``sys.modules`` only once this backend is constructed --- importing
``smokescreen`` does not import it.

The backend accepts CCL-native cosmological-parameter names
(``sigma8``, ``Omega_c``, ``Omega_b``, ``h``, ``n_s``, ...). It reads the
weak-lensing tracers' n(z) and the SACC row layout, and returns, for a given
parameter mapping, the theory vector aligned element-for-element to
``sacc_data.mean``.

.. autofunction:: build_ccl_theory_fn
'''
import numpy as np
import pyccl as ccl


# SACC data types the default backend understands, and the CCL correlation
# 'type' string (Fourier -> real space) each ξ± measurement maps to.
_XI_TYPES = {
    "galaxy_shear_xi_plus": "GG+",
    "galaxy_shear_xi_minus": "GG-",
}
_CL_TYPES = {"galaxy_shear_cl_ee"}


def _tracer_nz(sacc_data, name):
    """Return (z, nz) for a SACC tracer carrying a redshift distribution."""
    tracer = sacc_data.tracers[name]
    if not (hasattr(tracer, "z") and hasattr(tracer, "nz")):
        raise ValueError(
            f"Default CCL backend requires n(z) tracers; tracer '{name}' "
            f"({tracer.tracer_type}) carries no redshift distribution."
        )
    return np.asarray(tracer.z), np.asarray(tracer.nz)


def _plan_blocks(sacc_data):
    """
    Decompose the SACC into cosmic-shear blocks aligned to ``sacc_data.mean``.

    Returns a list of ``(data_type, tracer_pair, angle_values, row_indices)``,
    one per (data_type, tracer-pair) group the default backend supports. Raises
    if the SACC carries a data type this backend does not model.
    """
    supported = set(_XI_TYPES) | _CL_TYPES
    present = set(sacc_data.get_data_types())
    unsupported = present - supported
    if unsupported:
        raise ValueError(
            f"Default CCL backend only models cosmic-shear data types "
            f"{sorted(supported)}; SACC carries unsupported {sorted(unsupported)}. "
            f"Supply an explicit theory_fn for this SACC."
        )

    blocks = []
    for data_type in present:
        angle_key = "theta" if data_type in _XI_TYPES else "ell"
        for pair in sacc_data.get_tracer_combinations(data_type):
            idx = sacc_data.indices(data_type, pair)
            angles = np.array(
                [sacc_data.data[i].get_tag(angle_key) for i in idx]
            )
            blocks.append((data_type, pair, angles, np.asarray(idx)))
    return blocks


def build_ccl_theory_fn(sacc_data):
    """
    Build a default cosmic-shear ``theory_fn`` from a SACC file.

    Parameters
    ----------
    sacc_data : sacc.sacc.Sacc
        SACC carrying weak-lensing tracers with n(z) and cosmic-shear rows
        (``galaxy_shear_cl_ee`` and/or ``galaxy_shear_xi_plus`` /
        ``galaxy_shear_xi_minus``). The row layout it exposes is the layout the
        returned callable aligns to.

    Returns
    -------
    Callable[[Mapping[str, float]], np.ndarray]
        A ``theory_fn``: given a mapping of CCL-native cosmological-parameter
        names to values, it constructs a :class:`pyccl.Cosmology`, builds a
        :class:`pyccl.WeakLensingTracer` per bin, and returns the cosmic-shear
        theory vector aligned to ``sacc_data.mean``.
    """
    blocks = _plan_blocks(sacc_data)
    n_rows = len(sacc_data.mean)
    nz_cache = {
        name: _tracer_nz(sacc_data, name)
        for block in blocks
        for name in block[1]
    }
    # ℓ grid for the Cℓ used by the real-space correlation transform.
    ell_grid = np.unique(np.geomspace(2, 30000, 512).astype(int))

    def theory_fn(cosmo_params):
        # Default to pyccl-native transfer/power (no CAMB/CLASS dependency), so
        # the shipped backend runs against a bare pyccl install. A caller-supplied
        # theory_fn is free to route through a Boltzmann code.
        params = {"transfer_function": "eisenstein_hu",
                  "matter_power_spectrum": "halofit"}
        params.update(cosmo_params)
        cosmo = ccl.Cosmology(**params)
        wl = {
            name: ccl.WeakLensingTracer(cosmo, dndz=nz_cache[name])
            for name in nz_cache
        }
        vec = np.empty(n_rows)
        for data_type, (t1, t2), angles, idx in blocks:
            cl = ccl.angular_cl(cosmo, wl[t1], wl[t2], ell_grid)
            if data_type in _CL_TYPES:
                values = np.interp(angles, ell_grid, cl)
            else:
                theta_deg = angles / 60.0  # arcmin -> degrees
                values = ccl.correlation(
                    cosmo, ell=ell_grid, C_ell=cl, theta=theta_deg,
                    type=_XI_TYPES[data_type],
                )
            vec[idx] = values
        return vec

    return theory_fn
