# author: Arthur Loureiro <arthur.loureiro@fysik.su.se>
# license: BSD 3-Clause
'''
Conceal Data Vector (:mod:`smokescreen.datavector`)
====================================================

.. currentmodule:: smokescreen.datavector

The :mod:`smokescreen.datavector` module provides functionalities
to conceal (blind) data vectors in the context of cosmological analysis.

Blinding adds a theory difference to the data vector,
``d -> d + t(hidden) - t(fiducial)``. The theory source is a single
``theory_fn(cosmo_params) -> np.ndarray`` protocol: any callable returning a
vector aligned to the SACC rows being concealed is a valid backend. The fork
ships a default CCL cosmic-shear backend (see
:mod:`smokescreen.backends.ccl`); power users supply their own callable.

The blinding itself is a vector operation: :func:`concealing_factor` is the
whole of it, and it never sees a SACC. It splits in two --- draw the hidden
point from the seed, then combine the two theory vectors --- and the second
half, :func:`factor_from_params`, is exposed for callers that already hold
their hidden point and must not draw it a second time.
:class:`ConcealDataVector` is the SACC adapter over that function --- it reads
the rows, applies the factor, and writes the blinded file. Callers that hold
their own data container (a pipeline blinding selected rows of a larger file,
say) want the function.


Concealing Factor
-----------------

.. autofunction:: concealing_factor

.. autofunction:: factor_from_params


Conceal Data Vector Class
-------------------------

.. autoclass:: ConcealDataVector
   :members:
   :undoc-members:
'''
import datetime
import getpass
from copy import deepcopy

import numpy as np
import sacc

from smokescreen.param_shifts import DRAW_SCHEME, draw_param_shifts, seed_commitment


def _check_shift_keys(fiducial_params, shifts_dict):
    """Every shifted parameter must have a fiducial value; raise if one does not."""
    unknown = set(shifts_dict) - set(fiducial_params)
    if unknown:
        raise ValueError(
            f"shifts_dict keys {sorted(unknown)} are not keys of "
            f"fiducial_params {sorted(fiducial_params)}; every "
            f"shifted parameter must have a fiducial value."
        )


def _overlay_shifts(fiducial_params, shifts):
    """Overlay drawn deltas on the fiducial point: ``fiducial[k] + shift[k]``."""
    concealed = deepcopy(dict(fiducial_params))
    for k, delta in shifts.items():
        concealed[k] = fiducial_params[k] + delta
    return concealed


def concealing_factor(fiducial_params, shifts_dict, *, seed, theory_fn,
                      shift_distr="flat", factor_type="add"):
    r"""
    Concealing (blinding) factor for a hidden parameter shift, per Muir et al. 2019.

    The whole of the blinding operation, as a plain vector: draw the hidden
    deltas from ``seed``, overlay them on the fiducial point, evaluate
    ``theory_fn`` at both points, and combine the two theory vectors. No SACC
    and no data vector enter here --- the caller adds (or multiplies) the
    returned factor into whatever container holds its data.

    Parameters
    ----------
    fiducial_params : Mapping[str, float]
        The fiducial point, a plain mapping of cosmological-parameter name to
        value. Interpreting the names is the ``theory_fn``'s job.
    shifts_dict : Mapping[str, float | tuple[float, float]]
        Maps a parameter name (a key of ``fiducial_params``) to its shift
        envelope, interpreted as a *delta* about zero. See
        :func:`smokescreen.param_shifts.draw_param_shifts`.
    seed : int or str
        Random seed. Keyword-only, and no default: blinding custody depends on
        the seed being a deliberate, secret choice.
    theory_fn : Callable[[Mapping[str, float]], np.ndarray]
        Theory backend. Keyword-only and required --- there is no default
        backend at this level, so nothing here imports ``pyccl``. To build the
        shipped cosmic-shear backend from a SACC, use
        :func:`smokescreen.backends.ccl.build_ccl_theory_fn`.
    shift_distr : str
        ``"flat"`` (uniform over the delta envelope) or ``"gaussian"``
        (zero-mean Gaussian with sigma = the ``float`` half-width). Default
        ``"flat"``.
    factor_type : str
        ``"add"`` (default) or ``"mult"``.

    Returns
    -------
    np.ndarray
        The concealing factor, in the row order ``theory_fn`` returns.

    Notes
    -----
    factor_type="add":
        .. math:: f^{\rm add} = t(\theta_{\rm hidden}) - t(\theta_{\rm fid})

    factor_type="mult":
        .. math:: f^{\rm mult} = t(\theta_{\rm hidden}) / t(\theta_{\rm fid})

    See Also
    --------
    factor_from_params : the same combination, for a hidden point already drawn.
    """
    fiducial_params = dict(fiducial_params)
    _check_shift_keys(fiducial_params, shifts_dict)

    shifts = draw_param_shifts(shifts_dict, seed, shift_distr=shift_distr)
    concealed_params = _overlay_shifts(fiducial_params, shifts)

    return factor_from_params(fiducial_params, concealed_params,
                              theory_fn=theory_fn, factor_type=factor_type)


def factor_from_params(fiducial_params, concealed_params, *, theory_fn,
                       factor_type="add"):
    r"""
    Concealing factor for a hidden point that has already been drawn.

    The lower half of :func:`concealing_factor`: no seed, no draw, no shift
    envelope --- just the two theory evaluations and their combination. Use it
    when the hidden point is already in hand, so that it is drawn exactly once
    and the factor cannot drift from the parameters the caller believes
    produced it.

    Parameters
    ----------
    fiducial_params : Mapping[str, float]
        The fiducial point.
    concealed_params : Mapping[str, float]
        The hidden point: the fiducial point with the drawn deltas overlaid.
    theory_fn : Callable[[Mapping[str, float]], np.ndarray]
        Theory backend, keyword-only and required. See
        :func:`concealing_factor`.
    factor_type : str
        ``"add"`` (default) or ``"mult"``.

    Returns
    -------
    np.ndarray
        The concealing factor, in the row order ``theory_fn`` returns.
    """
    if factor_type not in ("add", "mult"):
        raise NotImplementedError('Only "add" and "mult" concealing factor is implemented')

    theory_fid = np.asarray(theory_fn(fiducial_params))
    theory_conceal = np.asarray(theory_fn(concealed_params))
    if theory_conceal.shape != theory_fid.shape:
        raise ValueError(
            f"theory_fn returned shape {theory_conceal.shape} at the hidden "
            f"point but {theory_fid.shape} at the fiducial point; the two must "
            f"span the same rows."
        )

    if factor_type == "add":
        return theory_conceal - theory_fid
    return theory_conceal / theory_fid


class ConcealDataVector():
    """
    Conceal (blind) a measured data vector by adding a theory difference.

    The SACC adapter over :func:`concealing_factor`: it reads the rows the
    factor must span, applies the factor to ``sacc_data.mean``, and writes the
    blinded file. The blinding arithmetic itself lives in that function, which
    knows nothing about SACC.

    Both theory vectors (fiducial and hidden) come from a single ``theory_fn``:
    the shipped default CCL backend when none is supplied, else the caller's.
    There is no likelihood, no ``pyccl.Cosmology`` object on this path, and no
    systematics dictionary transiting the class --- systematics, if a backend
    needs them, are closed over inside that backend's ``theory_fn``.

    Parameters
    ----------
    fiducial_params : Mapping[str, float]
        The fiducial point, a plain mapping of cosmological-parameter name to
        value over a free-form parameter space (CCL-native names, e.g.
        ``sigma8``, ``Omega_c``). Interpreting the names is the ``theory_fn``'s
        job; this is not a ``pyccl.Cosmology``.
    shifts_dict : Mapping[str, float | tuple[float, float]]
        Maps a parameter name (a key of ``fiducial_params``) to its shift
        envelope, interpreted as a *delta* about zero. A ``float`` ``h`` is the
        symmetric delta envelope ``(-h, +h)``; a ``(lo, hi)`` tuple is a delta
        box. The drawn delta is added to ``fiducial_params[k]``.
    sacc_data : sacc.sacc.Sacc
        Data vector to be concealed. It must contain exactly the rows the
        ``theory_fn`` returns, in ``sacc_data.mean`` order (extract-then-blind:
        the caller scopes to the to-be-blinded block before construction).
    seed : int or str
        Random seed. No default: blinding custody depends on the seed being a
        deliberate, secret choice. A ``str`` is normalized to an ``int``.
    theory_fn : Callable[[Mapping[str, float]], np.ndarray], optional
        Theory backend. When ``None``, the default CCL cosmic-shear backend is
        built from ``sacc_data``.
    shift_distr : str
        ``"flat"`` (uniform over the delta envelope) or ``"gaussian"``
        (zero-mean Gaussian with sigma = the ``float`` half-width). Default
        ``"flat"``.
    debug : bool
        If True, prints debug information and makes
        :meth:`calculate_concealing_factor` return the factor. Default False.
    input_format : str, optional
        Original SACC file format for output preservation (``'fits'`` or
        ``'hdf5'``). When ``None``, taken from the SACC object if
        :func:`smokescreen.utils.load_sacc_file` tagged it, else ``'fits'``.

    Notes
    -----
    The keyword arguments are spelled out rather than collected in
    ``**kwargs``: a misspelled ``theory_fn`` would otherwise be swallowed
    silently and fall back to the default CCL backend, blinding with one theory
    and unblinding with another.
    """
    def __init__(self, fiducial_params, shifts_dict, sacc_data, *,
                 seed, theory_fn=None, shift_distr="flat",
                 debug=False, input_format=None):
        assert isinstance(sacc_data, sacc.sacc.Sacc), "sacc_data must be a sacc object"

        self.fiducial_params = dict(fiducial_params)
        self.shifts_dict = dict(shifts_dict)
        _check_shift_keys(self.fiducial_params, self.shifts_dict)
        self.sacc_data = sacc_data
        self.seed = seed
        self.shift_distr = shift_distr
        self._debug = bool(debug)

        # detect original file format for output preservation
        if input_format is None:
            input_format = getattr(sacc_data, '_smokescreen_input_format', 'fits')
        self._input_format = input_format

        # theory backend: caller-supplied, or the default CCL cosmic-shear one
        # (imported lazily so `import smokescreen` never imports pyccl).
        if theory_fn is None:
            from smokescreen.backends.ccl import build_ccl_theory_fn
            self.theory_fn = build_ccl_theory_fn(sacc_data)
        else:
            self.theory_fn = theory_fn

        # The hidden point, drawn once here and used everywhere after: the
        # factor comes from factor_from_params, not from a second draw, so
        # theory_vec_conceal and the factor cannot disagree.
        self.__shifts = draw_param_shifts(self.shifts_dict, seed,
                                          shift_distr=shift_distr)
        self.__concealed_params = _overlay_shifts(self.fiducial_params,
                                                  self.__shifts)

        # shape guard: theory vector must align to the SACC rows
        self._theory_cache = {}
        self.theory_vec_fid = self._checked_theory_vec(self.fiducial_params)

        if self._debug:
            print(f"[DEBUG] Shifts: {self.__shifts}")
            print(f"[DEBUG] Concealed params: {self.__concealed_params}")

    def _checked_theory_vec(self, params):
        """Evaluate theory_fn and enforce a 1-D vector aligned to sacc_data.mean.

        This is the alignment guard the SACC coupling buys, so it wraps every
        theory evaluation the class makes --- including the two inside
        :func:`concealing_factor`. Memoized on the parameter point, so the
        fiducial and hidden vectors are each computed exactly once however many
        times they are asked for.
        """
        try:
            key = tuple(sorted(params.items()))
            hit = self._theory_cache.get(key)
        except TypeError:  # uncomparable or unhashable value: skip the memo
            key, hit = None, None
        if hit is not None:
            return hit

        vec = np.asarray(self.theory_fn(params))
        if vec.shape != np.shape(self.sacc_data.mean):
            raise ValueError(
                f"theory_fn returned shape {vec.shape} but sacc_data.mean has "
                f"shape {np.shape(self.sacc_data.mean)}; theory_fn must return "
                f"a 1-D vector aligned to the SACC rows."
            )
        if key is not None:
            self._theory_cache[key] = vec
        return vec

    def calculate_concealing_factor(self, factor_type="add"):
        r"""
        Calculate the concealing (blinding) factor, per Muir et al. 2019.

        Delegates to :func:`factor_from_params` with the hidden point drawn at
        construction, evaluating the theory through the class's alignment
        guard. The draw is never repeated, so the factor and
        ``theory_vec_conceal`` always describe the same hidden point.

        Parameters
        ----------
        factor_type : str
            ``"add"`` (default) or ``"mult"``.

        Returns
        -------
        np.ndarray
            Concealing factor (returned only in debug mode).

        Notes
        -----
        type="add":
            .. math:: f^{\rm add} = t(\theta_{\rm hidden}) - t(\theta_{\rm fid})

        type="mult":
            .. math:: f^{\rm mult} = t(\theta_{\rm hidden}) / t(\theta_{\rm fid})
        """
        factor = factor_from_params(
            self.fiducial_params, self.__concealed_params,
            theory_fn=self._checked_theory_vec, factor_type=factor_type,
        )
        # both theory points are memoized by now, so this costs no extra call
        self.theory_vec_conceal = self._checked_theory_vec(self.__concealed_params)
        # only once the delegation succeeded: a rejected factor_type must leave
        # the object exactly as it was, not half-updated.
        self.factor_type = factor_type
        self.__concealing_factor = factor

        if self._debug:
            return self.__concealing_factor

    def apply_concealing_to_likelihood_datavec(self):
        r"""
        Apply the concealing (blinding) factor to the SACC data vector.

        Returns
        -------
        np.ndarray
            Concealed (blinded) data vector: ``sacc_data.mean + factor``
            (``"add"``) or ``sacc_data.mean * factor`` (``"mult"``).
        """
        self.data_vector = self.sacc_data.mean
        if self.factor_type == "add":
            self.concealed_data_vector = self.data_vector + self.__concealing_factor
        elif self.factor_type == "mult":
            self.concealed_data_vector = self.data_vector * self.__concealing_factor
        else:
            raise NotImplementedError('Only "add" and "mult" blinding factor is implemented')
        return self.concealed_data_vector

    def save_concealed_datavector(self, path_to_save, file_root,
                                  return_sacc=False, output_format=None,
                                  suffix=None, stamp_seed=False):
        """
        Save the concealed (blinded) data vector to a SACC file.

        The blinded vector overwrites the mean of a deep copy of ``sacc_data``;
        the covariance is carried over unchanged. Metadata is stamped with the
        concealed flag, creator, timestamp,
        :func:`~smokescreen.param_shifts.seed_commitment` (a digest, not the
        seed), and :data:`~smokescreen.param_shifts.DRAW_SCHEME`, so an
        unblind attempt with different draw semantics can be refused rather
        than silently subtract the wrong shift. Writing goes through SACC's
        own ``save_fits`` / ``save_hdf5``.

        Parameters
        ----------
        path_to_save : str
            Directory to save the blinded data vector.
        file_root : str
            Root of the file name.
        return_sacc : bool
            If True, returns the sacc object with the blinded data vector.
        output_format : str, optional
            Output format ('fits' or 'hdf5'). Defaults to the detected input
            format.
        suffix : str, optional
            Suffix for the output file name. Defaults to
            'concealed_data_vector'.
        stamp_seed : bool
            If True, additionally stamp the raw seed as ``seed_smokescreen``,
            as upstream Smokescreen does. Default False: the seed plus the
            (public) config is enough to reconstruct the shift, so the file
            that circulates while blind should not carry it.

        Returns
        -------
        sacc.sacc.Sacc or None
        """
        if suffix is None:
            suffix = "concealed_data_vector"
        if output_format is None:
            output_format = getattr(self, '_input_format', 'fits')

        concealed_sacc = deepcopy(self.sacc_data)
        concealed_sacc.mean = self.concealed_data_vector

        # copy metadata from the original sacc file, then stamp custody info:
        concealed_sacc.metadata = dict(self.sacc_data.metadata)
        concealed_sacc.metadata['concealed'] = True
        concealed_sacc.metadata['creator'] = getpass.getuser()
        concealed_sacc.metadata['creation'] = datetime.datetime.now().isoformat()
        concealed_sacc.metadata['info'] = 'Concealed (blinded) data-vector, created by Smokescreen.'
        concealed_sacc.metadata['seed_commitment'] = seed_commitment(self.seed)
        concealed_sacc.metadata['draw_scheme'] = DRAW_SCHEME
        if stamp_seed:
            concealed_sacc.metadata['seed_smokescreen'] = self.seed

        if output_format == 'hdf5':
            ext = '.hdf5'
            save_method = concealed_sacc.save_hdf5
        else:  # default to FITS
            ext = '.fits'
            save_method = concealed_sacc.save_fits

        output_path = f"{path_to_save}/{file_root}_{suffix}{ext}"
        save_method(output_path, overwrite=True)
        if return_sacc:
            return concealed_sacc
        return None
