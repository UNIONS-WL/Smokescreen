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

from smokescreen.param_shifts import draw_param_shifts


def _check_shift_keys(fiducial_params, shifts_dict):
    """Every shifted parameter must have a fiducial value; raise if one does not."""
    unknown = set(shifts_dict) - set(fiducial_params)
    if unknown:
        raise ValueError(
            f"shifts_dict keys {sorted(unknown)} are not keys of "
            f"fiducial_params {sorted(fiducial_params)}; every "
            f"shifted parameter must have a fiducial value."
        )


class ConcealDataVector():
    """
    Conceal (blind) a measured data vector by adding a theory difference.

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
        self.shifts_dict = shifts_dict
        _check_shift_keys(self.fiducial_params, shifts_dict)
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

        # draw the hidden shift and overlay it on the fiducial point
        self.__shifts = draw_param_shifts(self.shifts_dict, seed,
                                          shift_distr=shift_distr)
        self.__concealed_params = self._overlay_shifts(self.__shifts)

        # shape guard: theory vector must align to the SACC rows
        self.theory_vec_fid = self._checked_theory_vec(self.fiducial_params)

        if self._debug:
            print(f"[DEBUG] Shifts: {self.__shifts}")
            print(f"[DEBUG] Concealed params: {self.__concealed_params}")

    def _checked_theory_vec(self, params):
        """Evaluate theory_fn and enforce a 1-D vector aligned to sacc_data.mean."""
        vec = np.asarray(self.theory_fn(params))
        if vec.shape != np.shape(self.sacc_data.mean):
            raise ValueError(
                f"theory_fn returned shape {vec.shape} but sacc_data.mean has "
                f"shape {np.shape(self.sacc_data.mean)}; theory_fn must return "
                f"a 1-D vector aligned to the SACC rows."
            )
        return vec

    def _overlay_shifts(self, shifts):
        """Overlay drawn deltas on the fiducial point: fiducial[k] + shift[k]."""
        concealed = deepcopy(self.fiducial_params)
        for k, delta in shifts.items():
            concealed[k] = self.fiducial_params[k] + delta
        return concealed

    def calculate_concealing_factor(self, factor_type="add"):
        r"""
        Calculate the concealing (blinding) factor, per Muir et al. 2019.

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
        self.factor_type = factor_type

        # fiducial vector computed once in __init__ (shape guard); recompute
        # nothing there --- reuse it and evaluate only the hidden point here.
        self.theory_vec_conceal = self._checked_theory_vec(self.__concealed_params)

        if factor_type == "add":
            self.__concealing_factor = self.theory_vec_conceal - self.theory_vec_fid
        elif factor_type == "mult":
            self.__concealing_factor = self.theory_vec_conceal / self.theory_vec_fid
        else:
            raise NotImplementedError('Only "add" and "mult" concealing factor is implemented')
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
                                  suffix=None):
        """
        Save the concealed (blinded) data vector to a SACC file.

        The blinded vector overwrites the mean of a deep copy of ``sacc_data``;
        the covariance is carried over unchanged (blinding shifts the mean and
        never touches the covariance). Metadata is stamped with the concealed
        flag, creator, timestamp, and seed. Writing goes through SACC's own
        ``save_fits`` / ``save_hdf5``.

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
