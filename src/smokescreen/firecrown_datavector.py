# author: Arthur Loureiro <arthur.loureiro@fysik.su.se>
# license: BSD 3-Clause
'''
Firecrown concealment path (:mod:`smokescreen.firecrown_datavector`)
===================================================================

.. currentmodule:: smokescreen.firecrown_datavector

Inherited-from-upstream firecrown-likelihood concealment. This module carries
the original DESC Smokescreen data-vector-blinding class, which drives theory
from a firecrown likelihood + CCL cosmology.

**This path is inherited from upstream and is not supported in this fork.**
It is retained so the fork does not amputate upstream's firecrown integration.
It is *not* wired into the fork's default flow: nothing in
:mod:`smokescreen`'s main modules imports it, no test in this fork exercises
it, and firecrown is neither declared as a dependency nor installed. All
firecrown (and ``pyccl``) imports are function-local, so importing this module
does not import firecrown --- and importing :mod:`smokescreen` does not import
this module. The supported concealment path is
:class:`smokescreen.datavector.ConcealDataVector`, which drives theory from a
``theory_fn`` protocol and ships a default CCL backend.


Firecrown Conceal Data Vector Class
-----------------------------------

.. autoclass:: FirecrownConcealDataVector
   :members:
   :undoc-members:
'''
import os
import types
import inspect
import datetime
import getpass
from copy import deepcopy

import numpy as np
import sacc


def _import_firecrown():
    """
    Lazily import the firecrown symbols the concealment path needs.

    Kept function-local so that importing this module (and, transitively,
    :mod:`smokescreen`) never imports firecrown. Firecrown is inherited from
    upstream and unsupported in this fork; it is not installed here.
    """
    from packaging.version import Version
    import firecrown

    # Handle different Firecrown versions
    if Version(firecrown.__version__) >= Version("1.15.0a0"):
        from firecrown.likelihood import (
            load_likelihood,
            load_likelihood_from_module_type,
            NamedParameters,
        )
    else:
        from firecrown.likelihood.likelihood import (
            load_likelihood,
            load_likelihood_from_module_type,
            NamedParameters,
        )
    from firecrown.parameters import ParamsMap
    from firecrown.updatable import get_default_params_map
    from firecrown.utils import save_to_sacc
    from firecrown.ccl_factory import PoweSpecAmplitudeParameter

    return {
        "load_likelihood": load_likelihood,
        "load_likelihood_from_module_type": load_likelihood_from_module_type,
        "NamedParameters": NamedParameters,
        "ParamsMap": ParamsMap,
        "get_default_params_map": get_default_params_map,
        "save_to_sacc": save_to_sacc,
        "PoweSpecAmplitudeParameter": PoweSpecAmplitudeParameter,
    }


def _load_module_from_path(path):
    """Load a module from a given filesystem path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modify_default_params(default_params, ccl_cosmology, systematics=None):
    """Override firecrown defaults with the CCL cosmology and systematics."""
    for key in default_params:
        if key in ccl_cosmology:
            default_params[key] = ccl_cosmology[key]
        elif systematics is not None and key in systematics:
            default_params[key] = systematics[key]
    return default_params


def _draw_flat_or_deterministic_param_shifts(cosmo, shifts_dict, seed):
    """
    Inherited draw: single values are deterministic absolute shifts, tuples are
    absolute ``(lo, hi)`` uniform bounds. Validated against CCL parameter names.
    Uses the process-global ``np.random`` for reproducibility with upstream.
    """
    from smokescreen.utils import string_to_seed

    if type(seed) is str:
        seed = string_to_seed(seed)
    np.random.seed(seed)

    for key in shifts_dict.keys():
        try:
            cosmo._params[key]
        except (AttributeError, KeyError) as error:
            raise ValueError(f"[{error}]Key {key} not in cosmology parameters")
    shifts = {}
    # loop over the ccl dict keys to ensure params are drawn in the same order
    for key in cosmo.to_dict().keys():
        if key in shifts_dict.keys():
            if isinstance(shifts_dict[key], tuple):
                if len(shifts_dict[key]) == 2:
                    shifts[key] = np.random.uniform(shifts_dict[key][0], shifts_dict[key][1])
                else:
                    raise ValueError(f"Tuple {shifts_dict[key]} has to be of length 2")
            else:
                shifts[key] = shifts_dict[key]
    return shifts


def _draw_gaussian_param_shifts(cosmo, shifts_dict, seed):
    """
    Inherited Gaussian draw: ``(mean, std)`` tuples, validated against CCL
    parameter names. Uses the process-global ``np.random``.
    """
    from smokescreen.utils import string_to_seed

    if type(seed) is str:
        seed = string_to_seed(seed)
    np.random.seed(seed)

    for key in shifts_dict.keys():
        try:
            cosmo._params[key]
        except (AttributeError, KeyError) as error:
            raise ValueError(f"[{error}]Key {key} not in cosmology parameters")
    shifts = {}
    for key in cosmo.to_dict().keys():
        if key in shifts_dict.keys():
            if isinstance(shifts_dict[key], tuple):
                if len(shifts_dict[key]) == 2:
                    shifts[key] = np.random.normal(shifts_dict[key][0], shifts_dict[key][1])
                else:
                    raise ValueError(f"Tuple {shifts_dict[key]} has to be of length 2")
            else:
                raise ValueError(f"Value {shifts_dict[key]} has to be a tuple of length 2")
    return shifts


class FirecrownConcealDataVector():
    """
    Inherited-from-upstream concealment driven by a firecrown likelihood.

    **Unsupported in this fork** --- retained from upstream, not tested, not
    installed, not wired into the default flow. Use
    :class:`smokescreen.datavector.ConcealDataVector` instead.

    FIXME: Only cosmological parameters are supported for now for the shifts

    Parameters
    ----------
    cosmo : pyccl.Cosmology
        Cosmology object from CCL with a fiducial cosmology.
    likelihood : str or module
        path to the likelihood or a module containing the likelihood
        must contain both `build_likelihood` and `compute_theory_vector` methods
    shifts_dict : dict
        Dictionary of parameter names and corresponding shift widths.
    sacc_data : sacc.sacc.Sacc
        Data-vector to be concealed (blinded).
    systm_dict : dict
        Dictionary of systematics names and corresponding fiducial values.
    seed : int or str
        Random seed.

    Keyword Arguments
    -----------------
    shift_distr : str
        Type of shift to be applied. Default is "flat".
    debug : bool
        If True, prints debug information. Default is False.
    """
    def __init__(self, cosmo, likelihood, shifts_dict, sacc_data, systm_dict=None,
                 seed="2112", **kwargs):
        self.cosmo = cosmo
        self.systematics_dict = systm_dict
        self.sacc_data = sacc_data
        self.seed = seed
        assert isinstance(self.sacc_data, sacc.sacc.Sacc), "sacc_data must be a sacc object"
        self.shifts_dict = shifts_dict

        # detect original file format for output preservation
        if 'input_format' in kwargs:
            self._input_format = kwargs['input_format']
        else:
            self._input_format = getattr(sacc_data, '_smokescreen_input_format', 'fits')

        # load the shifts
        if 'shift_distr' in kwargs:
            self.__shifts = self._load_shifts(seed, shift_distr=kwargs['shift_distr'])
        else:
            self.__shifts = self._load_shifts(seed)

        # create concealed cosmology object:
        self.__concealed_cosmo = self._create_concealed_cosmo(self.__shifts)

        if 'debug' in kwargs and kwargs['debug']:
            self._debug = True
            print(f"[DEBUG] Shifts: {self.__shifts}")
            print(f"[DEBUG] Concealed Cosmology: {self.__concealed_cosmo}")
        else:
            self._debug = False

        # load the likelihood
        self.likelihood, self.tools = self._load_likelihood(likelihood,
                                                            self.sacc_data)

        # load the systematics
        if self.systematics_dict is None:
            self.systematics = self._load_default_systematics(self.likelihood)
        else:
            self.systematics = self._load_systematics(self.systematics_dict, self.likelihood)

    def _load_likelihood(self, likelihood, sacc_data):
        """
        Loads the likelihood either from a python module or from a file.
        """
        fc = _import_firecrown()
        build_parameters = fc["NamedParameters"]({'sacc_data': sacc_data})

        if type(likelihood) is str:
            if not os.path.isfile(likelihood):
                raise FileNotFoundError(f'Could not find file {likelihood}')
            self._test_likelihood(likelihood, 'str')
            likelihood, tools = fc["load_likelihood"](likelihood, build_parameters)
            self._check_amplitude_parameter(tools)
            if not hasattr(likelihood, 'compute_theory_vector'):  # pragma: no cover
                raise AttributeError('Likelihood does not have a compute_vector method')
            return likelihood, tools

        elif isinstance(likelihood, types.ModuleType):
            self._test_likelihood(likelihood, 'module')
            likelihood, tools = fc["load_likelihood_from_module_type"](likelihood,
                                                                       build_parameters)
            self._check_amplitude_parameter(tools)
            if not hasattr(likelihood, 'compute_theory_vector'):  # pragma: no cover
                raise AttributeError('Likelihood does not have a compute_vector method')
            return likelihood, tools
        else:
            raise TypeError('Likelihood must be a string path to a likelihood module or a module')

    def _test_likelihood(self, likelihood, like_type):
        """Tests if the likelihood has the required methods."""
        if like_type == "str":
            likelihood = _load_module_from_path(likelihood)

        if not hasattr(likelihood, 'build_likelihood'):
            raise AttributeError('Likelihood does not have a build_likelihood method')

        if self.sacc_data is not None:
            sig = inspect.signature(likelihood.build_likelihood)
            likefunc_params = sig.parameters
            assert len(likefunc_params) >= 1, ("A sacc was provided, ",
                                               "the likelihood must require a",
                                               "build_parameters NamedParameters object!")

    def _check_amplitude_parameter(self, tools):
        """Checks the amplitude parameter set in the tools matches the concealed one."""
        fc = _import_firecrown()
        PoweSpecAmplitudeParameter = fc["PoweSpecAmplitudeParameter"]
        _amplitude_param = tools.ccl_factory.amplitude_parameter

        if _amplitude_param is PoweSpecAmplitudeParameter.SIGMA8:
            _required_param = 'sigma8'
        elif _amplitude_param is PoweSpecAmplitudeParameter.AS:
            _required_param = 'A_s'
        else:
            raise ValueError(f"Amplitude parameter {_amplitude_param} not supported")

        _error_msg = "\n You probably need to set the amplitude parameter [A_s/sigma8] "
        _error_msg += "that you want to conceal when calling ModelingTools in your likelihood "
        _error_msg += "module. \n The amplitude parameter is currently set to"
        _error_msg += f" {_amplitude_param} and Firecrown won't let Smokescreen change that."

        if _required_param not in self.cosmo.to_dict().keys():
            error_msg = f"Cosmology does not have the required parameter {_required_param}"
            error_msg += _error_msg
            raise ValueError(error_msg)

        if any(param in self.shifts_dict for param in ['A_s', 'sigma8']):
            if _required_param not in self.shifts_dict.keys():
                error_msg = "Shifts dictionary does not have the required parameter "
                error_msg += f"{_required_param}"
                error_msg += _error_msg
                raise ValueError(error_msg)

    def _load_default_systematics(self, likelihood):
        """Loads the default systematics from the likelihood."""
        fc = _import_firecrown()
        req_systematics = likelihood.required_parameters()
        default_systematics = req_systematics.get_default_values()
        return fc["ParamsMap"](default_systematics)

    def _load_systematics(self, systematics_dict, likelihood):
        """Loads the systematics from the systematics dictionary."""
        fc = _import_firecrown()
        likelihood_req_systematics = list(likelihood.required_parameters().get_params_names())
        if self._debug:
            print(f"[DEBUG] Likelihood requires systematics: {likelihood_req_systematics}")
        for key in likelihood_req_systematics:
            if key not in systematics_dict.keys():
                raise ValueError(f"Systematic {key} not in likelihood systematics")
        return fc["ParamsMap"](systematics_dict)

    def _load_shifts(self, seed, shift_distr="flat"):
        """Loads the shifts from the shifts dictionary."""
        if shift_distr == "flat":
            return _draw_flat_or_deterministic_param_shifts(self.cosmo, self.shifts_dict, seed)
        elif shift_distr == "gaussian":
            return _draw_gaussian_param_shifts(self.cosmo, self.shifts_dict, seed)
        else:
            raise NotImplementedError('Only flat and gaussian shifts are implemented')

    def _create_concealed_cosmo(self, shifts):
        """Creates a blinded cosmology object with the shifts applied."""
        import pyccl as ccl

        concealed_cosmo_dict = deepcopy(self.cosmo.to_dict())
        try:
            del concealed_cosmo_dict['extra_parameters']
        except KeyError:  # pragma: no cover
            pass
        for k in shifts.keys():
            concealed_cosmo_dict[k] = shifts[k]
        return ccl.Cosmology(**concealed_cosmo_dict)

    def calculate_concealing_factor(self, factor_type="add"):
        r"""
        Calculates the concealing (blinding) factor, per Muir et al. 2019.

        Notes
        -----
        type="add": :math:`f^{\rm add} = d(\theta_{\rm blind}) - d(\theta_{\rm fid})`

        type="mult": :math:`f^{\rm mult} = d(\theta_{\rm blind}) / d(\theta_{\rm fid})`
        """
        fc = _import_firecrown()
        self.factor_type = factor_type

        _firecrown_defaults = fc["get_default_params_map"](self.tools, self.likelihood)

        _params_reference = _modify_default_params(_firecrown_defaults,
                                                   self.cosmo.to_dict(),
                                                   self.systematics_dict)
        self.tools.update(_params_reference)
        self.tools.prepare()
        self.likelihood.update(_params_reference)
        self.theory_vec_fid = self.likelihood.compute_theory_vector(self.tools)
        self.likelihood.reset()
        self.tools.reset()

        __params_concealed = _modify_default_params(_firecrown_defaults,
                                                    self.__concealed_cosmo.to_dict(),
                                                    self.systematics_dict)
        self.tools.update(__params_concealed)
        self.tools.prepare()
        self.likelihood.update(__params_concealed)
        self.theory_vec_conceal = self.likelihood.compute_theory_vector(self.tools)

        if self.factor_type == "add":
            self.__concealing_factor = self.theory_vec_conceal - self.theory_vec_fid
        elif self.factor_type == "mult":
            self.__concealing_factor = self.theory_vec_conceal / self.theory_vec_fid
        else:
            raise NotImplementedError('Only "add" and "mult" concealing factor is implemented')
        if self._debug:
            return self.__concealing_factor

    def apply_concealing_to_likelihood_datavec(self):
        r"""Applies the concealing (blinding) factor to the data-vector."""
        self.data_vector = self.likelihood.get_data_vector()
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
        """Saves the concealed (blinded) data-vector to a file."""
        fc = _import_firecrown()
        if suffix is None:
            suffix = "concealed_data_vector"
        if output_format is None:
            output_format = getattr(self, '_input_format', 'fits')

        idx = self.likelihood.get_sacc_indices()
        concealed_sacc = fc["save_to_sacc"](self.sacc_data,
                                            self.concealed_data_vector,
                                            idx)
        concealed_sacc.metadata = self.sacc_data.metadata
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
