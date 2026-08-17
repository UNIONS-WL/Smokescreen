Usage
======

Currently, only data vector concealment is implemented in Smokescreen.
Posterior-level concealment is under development.

Data Vector Concealment (blinding)
-----------------------------------

Smokescreen conceals a data vector by adding a theory difference to it,
following the `Muir et al. (2019) <https://arxiv.org/abs/1911.05929>`_ method:

.. math::

   d \rightarrow d + t(\theta_{\rm hidden}) - t(\theta_{\rm fid})

The hidden point is the fiducial point plus a secret, seeded parameter shift.
Anyone who knows the seed and the (public) configuration can reconstruct the
shift and unblind; anyone who does not, cannot.

To conceal a data vector you need four things:

* **Fiducial parameters** --- a plain mapping of parameter name to value.
  With the default backend these are CCL-native names, e.g.
  ``{"sigma8": 0.8, "Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67, "n_s": 0.96}``.
  This is a mapping, not a ``pyccl.Cosmology`` object.

* **A shifts dictionary** --- the envelope each shifted parameter is drawn
  from. See `Shift envelopes`_ below.

* **A SACC data vector** --- containing exactly the rows to be concealed.

* **A seed** --- an ``int`` or ``str``. It is keyword-only and has no default:
  blinding custody depends on the seed being a deliberate, secret choice.

The theory itself comes from a single callable,
``theory_fn(cosmo_params) -> np.ndarray``. When you do not supply one, the
built-in CCL cosmic-shear backend is built from your SACC file. There is no
likelihood object and no systematics dictionary on this path.

Shift envelopes
~~~~~~~~~~~~~~~~

Every value in ``shifts_dict`` is a **delta about zero**, added to the fiducial
value. It is not an absolute parameter value and not a range of parameter
values.

.. code-block:: python

   # symmetric delta envelope: the drawn delta lies in (-0.05, +0.05),
   # so the hidden sigma8 lies in (fid - 0.05, fid + 0.05)
   {"sigma8": 0.05}

   # explicit delta box: the drawn delta lies in [lo, hi]
   {"Omega_c": (-0.03, 0.05)}

   # gaussian draws take only the float form; the value is the sigma of a
   # zero-mean gaussian delta
   {"sigma8": 0.02}   # with shift_distr="gaussian"

Every key of ``shifts_dict`` must also be a key of ``fiducial_params``;
construction raises ``ValueError`` if one is not.

Draws are **per-key independent**: the delta for a key depends only on
``(key, seed, shift_distr)``, never on the other keys present or on the
iteration order of the mapping. These semantics are versioned by
:data:`smokescreen.param_shifts.DRAW_SCHEME`, which is stamped into every
blinded file.

From a notebook or your code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from smokescreen import ConcealDataVector
   from smokescreen.utils import load_sacc_file

   sacc_data, input_format = load_sacc_file("cosmicshear_sacc.fits")

   fiducial_params = {"sigma8": 0.8, "Omega_c": 0.25,
                      "Omega_b": 0.05, "h": 0.67, "n_s": 0.96}
   shifts_dict = {"sigma8": 0.05, "Omega_c": (-0.03, 0.05)}

   smoke = ConcealDataVector(fiducial_params, shifts_dict, sacc_data,
                             seed="a-high-entropy-secret",
                             input_format=input_format)

   smoke.calculate_concealing_factor()          # factor_type="add" by default
   concealed_dv = smoke.apply_concealing_to_likelihood_datavec()

   smoke.save_concealed_datavector("./output", "cosmicshear_sacc")

The full signature is keyword-only after ``sacc_data``::

   ConcealDataVector(fiducial_params, shifts_dict, sacc_data, *,
                     seed, theory_fn=None, shift_distr="flat",
                     debug=False, input_format=None)

``shift_distr`` is ``"flat"`` (default) or ``"gaussian"``.
``calculate_concealing_factor(factor_type="add")`` also accepts ``"mult"``,
which divides rather than subtracts and is applied multiplicatively.
With ``debug=True`` the drawn shifts are printed and
``calculate_concealing_factor`` returns the factor.

The concealing factor alone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The blinding arithmetic is a pure vector operation, and
:func:`~smokescreen.datavector.concealing_factor` is the whole of it. It never
touches a SACC file. Reach for it when you hold your own data container --- for
example a pipeline that blinds selected rows of a larger file:

.. code-block:: python

   from smokescreen import concealing_factor

   factor = concealing_factor(fiducial_params, shifts_dict,
                              seed="a-high-entropy-secret",
                              theory_fn=my_theory_fn)
   blinded = my_data_vector + factor

Here ``theory_fn`` is required and keyword-only: at this level there is no
default backend, so nothing in this call path imports ``pyccl``.

If you have already drawn the hidden point yourself, use
:func:`~smokescreen.datavector.factor_from_params` instead. It is the lower
half of the same operation --- no seed, no draw, just the two theory
evaluations and their combination --- so the hidden point is drawn exactly
once and the factor cannot drift from the parameters you believe produced it:

.. code-block:: python

   from smokescreen import factor_from_params
   from smokescreen.param_shifts import draw_param_shifts

   shifts = draw_param_shifts(shifts_dict, seed="a-high-entropy-secret")
   concealed = {k: v + shifts.get(k, 0.0) for k, v in fiducial_params.items()}
   factor = factor_from_params(fiducial_params, concealed,
                               theory_fn=my_theory_fn)

The default CCL backend and its SACC contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``theory_fn`` is ``None``,
:class:`~smokescreen.datavector.ConcealDataVector` builds the default backend
with :func:`smokescreen.backends.ccl.build_ccl_theory_fn`. That backend imposes
a contract on the SACC file:

* It models **cosmic shear only** --- data types ``galaxy_shear_cl_ee``,
  ``galaxy_shear_xi_plus`` and ``galaxy_shear_xi_minus``. Any other data type
  in the file raises; supply your own ``theory_fn`` for such a SACC.

* Every tracer used by those rows must carry a redshift distribution
  (``z``/``nz``); the backend builds one ``pyccl.WeakLensingTracer`` per bin.

* The returned vector is aligned element-for-element to ``sacc_data.mean``, and
  the class enforces this: a ``theory_fn`` whose output shape does not match
  raises ``ValueError``. Extract first, then blind --- scope the SACC to the
  block you intend to conceal before construction.

Two conventions the backend assumes and does not verify:

* The ``theta`` tag of ``xi_plus``/``xi_minus`` rows is read as
  **arcminutes** and converted to degrees for ``pyccl.correlation``.
  Angular power spectra are computed once on an internal 512-point log-spaced
  grid over :math:`\ell \in [2, 30000]`; ``galaxy_shear_cl_ee`` rows are
  linearly interpolated from that grid onto their ``ell`` tag rather than
  evaluated at it.

* The cosmology defaults to ``transfer_function="eisenstein_hu"`` and
  ``matter_power_spectrum="halofit"``, so the backend runs against a bare
  ``pyccl`` install with no Boltzmann code. Both are overridable by putting
  them in ``fiducial_params``. For a Boltzmann-code power spectrum, supply your
  own ``theory_fn``.

Supplying your own theory backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any callable that takes a parameter mapping and returns a vector aligned to the
rows being concealed is a valid backend. Systematics, tracer choices and
nuisance parameters are closed over inside it --- they never transit
Smokescreen:

.. code-block:: python

   def my_theory_fn(cosmo_params):
       # ... your own model, using whatever machinery you like ...
       return theory_vector      # 1-D, aligned to sacc_data.mean

   smoke = ConcealDataVector(fiducial_params, shifts_dict, sacc_data,
                             seed=2112, theory_fn=my_theory_fn)

.. note::
   The firecrown integration path is inherited from upstream
   `LSSTDESC/Smokescreen <https://github.com/LSSTDESC/Smokescreen>`_ and is
   unsupported in this fork: not installed, not tested, not maintained. The
   default and supported theory path is the built-in CCL backend, or a
   ``theory_fn`` of your own.

What is written to the blinded file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``save_concealed_datavector(path_to_save, file_root, return_sacc=False,
output_format=None, suffix=None, stamp_seed=False)`` writes a deep copy of the
input SACC with the blinded mean. The covariance is carried over untouched ---
blinding shifts the mean and never the covariance. The output format defaults
to the detected input format, and the file name is
``{file_root}_{suffix}{ext}`` with ``suffix`` defaulting to
``concealed_data_vector``.

The following metadata is stamped:

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``concealed``
     - ``True``
   * - ``creator``, ``creation``, ``info``
     - Who blinded the file, when, and with what
   * - ``seed_commitment``
     - sha256 hex digest of
       :data:`~smokescreen.param_shifts.COMMITMENT_DOMAIN` followed by
       ``str(seed)`` --- a commitment to the seed, **not** the seed
   * - ``draw_scheme``
     - The :data:`~smokescreen.param_shifts.DRAW_SCHEME` the shifts were drawn under

.. warning::
   The raw seed is **not** written by default. The commitment ties the blinded
   product to the blind that produced it: a later-revealed seed either matches
   the digest or it does not. The digest is domain-separated, so it does not
   share a hash domain with the RNG seed derivation and cannot carry the seed
   itself. It still conceals the seed only to the extent that the seed is
   unguessable --- a small integer is trivially brute-forced from its digest,
   so choose a high-entropy seed. Pass ``stamp_seed=True`` to write the
   raw seed as ``seed_smokescreen``, as upstream Smokescreen does.

``draw_scheme`` lets an unblind attempt made with an install whose draw
semantics differ fail loudly, rather than silently subtracting the wrong shift.

From the command line
~~~~~~~~~~~~~~~~~~~~~~

The ``datavector`` subcommand blinds a cosmic-shear SACC file with the default
CCL backend, then encrypts the original:

.. code-block:: bash

   smokescreen datavector --config configuration_file.yaml

The configuration keys are the subcommand's arguments:

.. code-block:: yaml

    path_to_sacc: "./cosmicshear_sacc.fits"
    fiducial_params:
        sigma8: 0.8
        Omega_c: 0.25
        Omega_b: 0.05
        h: 0.67
        n_s: 0.96
    shifts_dict:
        sigma8: 0.05
        Omega_c: [-0.03, 0.05]
    seed: 2112
    shift_type: "add"                # or "mult"
    shift_distribution: "flat"       # or "gaussian"
    path_to_output: "./output"       # default: the input file's directory
    keep_original_sacc: true
    # output_suffix: "concealed_data_vector"

A template with every argument and its default:

.. code-block:: bash

   smokescreen datavector --print_config > template_config.yaml

Any key can also be given directly on the command line, e.g. ``--seed 2112``.

.. warning::
   The original SACC file is **encrypted and then deleted** by default. Set
   ``keep_original_sacc: true`` (or ``--keep_original_sacc true``) to keep it.
   The decryption key is written next to the output as ``{root}.key``.

.. note::
   Only the default CCL backend is reachable from the CLI, so the file must
   satisfy that backend's contract above. A custom ``theory_fn`` requires the
   Python API.

Encrypting and Decrypting SACC files
-------------------------------------

Smokescreen can encrypt and decrypt SACC files, which is useful when you want
to hand over a data vector without handing over the data. Encryption uses the
`cryptography <https://cryptography.io/en/latest/>`_ library with a symmetric
key, so whoever decrypts the file needs that key.

When you run the data-vector concealment subcommand, encryption of the original
file happens automatically. The key is saved beside the encrypted file, named
after the original with a ``.key`` extension.

Encrypting files
~~~~~~~~~~~~~~~~

.. code-block:: bash

   smokescreen encrypt --path_to_sacc path/to/sacc.fits \
       --path_to_save path/to/save/ [--keep_original true]

``--path_to_save`` is an output directory and must already exist. It defaults
to the directory holding the input file.

This writes an ``.encrpt`` file and a ``.key`` file. From Python:

.. code-block:: python

   from smokescreen.encryption import encrypt_file
   encrypt_file('path/to/sacc.fits', 'path/to/save/',
                save_file=True, keep_original=False)

.. warning::
   As with the CLI, the original file is deleted unless ``keep_original`` is
   set.

Decrypting files
~~~~~~~~~~~~~~~~

.. code-block:: bash

   smokescreen decrypt --path_to_sacc path/to/sacc.encrpt \
       --path_to_key path/to/sacc.key

or from Python:

.. code-block:: python

   from smokescreen.encryption import decrypt_file
   decrypt_file('path/to/sacc.encrpt', 'path/to/sacc.key', save_file=True)

``save_file`` defaults to ``False`` in the Python API; the decrypted bytes are
returned either way.

Posterior Concealment (blinding)
---------------------------------

.. warning::

    **UNDER DEVELOPMENT**
