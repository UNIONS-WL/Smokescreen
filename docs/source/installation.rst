Installation
============

Regular Installation
--------------------

The single supported install path for this fork is pip, pinned at a release
tag:

.. code-block:: bash

   pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>

This resolves the full declared runtime closure — including ``pyccl``, which
the built-in default CCL theory backend uses — into any Python (>=3.12)
environment. No conda environment is required, and the fork is not published
to PyPI.

.. note::
   The firecrown integration path is inherited from upstream
   `LSSTDESC/Smokescreen <https://github.com/LSSTDESC/Smokescreen>`_ and is
   unsupported in this fork: not installed, not tested, not maintained. The
   default and supported theory path is the built-in CCL backend.

Developer Installation
-----------------------

Clone the repository and install it (editable, with the test tooling) into a
fresh virtual environment:

.. code-block:: bash

   git clone https://github.com/UNIONS-WL/Smokescreen.git
   cd Smokescreen
   python -m venv .venv && source .venv/bin/activate
   python -m pip install -e '.[test]'

Testing the installation
------------------------

You can test the developer installation by running the unit tests from the
Smokescreen directory:

.. code-block:: bash

   pytest .
