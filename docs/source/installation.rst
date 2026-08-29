Installation
============

Regular Installation
--------------------

Install with pip, pinned at a release tag:

.. code-block:: bash

   pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>

This resolves the full runtime closure — including ``pyccl``, used by the
built-in default CCL theory backend — into any Python (>=3.12) environment.
No conda environment is required, and the fork is not published to PyPI.

To install the test tooling as well:

.. code-block:: bash

   pip install "smokescreen[test] @ git+https://github.com/UNIONS-WL/Smokescreen@<tag>"

.. note::
   The firecrown integration path (``smokescreen.firecrown_datavector``) is
   unsupported in this fork: firecrown is not a declared dependency, the
   module is not tested, and it is not maintained. The supported theory paths
   are the built-in CCL backend and a ``theory_fn`` of your own.

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
