"""Import-discipline guarantees, verified in fresh subprocesses.

`import smokescreen` and `import smokescreen.datavector` must not pull in pyccl
or firecrown; pyccl enters sys.modules only once the default CCL backend is
constructed.
"""
import subprocess
import sys


def _run(code):
    res = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert res.returncode == 0, f"subprocess failed:\n{res.stdout}\n{res.stderr}"


def test_import_smokescreen_no_pyccl_no_firecrown():
    _run(
        "import sys, smokescreen\n"
        "assert 'pyccl' not in sys.modules, 'pyccl imported by smokescreen'\n"
        "assert 'firecrown' not in sys.modules, 'firecrown imported by smokescreen'\n"
    )


def test_import_datavector_no_pyccl_no_firecrown():
    _run(
        "import sys, smokescreen.datavector\n"
        "assert 'pyccl' not in sys.modules, 'pyccl imported by datavector'\n"
        "assert 'firecrown' not in sys.modules, 'firecrown imported by datavector'\n"
    )


def test_building_ccl_backend_imports_pyccl():
    _run(
        "import sys\n"
        "from smokescreen.backends.ccl import build_ccl_theory_fn\n"
        "assert 'pyccl' in sys.modules, 'pyccl not imported by ccl backend'\n"
    )
