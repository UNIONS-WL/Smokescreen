<p align="center">
  <img src="docs/source/_static/smokescreen.png" width="409.6" height="338"
 alt="Smokescreen"/>
</p>
<div align="center">
  
[![CI](https://github.com/UNIONS-WL/Smokescreen/actions/workflows/CI.yml/badge.svg)](https://github.com/UNIONS-WL/Smokescreen/actions/workflows/CI.yml)
[![Documentation](https://github.com/UNIONS-WL/Smokescreen/actions/workflows/build_docs.yml/badge.svg?branch=main)](https://github.com/UNIONS-WL/Smokescreen/actions/workflows/build_docs.yml)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/UNIONS-WL/Smokescreen/blob/main/LICENSE)
[![status](https://joss.theoj.org/papers/e878b1039a491368b19e93eb7a614e93/status.svg)](https://joss.theoj.org/papers/e878b1039a491368b19e93eb7a614e93)
[![LSST DESC Blinding Slack](https://img.shields.io/badge/join-Slack-4A154B)](https://lsstc.slack.com/archives/CT14ZF2AH)



</div>

# Smokescreen: DESC Modules for data concealment (blinding)
> :warning: Important notice :warning: : the term "blinding" is used in the context of data concealment for scientific analysis. We understand this is an outdated term and we are working to update it to a more appropriate term. If you have any suggestions, please let us know.
This repository (under development) contains the modules for data concealment (blinding) at the following levels of the analysis:
- Data-vector measurements
- Posterior distribution [not yet developed]
- (TBC) Catalogues

**This fork's documentation is the Sphinx source in [`docs/source/`](docs/source)** — start with [installation](docs/source/installation.rst) and [usage](docs/source/usage.rst). It is not published to a website; build it locally with `pip install -e ".[docs]"`, then `cd docs && python -m sphinx -b html source build`. Upstream's documentation lives at [lsstdesc.org/Smokescreen](https://lsstdesc.org/Smokescreen/); its install instructions and its firecrown-based API do not apply to this fork.

> **Note:** this is the [UNIONS-WL](https://github.com/UNIONS-WL) fork of
> [LSSTDESC/Smokescreen](https://github.com/LSSTDESC/Smokescreen). It ships a
> `theory_fn(cosmo_params) -> np.ndarray` theory-backend protocol; the built-in
> CCL backend is the default and supported theory path. The firecrown
> integration path is inherited from upstream and unsupported in this fork:
> not installed, not tested, not maintained.

## Installation
The single supported install path is pip, pinned at a release tag:
```bash
pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>
```

This resolves the full runtime closure — including `pyccl` for the default CCL
backend — with no conda environment and no PyPI package. To provision the test
tooling as well, install the `[test]` extra:
```bash
pip install "smokescreen[test] @ git+https://github.com/UNIONS-WL/Smokescreen@<tag>"
```

For developer installation and other instructions see [`docs/source/installation.rst`](docs/source/installation.rst).

For questions about the upstream package contact @arthurmloureiro, @jessmuir, or @jablazek

---

**Legacy 2pt Cosmosis Blinding**

Legacy Blinding scripts for 2pt data vector blinding with Cosmosis moved to a [new repository](https://github.com/LSSTDESC/legacy_blinding).



