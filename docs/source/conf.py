# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# flake8: noqa

project = 'Smokescreen'
copyright = 'LSST Dark Energy Science Collaboration '
author = 'LSST DESC (Maintainer: Arthur Loureiro <arthur.loureiro@fysik.su.se>)'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    'sphinx.ext.napoleon',
    'sphinxcontrib.autoprogram',
    'sphinxcontrib.datatemplates'
]

templates_path = ['_templates']
exclude_patterns = []

# The docs build with -n -W: an unresolvable cross-reference to something
# Smokescreen itself documents is a build failure. What nitpick mode cannot
# resolve is the *external* half of numpydoc type strings — third-party classes
# with no intersphinx inventory configured here, and the bare qualifiers
# napoleon splits out of "str, optional". Ignoring those keeps the -W gate
# pointed at our own references instead of at other projects' object indices.
nitpick_ignore = [
    ('py:class', 'np.ndarray'),
    ('py:class', 'optional'),
    ('py:class', 'module'),
    ('py:class', 'sacc.sacc.Sacc'),
    ('py:class', 'Cosmology'),
    ('py:class', 'pyccl.Cosmology'),
    ('py:class', 'pyccl.WeakLensingTracer'),
    ('py:func', 'pyccl.correlation'),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the theme
html_theme = 'sphinx_rtd_theme'
html_logo = '_static/bkp_logo.png'
# Optionally, you can customize the theme further with theme-specific options
# These are options specifically for the Wagtail Theme.
# more info here: https://sphinx-wagtail-theme.readthedocs.io/en/latest/index.html
html_theme_options = {
    'titles_only': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

html_css_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',
]
html_static_path = ['_static']
html_js_files = [
    'custom.js',
]
