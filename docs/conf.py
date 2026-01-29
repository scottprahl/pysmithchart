"""
Sphinx configuration for pysmithchart documentation.

Uses:
- sphinx.ext.napoleon for Google-style docstrings
- nbsphinx for rendering Jupyter notebooks (pre-executed; no execution on RTD)
"""

from importlib.metadata import version as pkg_version

project = "pysmithchart"
release = pkg_version(project)
version = release

root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "nbsphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
]

napoleon_use_param = False
napoleon_use_rtype = False
numpydoc_show_class_members = False

exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
]

suppress_warnings = [
    "ref.mpltype",
    "docutils",  # Suppress docutils warnings from matplotlib
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = False

html_theme = "sphinx_rtd_theme"
html_scaled_image_link = False
html_sourcelink_suffix = ""
