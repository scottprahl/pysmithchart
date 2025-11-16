Changelog
=========

0.4.0
-----
* added jupyterlite support
* improved readme
* simplified MANIFEST.in
* use venv to build and test
* use requirements-dev.txt
* Makefile has many more targets

0.3.0
-----
* changed name to pysmithchart
* fix bitrot so it works with current python and matplotlib
* address deprecation warnings
* add ability to set colors for real and imaginary grids
* add docstrings to most functions
* refactor into multiple files
* standardize formatting using black
* add test cases from various people who have forked pysmithplot
* packaged for release on pypi
* added documentation on readthedocs

0.2.0
------
* last release of pySmithPlot by @vMeijin
* Support for Python 3
* improved grid generation algorithm
* plot() now also handles also single numbers and purely real data
* plot() can now interpolate lines between points or generate an equidistant spacing
* changed handling of input data and renormalization; now the actual datatype (S,Z,Y-Parameter) can be specified when calling plot()
* changed behaviour for normalization and placement of the label
* added some parameter checks
* removed default matplotlib settings
* renamed some parameters to improve consistency
* fixed issues with Unicode symbols
* fixed issues with grid generation
* fixed issues with axis label display and placement