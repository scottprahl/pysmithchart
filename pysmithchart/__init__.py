"""
This module registers the matplotlib projection "smith".

The `pysmithchart` package provides tools for working with Smith charts, enabling visualization and
analysis of reflection coefficients, impedances, and admittances in RF and microwave engineering.

Modules:
    axes: Defines the `SmithAxes` class, which implements the custom Smith chart projection.
    core: Core initialization and configuration for SmithAxes.
    transforms: Coordinate transformation operations (Möbius, etc.).
    grid: Grid drawing functionality.
    plotting: Plotting, text, and annotation methods.
    helpers: Utility and helper methods.

Constants:
    R_DOMAIN: Scattering parameter domain for plotting reflection coefficients.
    Z_DOMAIN: Impedance parameter domain for plotting normalized impedances.
    Y_DOMAIN: Admittance parameter domain for plotting normalized admittances.
    NORM_Z_DOMAIN: Absolute parameter domain for plotting unnormalized values.

Public API:
    - SmithAxes: The custom projection class for Smith charts.
    - R_DOMAIN: Constant for S-parameter plotting.
    - Z_DOMAIN: Constant for Z-parameter plotting.
    - Y_DOMAIN: Constant for Y-parameter plotting.
    - NORM_Z_DOMAIN: Constant for A-parameter plotting.

Example:
    Import the module and plot a reflection coefficient using the Smith chart projection:

    >>> import matplotlib.pyplot as plt
    >>> from pysmithchart import R_DOMAIN
    >>> plt.subplot(1, 1, 1, projection="smith")
    >>> plt.plot([0.5 + 0.3j, -0.2 - 0.1j], 'o', domain=R_DOMAIN)
    >>> plt.show()
"""

from matplotlib.projections import register_projection

# Import constants FIRST
from .constants import NORM_Z_DOMAIN, R_DOMAIN, Z_DOMAIN, Y_DOMAIN, NORM_Y_DOMAIN

# Now import axes AFTER constants
from .axes import SmithAxes

# Register the Smith projection
register_projection(SmithAxes)

# Public API for wildcard imports
__all__ = ["SmithAxes", "R_DOMAIN", "Z_DOMAIN", "Y_DOMAIN", "NORM_Z_DOMAIN"]

__version__ = "0.6.0"
__author__ = "Paul Staerke, Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2025-2026 Scott Prahl"
__license__ = "BSD-3-Clause"
__url__ = "https://github.com/scottprahl/pysmithchart.git"

if "site-packages" in __file__:
    raise RuntimeError("pysmithchart is not running from a development checkout")
