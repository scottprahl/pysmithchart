"""
Smith chart visualization using matplotlib custom projections.

The `pysmithchart` package provides tools for creating Smith charts, enabling
visualization and analysis of reflection coefficients, impedances, and admittances
in RF and microwave engineering applications.

After importing this pysmithchart, the 'smith' projection becomes available in matplotlib.

Public API:
    Domain constants for specifying parameter types in plot(), scatter(), and text()

    - R_DOMAIN: S-parameters (reflection coefficients)
    - Z_DOMAIN: Impedance in ohms (automatically normalized by Z₀)
    - NORM_Z_DOMAIN: Pre-normalized impedance values
    - Y_DOMAIN: Admittance in Siemens (automatically normalized by Y₀)
    - NORM_Y_DOMAIN: Pre-normalized admittance values

More documentation at <https://pysmithchart.readthedocs.io>

Example:
    Plot impedance values in ohms is the default::

        import matplotlib.pyplot as plt
        import pysmithchart

        ax = plt.subplot(111, projection = "smith")
        ax.plot([50 + 25j, 75 - 10j], 's-')
        plt.show()

    Plot reflection coefficients on a Smith chart::

        import matplotlib.pyplot as plt
        import pysmithchart
        from pysmithchart import R_DOMAIN

        ax = plt.subplot(111, projection = "smith", domain=R_DOMAIN)
        ax.plot([0.5 + 0.3j, -0.2 - 0.1j], 'o-')
        plt.show()
"""

from matplotlib.projections import register_projection

# Import constants FIRST
from .constants import NORM_Z_DOMAIN, R_DOMAIN, Z_DOMAIN, Y_DOMAIN, NORM_Y_DOMAIN

# Now import axes AFTER constants
from .axes import SmithAxes

# Register the Smith projection
register_projection(SmithAxes)

# Public API - only export domain constants
# SmithAxes is accessed via projection="smith", not direct import
__all__ = ["R_DOMAIN", "Z_DOMAIN", "NORM_Z_DOMAIN", "Y_DOMAIN", "NORM_Y_DOMAIN"]

__version__ = "0.9.0"
__author__ = "Scott Prahl, Paul Staerke"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2025-2026 Scott Prahl"
__license__ = "BSD-3-Clause"
__url__ = "https://github.com/scottprahl/pysmithchart.git"
