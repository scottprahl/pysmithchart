"""
Constants and Default Parameters for Smith Chart Plotting.

This module provides parameter type definitions and a comprehensive set of
constants used for configuring Smith Chart plots with matplotlib. These settings
include defaults for plot styling, gridlines, axes, and symbol representations,
as well as numerical tolerances.

Parameter Types
---------------
- ``R_DOMAIN``: Indicates scattering parameters (reflection coefficient Γ).
- ``Z_DOMAIN``: Indicates impedance parameters (always normalized by Z₀).
- ``NORM_Z_DOMAIN``: Indicates absolute/unnormalized values (plotted as-is).
- ``Y_DOMAIN``: Indicates admittance parameters.

Numerical Constants
-------------------
- ``SC_EPSILON``: Tolerance for numerical comparisons (1e-7).
- ``SC_INFINITY``: Value representing "infinity" (1e9).
- ``SC_NEAR_INFINITY``: 90% of ``SC_INFINITY``.
- ``SC_TWICE_INFINITY``: Twice ``SC_INFINITY``.
x
Matplotlib Default Parameters (``RC_DEFAULT_PARAMS``)
------------------------------------------------------
The default parameters used to configure matplotlib are defined in the dictionary below:

.. code-block:: python

    {
        "axes.axisbelow": True,
        "font.size": 12,
        "legend.fontsize": 12,
        "legend.fancybox": False,
        "legend.markerscale": 1,
        "legend.numpoints": 3,
        "legend.shadow": False,
        "lines.linestyle": "-",
        "lines.linewidth": 2,
        "lines.markeredgewidth": 1,
        "lines.markersize": 5,
        "xtick.labelsize": 10,
        "xtick.major.pad": 0,
        "ytick.labelsize": 10,
        "ytick.major.pad": 4,
    }

Smith Chart Default Parameters (``SC_DEFAULT_PARAMS``)
--------------------------------------------------------

Axes Settings:

- ``axes.xlabel.rotation`` (int): Rotation angle for x-axis labels (default: 90).
- ``axes.xlabel.fancybox`` (dict): Parameters for the label background box.
- ``axes.Z0`` (int): Reference impedance for normalization (default: 50).
- ``axes.radius`` (float): Radius of the plotting area (default: 0.43).
- ``axes.normalize`` (bool): If True, normalize the chart to the reference impedance.
- ``axes.normalize.label`` (bool): If True, display a normalization label.
- ``axes.normalize.label.position`` (complex): Position of the normalization label.
- ``axes.ylabel.correction`` (tuple): Correction for y-axis label positioning.

Grid Settings:

- ``grid.zorder`` (int): Z-order for grid lines (default: 1).
- ``grid.locator.precision`` (int): Number of significant decimals per decade (default: 2).
- ``grid.fancy`` (bool): Use fancy grid drawing for smith charts.
- ``grid.major.threshold`` (tuple): Visual threshold for spacing
- ``grid.minor.threshold`` (int): Visual threshold for minorspacing

Major Grid (Impedance):

- ``grid.Z.major.enable`` (bool): Enable the major grid.
- ``grid.Z.major.linestyle`` (str): Line style.
- ``grid.Z.major.linewidth`` (int): Line width.
- ``grid.Z.major.color`` (str): Color of grid lines (also set for color.x and color.y).
- ``grid.Z.major.real.divisions`` (int): Maximum divisions on the real axis.
- ``grid.Z.major.imag.divisions`` (int): Maximum divisions on the imaginary axis.

Minor Grid (Impedance):

- ``grid.Z.minor.enable`` (bool): Enable the minor grid.
- ``grid.Z.minor.capstyle`` (str): Cap style for dash segments.
- ``grid.Z.minor.dashes`` (list): Dash style pattern.
- ``grid.Z.minor.linewidth`` (float): Line width.
- ``grid.Z.minor.color`` (str): Color for grid lines (also set for color.x and color.y).
- ``grid.Z.minor.real.divisions`` (int or None): Number of divisions between major ticks on the real axis.
  If None, divisions are computed automatically per interval for uniform spacing.
- ``grid.Z.minor.imag.divisions`` (int or None): Number of divisions between major ticks on the imaginary axis.
  If None, divisions are computed automatically per interval for uniform spacing.

Admittance Grid Settings:

Major Grid (Admittance):

- ``grid.Y.major.enable`` (bool): Enable the major admittance grid.
- ``grid.Y.major.linestyle`` (str): Line style for admittance grid.
- ``grid.Y.major.linewidth`` (int): Line width for admittance grid.
- ``grid.Y.major.color`` (str): Color of admittance grid lines.
- ``grid.Y.major.color.x`` (str): Color for conductance circles.
- ``grid.Y.major.color.y`` (str): Color for susceptance circles.
- ``grid.Y.major.alpha`` (float): Alpha transparency for admittance grid.
- ``grid.Y.major.real.divisions`` (int): Maximum divisions on conductance axis.
- ``grid.Y.major.imag.divisions`` (int): Maximum divisions on susceptance axis.

Minor Grid (Admittance):

- ``grid.Y.minor.enable`` (bool): Enable the minor admittance grid.
- ``grid.Y.minor.linestyle`` (str): Line style for minor admittance grid.
- ``grid.Y.minor.capstyle`` (str): Cap style for dash segments.
- ``grid.Y.minor.dashes`` (list): Dash style pattern.
- ``grid.Y.minor.linewidth`` (float): Line width for minor admittance grid.
- ``grid.Y.minor.color`` (str): Color for minor admittance grid lines.
- ``grid.Y.minor.color.x`` (str): Color for minor conductance circles.
- ``grid.Y.minor.color.y`` (str): Color for minor susceptance circles.
- ``grid.Y.minor.alpha`` (float): Alpha transparency for minor admittance grid.
- ``grid.Y.minor.real.divisions`` (int or None): Number of divisions for conductance.
- ``grid.Y.minor.imag.divisions`` (int or None): Number of divisions for susceptance.

Plot Settings:

- ``plot.zorder`` (int): Z-order for plot lines.
- ``plot.marker.default`` (str): Default marker for line points.
- ``plot.default.domain``: Default domain for plots (REFLECTION, IMPEDANCE, ADMITTANCE, or ABSOLUTE).
- ``plot.default.interpolation`` (int): Number of interpolated steps between points.

Additional Parameter:

- ``init.updaterc`` (bool): Flag indicating whether to update matplotlib's rc parameters.
"""

# =============================================================================
# Numerical Constants
# =============================================================================
R_DOMAIN = "S"
Z_DOMAIN = "Z"
NORM_Z_DOMAIN = "z"
Y_DOMAIN = "Y"
NORM_Y_DOMAIN = "y"

SC_EPSILON = 1e-7
SC_INFINITY = 1e9
SC_NEAR_INFINITY = 0.9 * SC_INFINITY
SC_TWICE_INFINITY = 2.0 * SC_INFINITY


# =============================================================================
# Default Matplotlib Parameters (rcParams)
# =============================================================================
RC_DEFAULT_PARAMS = {
    "axes.axisbelow": True,
    "font.size": 12,
    "legend.fontsize": 12,
    "legend.fancybox": False,
    "legend.markerscale": 1,
    "legend.numpoints": 3,
    "legend.shadow": False,
    "lines.linestyle": "-",
    "lines.linewidth": 2,
    "lines.markeredgewidth": 1,
    "lines.markersize": 5,
    "xtick.labelsize": 10,
    "xtick.major.pad": 0,
    "ytick.labelsize": 10,
    "ytick.major.pad": 4,
}


# =============================================================================
# Default Smith Chart Parameters
# =============================================================================
SC_DEFAULT_PARAMS = {
    "axes.xlabel.rotation": 90,
    "axes.xlabel.fancybox": {
        "boxstyle": "round,pad=0.2,rounding_size=0.2",
        "facecolor": "w",
        "edgecolor": "w",
        "mutation_aspect": 0.75,
        "alpha": 1,
    },
    "axes.Z0": 50,
    "axes.radius": 0.43,
    "axes.normalize": True,
    "axes.normalize.label": True,
    "axes.normalize.label.position.x": 0.02,  #  left
    "axes.normalize.label.position.y": 0.02,  #  bottom
    "axes.ylabel.correction": (-1.5, 0, 0),
    # Grid settings
    "grid.zorder": 1,
    "grid.locator.precision": 2,
    "grid.fancy": True,
    "grid.major.threshold": (100, 50),
    "grid.minor.threshold": 10,
    # Outer boundary (Smith-chart frame)
    "grid.outer.enable": True,
    "grid.outer.color": "0.2",
    "grid.outer.linestyle": "-",
    "grid.outer.linewidth": 1,
    "grid.outer.alpha": 1.0,
    # Major grid settings (impedance)
    "grid.Z.major.enable": True,
    "grid.Z.major.linestyle": "-",
    "grid.Z.major.linewidth": 1,
    "grid.Z.major.color": "0.2",
    "grid.Z.major.color.x": "0.2",
    "grid.Z.major.color.y": "0.2",
    "grid.Z.major.alpha": 1.0,
    "grid.Z.major.real.divisions": 10,
    "grid.Z.major.imag.divisions": 16,
    # Minor grid settings (impedance)
    "grid.Z.minor.enable": False,
    "grid.Z.minor.linestyle": ":",
    "grid.Z.minor.capstyle": "round",
    "grid.Z.minor.dashes": [0.2, 2],
    "grid.Z.minor.linewidth": 0.75,
    "grid.Z.minor.color": "0.4",
    "grid.Z.minor.color.x": "0.4",
    "grid.Z.minor.color.y": "0.4",
    "grid.Z.minor.alpha": 1.0,
    "grid.Z.minor.real.divisions": None,
    "grid.Z.minor.imag.divisions": None,
    # Major admittance grid settings
    "grid.Y.major.enable": False,
    "grid.Y.major.linestyle": "-",
    "grid.Y.major.linewidth": 1,
    "grid.Y.major.color": "0.6",
    "grid.Y.major.color.x": "0.6",
    "grid.Y.major.color.y": "0.6",
    "grid.Y.major.alpha": 1.0,
    "grid.Y.major.real.divisions": 10,
    "grid.Y.major.imag.divisions": 16,
    # Minor admittance grid settings
    "grid.Y.minor.enable": False,
    "grid.Y.minor.linestyle": ":",
    "grid.Y.minor.capstyle": "round",
    "grid.Y.minor.dashes": [0.2, 2],
    "grid.Y.minor.linewidth": 0.75,
    "grid.Y.minor.color": "0.7",
    "grid.Y.minor.color.x": "0.7",
    "grid.Y.minor.color.y": "0.7",
    "grid.Y.minor.alpha": 1.0,
    "grid.Y.minor.real.divisions": None,
    "grid.Y.minor.imag.divisions": None,
    # Plot settings
    "plot.zorder": 4,
    "plot.marker.default": "o",
    "plot.default.domain": Z_DOMAIN,
    "plot.default.interpolation": 5,
    # Initialization flag
    "init.updaterc": True,
}

__all__ = [
    "SC_EPSILON",
    "SC_INFINITY",
    "SC_NEAR_INFINITY",
    "SC_TWICE_INFINITY",
    "RC_DEFAULT_PARAMS",
    "SC_DEFAULT_PARAMS",
]
