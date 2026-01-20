"""
Constants and Default Parameters for Smith Chart Plotting.

This module provides parameter type definitions and a comprehensive set of
constants used for configuring Smith Chart plots with matplotlib. These settings
include defaults for plot styling, gridlines, axes, and symbol representations,
as well as numerical tolerances.

Parameter Types
---------------
- ``REFLECTANCE_DOMAIN``: Indicates scattering parameters (reflection coefficient Γ).
- ``IMPEDANCE_DOMAIN``: Indicates impedance parameters (always normalized by Z₀).
- ``ABSOLUTE_DOMAIN``: Indicates absolute/unnormalized values (plotted as-is).
- ``ADMITTANCE_DOMAIN``: Indicates admittance parameters.

Numerical Constants
-------------------
- ``SC_EPSILON``: Tolerance for numerical comparisons (1e-7).
- ``SC_INFINITY``: Value representing "infinity" (1e9).
- ``SC_NEAR_INFINITY``: 90% of ``SC_INFINITY``.
- ``SC_TWICE_INFINITY``: Twice ``SC_INFINITY``.

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

Major Grid:

- ``grid.major.enable`` (bool): Enable the major grid.
- ``grid.major.linestyle`` (str): Line style.
- ``grid.major.linewidth`` (int): Line width.
- ``grid.major.color`` (str): Color of grid lines (also set for color.x and color.y).
- ``grid.major.xdivisions`` (int): Maximum divisions on the real axis.
- ``grid.major.ydivisions`` (int): Maximum divisions on the imaginary axis.
- ``grid.major.fancy`` (bool): Use fancy grid drawing.
- ``grid.major.fancy.threshold`` (tuple): Threshold for fancy grid styling.

Minor Grid:

- ``grid.minor.enable`` (bool): Enable the minor grid.
- ``grid.minor.capstyle`` (str): Cap style for dash segments.
- ``grid.minor.dashes`` (list): Dash style pattern.
- ``grid.minor.linewidth`` (float): Line width.
- ``grid.minor.color`` (str): Color for grid lines (also set for color.x and color.y).
- ``grid.minor.xdivisions`` (int or None): Number of divisions between major ticks on the real axis.
  If None, divisions are computed automatically per interval for uniform spacing.
- ``grid.minor.ydivisions`` (int or None): Number of divisions between major ticks on the imaginary axis.
  If None, divisions are computed automatically per interval for uniform spacing.
- ``grid.minor.fancy`` (bool): Use fancy minor grid styling.
- ``grid.minor.fancy.dividers`` (list): Dividers for the fancy grid.
- ``grid.minor.fancy.threshold`` (int): Threshold for switching to the next divider.

Plot Settings:

- ``plot.zorder`` (int): Z-order for plot lines.
- ``plot.marker.default`` (str): Default marker for line points.
- ``plot.default.domain``: Default domain for plots (REFLECTION, IMPEDANCE, ADMITTANCE, or ABSOLUTE).
- ``plot.default.interpolation`` (int): Number of interpolated steps between points.

Symbol Settings:

- ``symbol.infinity`` (str): Symbol for infinity. The trailing space prevents label cutoff.
- ``symbol.infinity.correction`` (int): Size correction for the infinity symbol.
- ``symbol.ohm`` (str): Symbol for the resistance unit (ohm).

Additional Parameter:

- ``init.updaterc`` (bool): Flag indicating whether to update matplotlib's rc parameters.
"""

# =============================================================================
# Numerical Constants
# =============================================================================
REFLECTANCE_DOMAIN = "S"
IMPEDANCE_DOMAIN = "Z"
ABSOLUTE_DOMAIN = "A"
ADMITTANCE_DOMAIN = "Y"

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
    # Axes settings
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
    # Outer boundary (Smith-chart frame)
    #
    # The outer boundary is rendered as the axes patch/spine (a circle). It is
    # configurable via the Smith-chart parameter system so that documentation
    # examples can rely on a single reusable parameter dictionary.
    #
    # Defaults are selected to preserve the historical look where the boundary
    # matched the major grid color.
    "grid.outer.enable": True,
    "grid.outer.color": "0.2",
    "grid.outer.linestyle": "-",
    "grid.outer.linewidth": 1,
    "grid.outer.alpha": 1.0,
    # Major grid settings
    "grid.major.enable": True,
    "grid.major.linestyle": "-",
    "grid.major.linewidth": 1,
    "grid.major.color": "0.2",
    "grid.major.color.x": "0.2",
    "grid.major.color.y": "0.2",
    "grid.major.alpha": 1.0,
    "grid.major.xdivisions": 10,
    "grid.major.ydivisions": 16,
    "grid.major.fancy": True,
    "grid.major.fancy.threshold": (100, 50),
    # Minor grid settings
    "grid.minor.enable": False,
    "grid.minor.linestyle": ":",
    "grid.minor.capstyle": "round",
    "grid.minor.dashes": [0.2, 2],
    "grid.minor.linewidth": 0.75,
    "grid.minor.color": "0.4",
    "grid.minor.color.x": "0.4",
    "grid.minor.color.y": "0.4",
    "grid.minor.alpha": 1.0,
    "grid.minor.xdivisions": None,
    "grid.minor.ydivisions": None,
    "grid.minor.fancy": True,
    "grid.minor.fancy.dividers": [1, 2, 3, 4, 5, 10, 20],
    "grid.minor.fancy.threshold": 35,
    # Plot settings
    "plot.zorder": 4,
    "plot.marker.default": "o",
    "plot.default.domain": IMPEDANCE_DOMAIN,
    "plot.default.interpolation": 5,
    # Initialization flag
    "init.updaterc": True,
    # Symbol settings
    "symbol.infinity": "∞ ",  # Trailing space prevents label cutoff.
    "symbol.infinity.correction": 8,
    "symbol.ohm": "Ω",
}

__all__ = [
    "SC_EPSILON",
    "SC_INFINITY",
    "SC_NEAR_INFINITY",
    "SC_TWICE_INFINITY",
    "RC_DEFAULT_PARAMS",
    "SC_DEFAULT_PARAMS",
]
