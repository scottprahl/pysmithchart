"""
Constants and Default Parameters for Smith Chart Plotting.

This module provides parameter type definitions and a comprehensive set of
constants used for configuring Smith Chart plots with matplotlib. These settings
include defaults for plot styling, gridlines, axes, and symbol representations,
as well as numerical tolerances.

Parameter Types
---------------
- ``S_PARAMETER``: Indicates scattering parameters (reflection coefficient Γ).
- ``Z_PARAMETER``: Indicates impedance parameters (always normalized by Z₀).
- ``A_PARAMETER``: Indicates absolute/unnormalized values (plotted as-is).
- ``Y_PARAMETER``: Indicates admittance parameters.

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
- ``axes.impedance`` (int): Reference impedance for normalization (default: 50).
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
- ``grid.major.xmaxn`` (int): Maximum intervals on the real axis.
- ``grid.major.ymaxn`` (int): Maximum intervals on the imaginary axis.
- ``grid.major.fancy`` (bool): Use fancy grid drawing.
- ``grid.major.fancy.threshold`` (tuple): Threshold for fancy grid styling.

Minor Grid:

- ``grid.minor.enable`` (bool): Enable the minor grid.
- ``grid.minor.capstyle`` (str): Cap style for dash segments.
- ``grid.minor.dashes`` (list): Dash style pattern.
- ``grid.minor.linewidth`` (float): Line width.
- ``grid.minor.color`` (str): Color for grid lines (also set for color.x and color.y).
- ``grid.minor.xauto`` (int): Automatic interval count for the real axis.
- ``grid.minor.yauto`` (int): Automatic interval count for the imaginary axis.
- ``grid.minor.fancy`` (bool): Use fancy minor grid styling.
- ``grid.minor.fancy.dividers`` (list): Dividers for the fancy grid.
- ``grid.minor.fancy.threshold`` (int): Threshold for switching to the next divider.

Plot Settings:

- ``plot.zorder`` (int): Z-order for plot lines.
- ``plot.marker.default`` (str): Default marker for line points.
- ``plot.marker.start`` (str): Marker for the first point (requires marker hack).
- ``plot.marker.end`` (str): Marker for the last point (requires marker hack).
- ``plot.marker.hack`` (bool): Enable the marker hack that uses code injection.
- ``plot.marker.rotate`` (bool): Rotate the end marker in the direction of the line.
- ``plot.default.datatype``: Default datatype for plots (S, Z, or Y parameter).
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
S_PARAMETER = "S"
Z_PARAMETER = "Z"
A_PARAMETER = "A"
Y_PARAMETER = "Y"

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
    "axes.impedance": 50,
    "axes.radius": 0.43,
    "axes.normalize": True,
    "axes.normalize.label": True,
    "axes.normalize.label.position.x": 0.98,  #  left
    "axes.normalize.label.position.y": 0.98,  #  bottom
    "axes.ylabel.correction": (-1.5, 0, 0),
    # Grid settings
    "grid.zorder": 1,
    "grid.locator.precision": 2,
    # Major grid settings
    "grid.major.enable": True,
    "grid.major.linestyle": "-",
    "grid.major.linewidth": 1,
    "grid.major.color": "0.2",
    "grid.major.color.x": "0.2",
    "grid.major.color.y": "0.2",
    "grid.major.xmaxn": 10,
    "grid.major.ymaxn": 16,
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
    "grid.minor.xauto": 4,
    "grid.minor.yauto": 4,
    "grid.minor.fancy": True,
    "grid.minor.fancy.dividers": [1, 2, 3, 5, 10, 20],
    "grid.minor.fancy.threshold": 35,
    # Plot settings
    "plot.zorder": 4,
    "plot.marker.default": "o",
    "plot.marker.start": "s",
    "plot.marker.end": "^",
    "plot.marker.hack": False,  # Note: Uses code injection and may produce unexpected behavior.
    "plot.marker.rotate": True,
    "plot.default.datatype": Z_PARAMETER,
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
