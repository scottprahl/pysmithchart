"""Core SmithAxes class with initialization and configuration."""

import copy

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pysmithchart.constants import SC_DEFAULT_PARAMS, RC_DEFAULT_PARAMS
from pysmithchart.constants import SC_NEAR_INFINITY, SC_TWICE_INFINITY, SC_EPSILON
from pysmithchart.constants import NORM_Z_DOMAIN
from pysmithchart.locators import MajorXLocator, MajorYLocator, MinorLocator

# Only export the mixin class, not imported symbols
__all__ = ['AxesCore']


class AxesCore:
    """Core functionality for SmithAxes including initialization and configuration.

    This class is designed to be used as a mixin with matplotlib.axes.Axes
    via multiple inheritance. Many methods that appear to be missing (like
    set_aspect, tick_params, text, etc.) are actually provided by Axes at runtime.
    """

    # pylint: disable=no-member  # Methods come from Axes mixin

    name = "smith"

    @classmethod
    def get_rc_params(cls):
        """Gets the default values for matplotlib parameters."""
        return RC_DEFAULT_PARAMS.copy()

    def update_scParams(self, sc_dict=None, reset=False, **kwargs):
        """
        Update scParams for the current instance.

        Args:
            sc_dict (dict, optional): Dictionary of parameters to update using dot notation.
                Example: {'grid.Z.major.color': 'blue', 'axes.Z0': 75}
            reset (bool, optional): If True, resets scParams to default values before updating.
            **kwargs: Additional key-value pairs (must use dot notation).

        Raises:
            KeyError: If an invalid parameter key is provided.

        Note:
            Parameters must use dot notation (e.g., 'grid.Z.major.color', not 'grid.Z.major_color').
            Use shortcuts like Z0, datatype in __init__ instead.
        """
        if reset:
            self.scParams = copy.deepcopy(SC_DEFAULT_PARAMS)

        # Merge sc_dict into kwargs for unified processing
        if sc_dict is not None:
            kwargs.update(sc_dict)

        # Process all parameters
        for key, value in kwargs.items():
            if key in self.scParams:
                self.scParams[key] = value
                # Handle color auto-propagation
                if key == "grid.Z.major.color":
                    self.scParams["grid.Z.major.color.x"] = value
                    self.scParams["grid.Z.major.color.y"] = value
                elif key == "grid.Z.minor.color":
                    self.scParams["grid.Z.minor.color.x"] = value
                    self.scParams["grid.Z.minor.color.y"] = value
            else:
                raise KeyError(f"'{key}' is not a valid scParams key. Use dot notation (e.g., 'grid.Z.major.color')")

    def __init__(self, *args, **kwargs):
        """
        Initializes a new instance of the `SmithAxes` class.

        This constructor builds a Smith chart as a custom Matplotlib axes projection.

        Args:
            *args: Positional arguments passed to matplotlib.axes.Axes
            **kwargs: Keyword arguments for Smith chart configuration

        Essential Shortcuts:
            Z0 (float): Reference impedance (default: 50Ω)
            domain (str): Default data domain (Z_DOMAIN, R_DOMAIN, etc.)
            grid (str): Grid type selection. Options:
                - 'impedance' (default): Impedance grid only
                - 'admittance': Admittance grid only
                - 'both': Both impedance and admittance grids

        smith_style (dict):
            Dictionary of Smith chart parameters to modify.

            Common parameters:
                'grid.Z.major.enable': True/False
                'grid.Z.minor.enable': True/False
                'grid.Z.major.color': 'blue', 'red', etc.
                'grid.Z.major.linestyle': '-', '--', ':', etc.
                'grid.Z.major.fancy': True/False
                'grid.Z.major.fancy.threshold': (50, 50)
                'axes.normalize': True/False
                'axes.normalize.label': True/False

            For all available parameters, see SC_DEFAULT_PARAMS in constants.py

        Examples:
            >>> # Minimal - just change Z0
            >>> fig.add_subplot(111, projection='smith', Z0=75)
            >>>
            >>> # Quick admittance chart
            >>> fig.add_subplot(111, projection='smith', grid='admittance')
            >>>
            >>> # Show both impedance and admittance grids
            >>> fig.add_subplot(111, projection='smith', grid='both')
            >>>
            >>> # Recommended approach with smith_style
            >>> ss = {
            ...     'grid.Z.major.color': 'blue',
            ...     'grid.Z.major.fancy.threshold': (50, 50),
            ...     'grid.Z.minor.enable': True
            ... }
            >>> fig.add_subplot(111, projection='smith', smith_style=ss)
            >>>
            >>> # Combine shortcuts with smith_style
            >>> fig.add_subplot(111, projection='smith',
            ...                 Z0=75,
            ...                 grid='admittance',
            ...                 smith_style={'grid.Y.major.color': 'blue'})
        """
        self._current_zorder = None
        self._normbox = None
        self._xaxis_pretransform = None
        self._xaxis_transform = None
        self._xaxis_text1_transform = None
        self._yaxis_stretch = None
        self._yaxis_correction = None
        self._yaxis_transform = None
        self._yaxis_text1_transform = None
        self._Y_major_arcs = None
        self._Y_minor_arcs = None
        self._Z_major_arcs = None
        self._Z_minor_arcs = None
        self._Z0 = 50
        self.scParams = copy.deepcopy(SC_DEFAULT_PARAMS)
        self.transProjection = None
        self.transAffine = None
        self.transDataToAxes = None
        self.transAxes = None
        self.transMoebius = None
        self.transData = None
        self.xaxis = None
        self.yaxis = None

        # Define shortcut mappings for user-friendly names
        SHORTCUT_MAP = {
            "Z0": "axes.Z0",
            "domain": "plot.default.domain",
        }

        # default is a smith chart with impedance grid
        sc_params_to_set = {
            "grid.Z.major.enable": True,
            "grid.Z.minor.enable": True,
            "grid.Y.major.enable": False,
            "grid.Y.minor.enable": False,
        }

        grid = kwargs.pop("grid", None)
        if grid is not None and grid != "impedance":
            grid = grid.lower()
            if grid == "admittance":
                sc_params_to_set.update(
                    {
                        "grid.Z.major.enable": False,
                        "grid.Z.minor.enable": False,
                        "grid.Y.major.enable": True,
                        "grid.Y.minor.enable": True,
                    }
                )
            elif grid == "both":
                sc_params_to_set.update(
                    {
                        "grid.Y.major.enable": True,
                        "grid.Y.minor.enable": True,
                    }
                )
            else:
                raise ValueError(f"Invalid 'grid' parameter: '{grid}'. Must be 'impedance', 'admittance', or 'both'.")

        if "smith_style" in kwargs:
            smith_style = kwargs.pop("smith_style")
            if not isinstance(smith_style, dict):
                raise TypeError("smith_style must be a dictionary")
            sc_params_to_set.update(smith_style)

        # Process shortcuts second
        for shortcut, internal_key in SHORTCUT_MAP.items():
            if shortcut in kwargs:
                sc_params_to_set[internal_key] = kwargs.pop(shortcut)

        # Separate matplotlib axes parameters from Smith chart parameters
        axes_kwargs = {}
        for key, value in list(kwargs.items()):
            # Check if this is a Smith chart parameter (uses dot notation)
            if "." in key:
                # Direct dot notation - use as-is
                sc_params_to_set[key] = kwargs.pop(key)
            elif key in RC_DEFAULT_PARAMS:
                # It's a matplotlib rcParam - leave in kwargs for Axes
                continue
            else:
                # Not a known Smith chart param or rcParam - pass to Axes
                axes_kwargs[key] = kwargs.pop(key)

        # Apply Smith chart parameters
        if sc_params_to_set:
            self.update_scParams(sc_dict=sc_params_to_set)

        if self._get_key("init.updaterc"):
            for key, value in RC_DEFAULT_PARAMS.items():
                if mp.rcParams[key] == mp.rcParamsDefault[key]:
                    mp.rcParams[key] = value

        Axes.__init__(self, *args, **axes_kwargs)  # pylint: disable=non-parent-init-called

        self.set_aspect(1, adjustable="box", anchor="C")
        self.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)

    def _get_key(self, key):
        """Get value for key from the local dictionary or the global matplotlib rcParams."""
        if key in self.scParams:
            return self.scParams[key]
        if key in mp.rcParams:
            return mp.rcParams[key]
        raise KeyError("%s is not a valid key" % key)

    def _init_axis(self):
        self.xaxis = mp.axis.XAxis(self)
        self.yaxis = mp.axis.YAxis(self)
        self._update_transScale()

    def _init_smith_chart(self):
        """
        Initialize Smith chart-specific configuration.

        Called once during __init__ to set up locators, formatters, limits,
        and other Smith chart-specific properties.
        """
        # Reset state
        self._normbox = None
        self._Z0 = self._get_key("axes.Z0")
        self._current_zorder = self._get_key("plot.zorder")

        # Set limits first (before locators which may trigger updates)
        Axes.set_xlim(self, 0, SC_TWICE_INFINITY)
        Axes.set_ylim(self, -SC_TWICE_INFINITY, SC_TWICE_INFINITY)

        # Configure axis locators
        # Use admittance divisions if only admittance is enabled, otherwise use impedance
        impedance_enabled = self._get_key("grid.Z.major.enable")
        admittance_enabled = self._get_key("grid.Y.major.enable")

        if admittance_enabled and not impedance_enabled:
            # Pure admittance chart - use Y divisions
            real_divs = self._get_key("grid.Y.major.real.divisions")
            imag_divs = self._get_key("grid.Y.major.imag.divisions")
            real_minor_divs = self._get_key("grid.Y.minor.real.divisions")
            imag_minor_divs = self._get_key("grid.Y.minor.imag.divisions")
        else:
            # Impedance chart or both - use Z divisions
            real_divs = self._get_key("grid.Z.major.real.divisions")
            imag_divs = self._get_key("grid.Z.major.imag.divisions")
            real_minor_divs = self._get_key("grid.Z.minor.real.divisions")
            imag_minor_divs = self._get_key("grid.Z.minor.imag.divisions")

        self.xaxis.set_major_locator(MajorXLocator(self, real_divs))
        self.yaxis.set_major_locator(MajorYLocator(self, imag_divs))
        self.xaxis.set_minor_locator(MinorLocator(real_minor_divs))
        self.yaxis.set_minor_locator(MinorLocator(imag_minor_divs))

        # Configure ticks
        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")

        # Configure x-axis labels (resistance/conductance)
        # Turn off matplotlib's automatic tick labels
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        # Manually add labels with full control over positioning
        self._add_manual_axis_labels()

    def _add_manual_axis_labels(self):
        """Manually add axis labels for both impedance and admittance modes."""
        bbox = self._get_key("axes.xlabel.fancybox")
        rotation = self._get_key("axes.xlabel.rotation")

        # Determine grid mode
        impedance_enabled = self._get_key("grid.Z.major.enable")
        admittance_enabled = self._get_key("grid.Y.major.enable")

        # Get tick locations
        x_major_locs = self.xaxis.get_majorticklocs()
        y_major_locs = self.yaxis.get_majorticklocs()

        # Add X-axis (real axis) labels, x_pos is in the NORM_Z_DOMAIN
        for loc in x_major_locs:
            if admittance_enabled and not impedance_enabled:
                if loc < SC_EPSILON:
                    label_text = "∞"
                    x_pos = 0  # right edge on admittance chart
                elif loc >= SC_NEAR_INFINITY:
                    label_text = "0"
                    x_pos = SC_NEAR_INFINITY  # left edge on admittance chart
                else:
                    label_text = ("%f" % loc).rstrip("0").rstrip(".")
                    x_pos = 1 / loc  # Reciprocal position
            else:
                if loc < SC_EPSILON:
                    label_text = "0"
                    x_pos = 0  # Left edge
                elif loc >= SC_NEAR_INFINITY:
                    label_text = "∞"
                    x_pos = SC_NEAR_INFINITY  # Right edge
                else:
                    label_text = ("%f" % loc).rstrip("0").rstrip(".")
                    x_pos = loc

            # Place label at bottom of chart where circle crosses real axis
            self.text(
                x_pos,
                0,
                label_text,
                verticalalignment="center",
                horizontalalignment="center",
                rotation_mode="anchor",
                rotation=rotation,
                bbox=bbox,
                domain=NORM_Z_DOMAIN,
                clip_on=False,
            )

        # Add Y-axis (imaginary axis) labels, y_pos is in the NORM_Z_DOMAIN
        for loc in y_major_locs:
            if abs(loc) < SC_EPSILON or abs(loc) >= SC_NEAR_INFINITY:
                continue

            # Regular value - format and position
            label_text = ("%f" % abs(loc)).rstrip("0").rstrip(".") + "j"
            if loc < 0:
                label_text = "-" + label_text

            # Position in NORM_Z_DOMAIN (normalized impedance space)
            # For admittance, label at reciprocal position
            if admittance_enabled and not impedance_enabled:
                y_pos = 1 / loc
            else:
                y_pos = loc

            # Calculate horizontal position for alignment
            x_moebius = np.real(self.moebius_z(y_pos * 1j))

            if x_moebius < -0.1:
                ha = "right"
            elif x_moebius > 0.1:
                ha = "left"
            else:
                ha = "center"

            # Vertical alignment based on top/bottom half
            # Positive y_pos = top half, negative y_pos = bottom half
            if y_pos > 0:
                va = "bottom"  # Label below the point (top half of chart)
            elif y_pos < 0:
                va = "top"  # Label above the point (bottom half of chart)
            else:
                va = "center"  # Center label (at y=0)

            # Place label using NORM_Z_DOMAIN (no special size for infinity)
            self.text(
                0,
                y_pos,
                label_text,
                verticalalignment=va,
                horizontalalignment=ha,
                domain=NORM_Z_DOMAIN,
                clip_on=False,
            )

        # Add normalization label if enabled
        if self._get_key("axes.normalize") and self._get_key("axes.normalize.label"):
            x = self._get_key("axes.normalize.label.position.x")
            y = self._get_key("axes.normalize.label.position.y")
            s = "Z₀ = %d Ω" % self._get_key("axes.Z0")
            self.text(x, y, s, fontsize=14, transform=self.transAxes)

        # Enable grids according to settings - grid() checks the enable flags internally
        self.grid(grid="both")

    def clear(self):
        """
        Clear the Smith chart axes.

        Resets the chart to a clean state. Called automatically during __init__
        and when user explicitly calls plt.cla().
        """
        # Reset Smith chart-specific state
        self._Z_major_arcs = []
        self._Z_minor_arcs = []
        self._Y_major_arcs = []
        self._Y_minor_arcs = []
        self._normbox = None

        # Temporarily disable grid to prevent issues during parent clear
        original_grid = getattr(self, "grid", None)
        if original_grid is not None:
            self.grid = lambda *args, **kwargs: None
        try:
            Axes.clear(self)
        finally:
            if original_grid is not None:
                self.grid = original_grid

        # Perform Smith chart initialization
        self._init_smith_chart()
