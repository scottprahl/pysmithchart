"""Core SmithAxes class with initialization and configuration."""

import copy

import numpy as np
import matplotlib as mp
from matplotlib.axes import Axes
from matplotlib.transforms import Affine2D

from pysmithchart import utils
from pysmithchart.constants import SC_DEFAULT_PARAMS, RC_DEFAULT_PARAMS
from pysmithchart.constants import SC_EPSILON, SC_INFINITY, SC_NEAR_INFINITY, SC_TWICE_INFINITY
from pysmithchart.formatters import RealFormatter, ImagFormatter
from pysmithchart.locators import MajorXLocator, MajorYLocator, MinorLocator


class AxesCore:
    """Core functionality for SmithAxes including initialization and configuration."""

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
                Example: {'grid.major.color': 'blue', 'axes.Z0': 75}
            reset (bool, optional): If True, resets scParams to default values before updating.
            **kwargs: Additional key-value pairs (must use dot notation).

        Raises:
            KeyError: If an invalid parameter key is provided.

        Note:
            Parameters must use dot notation (e.g., 'grid.major.color', not 'grid_major_color').
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
                if key == "grid.major.color":
                    self.scParams["grid.major.color.x"] = value
                    self.scParams["grid.major.color.y"] = value
                elif key == "grid.minor.color":
                    self.scParams["grid.minor.color.x"] = value
                    self.scParams["grid.minor.color.y"] = value
            else:
                raise KeyError(f"'{key}' is not a valid scParams key. Use dot notation (e.g., 'grid.major.color')")

    def __init__(self, *args, **kwargs):
        """
        Initializes a new instance of the `SmithAxes` class.

        This constructor builds a Smith chart as a custom Matplotlib axes projection.

        Args:
            *args: Positional arguments passed to matplotlib.axes.Axes
            **kwargs: Keyword arguments for Smith chart configuration

        Essential Shortcuts:
            Z0 (float): Reference impedance (default: 50Ω)
            domain (str): Default data domain (IMPEDANCE_DOMAIN, REFLECTANCE_DOMAIN, etc.)

        smith_params (dict, RECOMMENDED):
            Dictionary of Smith chart parameters using dot notation.
            This is the cleanest way to configure the chart.

            Common parameters:
                'grid.major.enable': True/False
                'grid.minor.enable': True/False
                'grid.major.color': 'blue', 'red', etc.
                'grid.major.linestyle': '-', '--', ':', etc.
                'grid.major.fancy': True/False
                'grid.major.fancy.threshold': (50, 50)
                'axes.normalize': True/False
                'axes.normalize.label': True/False

            For all available parameters, see SC_DEFAULT_PARAMS in constants.py

        Examples:
            >>> # Minimal - just change Z0
            >>> fig.add_subplot(111, projection='smith', Z0=75)
            >>>
            >>> # Recommended approach with smith_params
            >>> config = {
            ...     'grid.major.color': 'blue',
            ...     'grid.major.fancy.threshold': (50, 50),
            ...     'grid.minor.enable': True
            ... }
            >>> fig.add_subplot(111, projection='smith', smith_params=config)
            >>>
            >>> # Combine shortcuts with smith_params
            >>> fig.add_subplot(111, projection='smith',
            ...                 Z0=75,
            ...                 smith_params={'grid.major.color': 'blue'})
            >>>
            >>> # Direct dot notation also works (backwards compatible)
            >>> fig.add_subplot(111, projection='smith',
            ...                 **{'axes.Z0': 75, 'grid.major.color': 'blue'})

        Notes:
            The `smith_params` approach is recommended because it:
            - Keeps configuration separate from code
            - Makes complex setups more readable
            - Can be saved/loaded from config files
            - Avoids underscore proliferation in the API
        """
        self.transProjection = None
        self.transAffine = None
        self.transDataToAxes = None
        self.transAxes = None
        self.transMoebius = None
        self.transData = None
        self._xaxis_pretransform = None
        self._xaxis_transform = None
        self._xaxis_text1_transform = None
        self._yaxis_stretch = None
        self._yaxis_correction = None
        self._yaxis_transform = None
        self._yaxis_text1_transform = None
        self._majorarcs = None
        self._minorarcs = None
        self._Z0 = 50
        self._current_zorder = None
        self.scParams = copy.deepcopy(SC_DEFAULT_PARAMS)

        # Define shortcut mappings: user-friendly name -> internal scParams key
        # Only essential parameters that users commonly need
        SHORTCUT_MAP = {
            'Z0': 'axes.Z0',
            'domain': 'plot.default.domain',
        }

        # Process smith_params dictionary first (cleanest API)
        sc_params_to_set = {}
        if 'smith_params' in kwargs:
            smith_params = kwargs.pop('smith_params')
            if not isinstance(smith_params, dict):
                raise TypeError("smith_params must be a dictionary")
            sc_params_to_set.update(smith_params)

        # Process shortcuts second
        for shortcut, internal_key in SHORTCUT_MAP.items():
            if shortcut in kwargs:
                sc_params_to_set[internal_key] = kwargs.pop(shortcut)

        # Separate matplotlib axes parameters from Smith chart parameters
        axes_kwargs = {}
        for key, value in list(kwargs.items()):
            # Check if this is a Smith chart parameter (uses dot notation)
            if '.' in key:
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

        Axes.__init__(self, *args, **axes_kwargs)

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
        self.xaxis.set_major_locator(MajorXLocator(self, self._get_key("grid.major.xdivisions")))
        self.yaxis.set_major_locator(MajorYLocator(self, self._get_key("grid.major.ydivisions")))
        self.xaxis.set_minor_locator(MinorLocator(self._get_key("grid.minor.xdivisions")))
        self.yaxis.set_minor_locator(MinorLocator(self._get_key("grid.minor.ydivisions")))

        # Configure ticks
        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")

        # Configure x-axis labels (resistance)
        bbox = self._get_key("axes.xlabel.fancybox")
        rotation = self._get_key("axes.xlabel.rotation")
        for label in self.get_xticklabels():
            label.update({
                'verticalalignment': 'center',
                'horizontalalignment': 'center',
                'rotation_mode': 'anchor',
                'rotation': rotation,
                'bbox': bbox
            })
            self.add_artist(label)

        # Configure y-axis labels (reactance)
        inf_correction = self._get_key("symbol.infinity.correction")
        for tick, loc in zip(self.yaxis.get_major_ticks(), self.yaxis.get_majorticklocs()):
            label = tick.label1

            # Adjust size for infinity symbols
            if abs(loc) > SC_NEAR_INFINITY:
                label.set_size(label.get_size() + inf_correction)

            # Set alignment based on position
            label.set_verticalalignment("center")
            x_pos = np.real(self.moebius_z(loc * 1j))
            if x_pos < -0.1:
                label.set_horizontalalignment("right")
            elif x_pos > 0.1:
                label.set_horizontalalignment("left")
            else:
                label.set_horizontalalignment("center")

        # Set formatters
        self.yaxis.set_major_formatter(ImagFormatter(self))
        self.xaxis.set_major_formatter(RealFormatter(self))

        # Add normalization label if enabled
        if self._get_key("axes.normalize") and self._get_key("axes.normalize.label"):
            x = self._get_key("axes.normalize.label.position.x")
            y = self._get_key("axes.normalize.label.position.y")
            s = "Z₀ = %d Ω" % self._get_key("axes.Z0")
            self.text(x, y, s, fontsize=14, transform=self.transAxes)

        # Enable grids according to settings
        for grid_type in ["major", "minor"]:
            enable = self._get_key(f"grid.{grid_type}.enable")
            self.grid(visible=enable, which=grid_type)


    def clear(self):
        """
        Clear the Smith chart axes.

        Resets the chart to a clean state. Called automatically during __init__
        and when user explicitly calls plt.cla().
        """
        # Reset Smith chart-specific state
        self._majorarcs = []
        self._minorarcs = []
        self._normbox = None

        # Temporarily disable grid to prevent issues during parent clear
        original_grid = self.grid
        self.grid = lambda *args, **kwargs: None
        try:
            Axes.clear(self)
        finally:
            self.grid = original_grid

        # Perform Smith chart initialization
        self._init_smith_chart()

