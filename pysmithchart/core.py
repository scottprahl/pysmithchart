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
from pysmithchart.locators import RealMaxNLocator, ImagMaxNLocator, SmithAutoMinorLocator


class AxesCore:
    """Core functionality for SmithAxes including initialization and configuration."""

    name = "smith"

    @classmethod
    def get_rc_params(cls):
        """Gets the default values for matplotlib parameters."""
        return RC_DEFAULT_PARAMS.copy()

    def update_scParams(self, sc_dict=None, reset=False, **kwargs):
        """
        Update scParams for the current instance based on a dictionary or keyword arguments.

        Args:
            sc_dict (dict, optional): Dictionary of parameters to update.
            reset (bool, optional): If True, resets scParams to default values before updating.
            **kwargs: Additional key-value pairs to update parameters.

        Raises:
            KeyError: If an invalid parameter key is provided (unless `filter_dict` is True).
        """
        if reset:
            self.scParams = copy.deepcopy(SC_DEFAULT_PARAMS)

        if sc_dict is not None:
            for key, value in sc_dict.items():
                if key in self.scParams:
                    self.scParams[key] = value
                    if key == "grid.major.color":
                        self.scParams["grid.major.color.x"] = value
                        self.scParams["grid.major.color.y"] = value
                    elif key == "grid.minor.color":
                        self.scParams["grid.minor.color.x"] = value
                        self.scParams["grid.minor.color.y"] = value
                else:
                    raise KeyError("key '%s' is not in scParams" % key)

        remaining = kwargs.copy()
        for key in kwargs:
            key_dot = key.replace("_", ".")
            if key_dot in self.scParams:
                value = remaining.pop(key)
                self.scParams[key_dot] = value
                if key_dot == "grid.major.color":
                    self.scParams["grid.major.color.x"] = value
                    self.scParams["grid.major.color.y"] = value
                elif key_dot == "grid.minor.color":
                    self.scParams["grid.minor.color.x"] = value
                    self.scParams["grid.minor.color.y"] = value
                else:
                    self.scParams[key_dot] = value
            else:
                raise KeyError("key '%s' is not in scParams" % key_dot)

    def __init__(self, *args, **kwargs):
        """
        Initializes a new instance of the `SmithAxes` class.
        This constructor builds a Smith chart as a custom Matplotlib axes projection.
        It initializes instance-specific parameters, separates axes-related configurations
        from Smith chart-specific configurations, and applies default settings where applicable.
        Args:
            *args:
                Positional arguments passed to the base `matplotlib.axes.Axes` class.
            **kwargs:
                Keyword arguments for configuring the Smith chart or the underlying Matplotlib axes.
                These include:
                    - datatype: Default datatype for plotting (e.g., S_PARAMETER, IMPEDANCE).
                      If provided, sets "plot.default.datatype" for this instance.
                    - Smith chart parameters: Parameters specific to the Smith chart, such as
                      normalization, impedance, or appearance settings. See `update_scParams`
                      for a list of supported parameters.
                    - Axes parameters: Parameters unrelated to Smith chart functionality,
                      passed directly to the `matplotlib.axes.Axes` class.
        ...
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
        self._impedance = None
        self._normalize = None
        self._current_zorder = None
        self.scParams = copy.deepcopy(SC_DEFAULT_PARAMS)
        
        # Extract datatype before processing other kwargs
        datatype = kwargs.pop('datatype', None)
        
        axes_kwargs = {}
        for key in kwargs.copy():
            key_dot = key.replace("_", ".")
            if not (key_dot in self.scParams or key_dot in RC_DEFAULT_PARAMS):
                axes_kwargs[key] = kwargs.pop(key)  # Changed from key_dot to key
        
        self.update_scParams(**kwargs)
        
        # Set default datatype if provided
        if datatype is not None:
            self.scParams["plot.default.datatype"] = datatype
        
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

    def clear(self):
        """
        Clear the current Smith Chart axes and reset them to their initial state.

        This method overrides the base `clear` (clear axes) method from `matplotlib.axes.Axes`
        to perform additional setup and customization specific to the Smith Chart. It clears
        custom properties like arcs, gridlines, and axis formatting, while also restoring
        default configurations.

        Key Functionality:
            - Resets internal storage for major and minor arcs (`_majorarcs` and `_minorarcs`).
            - Temporarily disables the grid functionality during the base class's `clear` call
              to prevent unintended behavior.
            - Reinitializes important Smith Chart-specific properties, such as normalization
              settings, impedance, and z-order tracking.
            - Configures axis locators and formatters for real and imaginary components.
            - Updates axis tick alignment, label styles, and normalization box positioning.
            - Redraws the gridlines (major and minor) based on the current settings.

        Notes:
            - This method ensures that the Smith Chart maintains its specific configuration
              after clearing, unlike a standard `matplotlib` axes.
            - Labels and gridlines are re-added to maintain proper layering and alignment.

        Side Effects:
            - Resets `_majorarcs` and `_minorarcs` to empty lists.
            - Updates the axis locators, formatters, and gridlines.
            - Configures custom label alignment and appearance.
        """
        self._majorarcs = []
        self._minorarcs = []
        original_grid = self.grid
        self.grid = lambda *args, **kwargs: None
        try:
            Axes.clear(self)
        finally:
            self.grid = original_grid
        self._normbox = None
        self._impedance = self._get_key("axes.impedance")
        self._normalize = self._get_key("axes.normalize")
        self._current_zorder = self._get_key("plot.zorder")
        self.xaxis.set_major_locator(RealMaxNLocator(self, self._get_key("grid.major.xmaxn")))
        self.yaxis.set_major_locator(ImagMaxNLocator(self, self._get_key("grid.major.ymaxn")))

        self.xaxis.set_minor_locator(SmithAutoMinorLocator(self._get_key("grid.minor.xauto")))
        self.yaxis.set_minor_locator(SmithAutoMinorLocator(self._get_key("grid.minor.yauto")))

        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")
        Axes.set_xlim(self, 0, SC_TWICE_INFINITY)
        Axes.set_ylim(self, -SC_TWICE_INFINITY, SC_TWICE_INFINITY)
        for label in self.get_xticklabels():  # pylint: disable=not-callable
            label.set_verticalalignment("center")
            label.set_horizontalalignment("center")
            label.set_rotation_mode("anchor")
            label.set_rotation(self._get_key("axes.xlabel.rotation"))
            label.set_bbox(self._get_key("axes.xlabel.fancybox"))
            self.add_artist(label)
        for tick, loc in zip(self.yaxis.get_major_ticks(), self.yaxis.get_majorticklocs()):
            if abs(loc) > SC_NEAR_INFINITY:
                tick.label1.set_size(tick.label1.get_size() + self._get_key("symbol.infinity.correction"))
            tick.label1.set_verticalalignment("center")
            x = np.real(self.moebius_z(loc * 1j))
            if x < -0.1:
                tick.label1.set_horizontalalignment("right")
            elif x > 0.1:
                tick.label1.set_horizontalalignment("left")
            else:
                tick.label1.set_horizontalalignment("center")
        self.yaxis.set_major_formatter(ImagFormatter(self))
        self.xaxis.set_major_formatter(RealFormatter(self))

        if self._get_key("axes.normalize") and self._get_key("axes.normalize.label"):
            x = self._get_key("axes.normalize.label.position.x")
            y = self._get_key("axes.normalize.label.position.y")
            s = "Z₀ = %d Ω" % self._get_key("axes.impedance")
            self.text(x, y, s, fontsize=14, transform=self.transAxes)

        for grid in ["major", "minor"]:
            enable_tag = "grid.%s.enable" % grid
            enable_key = self._get_key(enable_tag)
            self.grid(visible=enable_key, which=grid)
