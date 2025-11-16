"""This module contains the implementation for the SmithAxes class."""

import copy
from collections.abc import Iterable
from numbers import Number
from types import MethodType

import numpy as np
import matplotlib as mp
from matplotlib.axes import Axes
from matplotlib.cbook import simple_linear_interpolation as linear_interpolation
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D, BboxTransformTo
from scipy.interpolate import splprep, splev

from pysmithchart import Z_PARAMETER, Y_PARAMETER, S_PARAMETER
from pysmithchart import utils
from pysmithchart.constants import SC_DEFAULT_PARAMS, RC_DEFAULT_PARAMS
from pysmithchart.constants import SC_EPSILON, SC_INFINITY, SC_NEAR_INFINITY, SC_TWICE_INFINITY
from pysmithchart.formatters import RealFormatter, ImagFormatter
from pysmithchart.locators import RealMaxNLocator, ImagMaxNLocator, SmithAutoMinorLocator
from pysmithchart.moebius_transform import MoebiusTransform
from pysmithchart.polar_transform import PolarTranslate

__all__ = ["SmithAxes"]


class SmithAxes(Axes):
    """
    A subclass of :class:`matplotlib.axes.Axes` specialized for rendering Smith Charts.

    This class implements a fully automatic Smith Chart with support for impedance
    normalization, custom grid configurations, and flexible marker handling. Default
    parameters (e.g., grid settings, marker styles, and plot defaults) are defined in
    :mod:`pysmithchart.constants`.

    Note:
        Parameter changes (such as grid updates) may not take effect immediately.
        To reset the chart, use the :meth:`clear` method.
    """

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
                    - Smith chart parameters: Parameters specific to the Smith chart, such as
                      normalization, impedance, or appearance settings. See `update_scParams`
                      for a list of supported parameters.
                    - Axes parameters: Parameters unrelated to Smith chart functionality,
                      passed directly to the `matplotlib.axes.Axes` class.

        Attributes:
            scParams (dict):
                A deep copy of the default Smith chart parameters (`SmithAxes.scDefaultParams`)
                for this instance. Modifications to these parameters are unique to the instance.
            _majorarcs (None or list):
                Holds major arcs on the Smith chart, initialized as `None` and set later during rendering.
            _minorarcs (None or list):
                Holds minor arcs on the Smith chart, initialized as `None` and set later during rendering.
            _impedance (None or float):
                Impedance value used for normalizing Smith chart calculations, if applicable.
            _normalize (None or bool):
                Indicates whether normalization is applied to the Smith chart.
            _current_zorder (None or float):
                Tracks the current Z-order of plotted elements for layering purposes.

        Notes:
            - Parameters in `kwargs` not recognized as Smith chart parameters or Matplotlib default parameters
              are treated as axes-specific configurations and passed to the base `matplotlib.axes.Axes` class.
            - This method calls `update_scParams` to apply Smith chart-specific settings.
            - If the `"init.updaterc"` parameter is enabled, this method also updates Matplotlib's `rcParams`
              with custom Smith chart defaults.

        See Also:
            - `update_scParams`: Updates Smith chart-specific parameters for the current instance.
            - `matplotlib.axes.Axes`: The base class for this custom projection.
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
        axes_kwargs = {}
        for key in kwargs.copy():
            key_dot = key.replace("_", ".")
            if not (key_dot in self.scParams or key_dot in RC_DEFAULT_PARAMS):
                axes_kwargs[key] = kwargs.pop(key_dot)
        self.update_scParams(**kwargs)
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
            x, y = utils.z_to_xy(self.moebius_inv_z(self._get_key("axes.normalize.label.position")))
            impedance = self._get_key("axes.impedance")
            s = "Z$_\\mathrm{0}$ = %d$\\,$%s" % (impedance, self._get_key("symbol.ohm"))
            box = self.text(x, y, s, ha="left", va="bottom")
            px = self._get_key("ytick.major.pad")
            py = px + 0.5 * box.get_fontsize()
            box.set_transform(self._yaxis_correction + Affine2D().translate(-px, -py))

        for grid in ["major", "minor"]:
            enable_tag = "grid.%s.enable" % grid
            enable_key = self._get_key(enable_tag)
            self.grid(visible=enable_key, which=grid)

    def _set_lim_and_transforms(self):
        """
        Configure the axis limits and transformation pipelines for the chart.

        This method defines and applies a series of transformations to map data
        space, Möbius space, axes space, and drawing space.

        Transformations:
            - `transProjection`: Maps data space to Möbius space using a Möbius transformation.
            - `transAffine`: Scales and translates Möbius space to fit axes space.
            - `transDataToAxes`: Combines `transProjection` and `transAffine` to map data space to axes space.
            - `transAxes`: Maps axes space to drawing space using the bounding box (`bbox`).
            - `transMoebius`: Combines `transAffine` and `transAxes` to map Möbius space to drawing space.
            - `transData`: Combines `transProjection` and `transMoebius` as data-to-drawing-space transform.

        X-axis transformations:
            - `_xaxis_pretransform`: Scales and centers the x-axis based on axis limits.
            - `_xaxis_transform`: Combines `_xaxis_pretransform` and `transData` for full x-axis mapping.
            - `_xaxis_text1_transform`: Adjusts x-axis label positions.

        Y-axis transformations:
            - `_yaxis_stretch`: Scales the y-axis based on axis limits.
            - `_yaxis_correction`: Applies additional translation to the y-axis for label adjustments.
            - `_yaxis_transform`: Combines `_yaxis_stretch` and `transData` for full y-axis mapping.
            - `_yaxis_text1_transform`: Combines `_yaxis_stretch` and `_yaxis_correction` for y label position
        """
        r = self._get_key("axes.radius")
        self.transProjection = MoebiusTransform(self)
        self.transAffine = Affine2D().scale(r, r).translate(0.5, 0.5)
        self.transDataToAxes = self.transProjection + self.transAffine
        self.transAxes = BboxTransformTo(self.bbox)
        self.transMoebius = self.transAffine + self.transAxes
        self.transData = self.transProjection + self.transMoebius
        self._xaxis_pretransform = Affine2D().scale(1, 2 * SC_TWICE_INFINITY).translate(0, -SC_TWICE_INFINITY)
        self._xaxis_transform = self._xaxis_pretransform + self.transData
        self._xaxis_text1_transform = Affine2D().scale(1.0, 0.0) + self.transData
        self._yaxis_stretch = Affine2D().scale(SC_TWICE_INFINITY, 1.0)
        self._yaxis_correction = self.transData + Affine2D().translate(*self._get_key("axes.ylabel.correction")[:2])
        self._yaxis_transform = self._yaxis_stretch + self.transData
        self._yaxis_text1_transform = self._yaxis_stretch + self._yaxis_correction

    def get_xaxis_transform(self, which="grid"):
        """Return the x-axis transformation for ticks or grid."""
        assert which in ["tick1", "tick2", "grid"]
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad_points):
        """Return the transformation for x-axis label placement."""
        return (self._xaxis_text1_transform, "center", "center")

    def get_yaxis_transform(self, which="grid"):
        """Return the y-axis transformation for ticks or grid."""
        assert which in ["tick1", "tick2", "grid"]
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad_points):
        """Return the transformation for y-axis label placement."""
        if hasattr(self, "yaxis") and len(self.yaxis.majorTicks) > 0:
            font_size = self.yaxis.majorTicks[0].label1.get_size()
        else:
            font_size = self._get_key("font.size")
        offset = self._get_key("axes.ylabel.correction")[2]
        return (
            self._yaxis_text1_transform + PolarTranslate(self, pad=pad_points + offset, font_size=font_size),
            "center",
            "center",
        )

    def _gen_axes_patch(self):
        """Generate the patch used to draw the Smith chart axes."""
        r = self._get_key("axes.radius") + 0.015
        c = self._get_key("grid.major.color.x")
        return Circle((0.5, 0.5), r, edgecolor=c)

    def _gen_axes_spines(self, locations=None, offset=0.0, units="inches"):
        """Generate the spines for the circular Smith chart axes."""
        spine = Spine.circular_spine(self, (0.5, 0.5), self._get_key("axes.radius"))
        spine.set_edgecolor(self._get_key("grid.major.color.x"))
        return {SmithAxes.name: spine}

    def set_xscale(self, *args, **kwargs):
        """
        Set the x-axis scale (only 'linear' is supported).

        Args:
            *args: Positional arguments for the scale (first argument must be 'linear').
            **kwargs: Keyword arguments for additional scale settings.
        """
        if len(args) == 0 or args[0] != "linear":
            raise NotImplementedError("Only 'linear' scale is supported for the x-axis.")
        Axes.set_xscale(self, *args, **kwargs)  # pylint: disable=not-callable

    def set_yscale(self, *args, **kwargs):
        """
        Set the y-axis scale (only 'linear' is supported).

        Args:
            *args: Positional arguments for the scale (first argument must be 'linear').
            **kwargs: Keyword arguments for additional scale settings.
        """
        if len(args) == 0 or args[0] != "linear":
            raise NotImplementedError("Only 'linear' scale is supported for the y-axis.")
        Axes.set_yscale(self, *args, **kwargs)  # pylint: disable=not-callable

    def set_xlim(self, *args, **kwargs):
        """
        Override the `set_xlim` method to enforce immutability.

        The x-axis limits for the Smith chart are fixed to `(0, infinity)` and cannot
        be modified. Any arguments passed to this method are ignored.
        """
        _ = (args, kwargs)  # Suppress "unused argument" warning
        Axes.set_xlim(self, 0, SC_TWICE_INFINITY)

    def set_ylim(self, *args, **kwargs):
        """
        Override the `set_ylim` method to enforce immutability.

        The y-axis limits for the Smith chart are fixed to `(-infinity, infinity)` and cannot
        be modified. Any arguments passed to this method are ignored.
        """
        _ = (args, kwargs)  # Suppress "unused argument" warning
        Axes.set_ylim(self, -SC_TWICE_INFINITY, SC_TWICE_INFINITY)

    def format_coord(self, x, y):
        """Format real and imaginary parts of a complex number."""
        sgn = "+" if y > 0 else "-"
        return "%.5f %s %.5fj" % (x, sgn, abs(y)) if x > 0 else ""

    def get_data_ratio(self):
        """Return the fixed aspect ratio of the Smith chart data."""
        return 1.0

    def can_zoom(self):
        """Check if zooming is enabled (always returns False)."""
        return False

    def start_pan(self, x, y, button):
        """Handle the start of a pan action (disabled for Smith chart)."""

    def end_pan(self):
        """Handle the end of a pan action (disabled for Smith chart)."""

    def drag_pan(self, button, key, x, y):
        """Handle panning during a drag action (disabled for Smith chart)."""

    def moebius_z(self, *args, normalize=None):
        """
        Apply a Möbius transformation to the input values.

        This function uses the ``utils.moebius_z`` method to compute the Möbius
        transformation: ``w = 1 - 2 * norm / (z + norm)``. The transformation can
        handle a single complex value or a combination of real and imaginary parts
        provided as separate arguments. The normalization value can be specified or
        determined automatically based on the instance's settings.

        Args:
            *args:
                Input arguments passed to ``utils.moebius_z``. These can include:

                - A single complex number or numpy.ndarray with ``dtype=complex``.
                - Two arguments representing the real and imaginary parts of a complex
                  number or array of complex numbers (floats or arrays of floats).

            normalize (bool or None, optional):
                If ``True``, normalizes the values to ``self._impedance``.
                If ``None``, uses the instance attribute ``self._normalize`` to determine
                behavior.
                If ``False``, no normalization is applied.

        Returns:
            complex or numpy.ndarray:
                The Möbius-transformed value(s), returned as a complex number or an array
                of complex numbers, depending on the input.
        """
        if normalize is not None:
            print("moebius normalize=", normalize)
        if normalize is None:
            normalize = self._normalize
        if normalize:
            norm = 1
        else:
            norm = self._get_key("axes.impedance")
        return utils.moebius_z(*args, norm=norm)

    def moebius_inv_z(self, *args, normalize=None):
        """
        Perform the inverse Möbius transformation.

        This method applies the inverse Möbius transformation formula:
        w = k * (1 - z)/(1 + z), where k is determined
        by the axes scale or normalization settings. The transformation is
        applied to complex numbers or real/imaginary pairs.

        Normalization is applied using the impedance (`self._impedance`) if enabled.
        This method uses the `utils.moebius_inv_z` utility for calculations.

        Args:
            *args:
                Input data to transform, either as:

                - `z` (complex): A complex number or `numpy.ndarray` with `dtype=complex`.
                - `x, y` (float): Real and imaginary parts, either as floats or
                  `numpy.ndarray` values with non-complex `dtype`.

            normalize (bool or None, optional):
                Specifies whether to normalize the transformation:
                - `True`: Normalize values to `self._impedance`.
                - `False`: No normalization is applied.
                - `None` (default): Use the instance's default normalization setting (`self._normalize`).

        Returns: Transformed data, either as a single complex value or a
                `numpy.ndarray` with `dtype=complex`.
        """
        normalize = self._normalize if normalize is None else normalize
        norm = 1 if normalize else self._get_key("axes.impedance")
        return utils.moebius_inv_z(*args, norm=norm)

    def real_interp1d(self, x, steps):
        """
        Interpolate a vector of real values with evenly spaced points.

        This method interpolates the given real values such that, after applying a Möbius
        transformation with an imaginary part of 0, the resulting points are evenly spaced.

        The result is mapped back to the original space using the inverse Möbius transformation.

        Args:
            x (iterable): Real values to interpolate.
            steps (int): Interpolation steps between two points.

        Returns: Interpolated real values.
        """
        return self.moebius_inv_z(linear_interpolation(self.moebius_z(np.array(x)), steps))

    def imag_interp1d(self, y, steps):
        """
        Interpolate a vector of imaginary values with evenly spaced points.

        This method interpolates the given imaginary values such that, after applying
        a Möbius transformation with a real part of 0, the resulting points are evenly spaced.

        The result is mapped back to the original space using the inverse Möbius transformation.

        Args:
            y (iterable): Imaginary values to interpolate.
            steps (int): Interpolation steps between two points.

        Returns: Interpolated imaginary values.
        """
        angs = np.angle(self.moebius_z(np.array(y) * 1j)) % (2 * np.pi)
        i_angs = linear_interpolation(angs, steps)
        return np.imag(self.moebius_inv_z(utils.ang_to_c(i_angs)))

    def legend(self, *args, **kwargs):
        """
        Create and display a legend for the Smith chart, filtering duplicate entries.

        This method customizes the legend behavior to ensure unique entries are displayed
        and applies a specialized handler for lines with custom markers. It also filters out
        duplicate legend labels, keeping only the first occurrence.

        Args:
            *args:
                Positional arguments passed directly to `matplotlib.axes.Axes.legend`.

            **kwargs:
                Keyword arguments for configuring the legend. Includes all standard arguments
                supported by `matplotlib.axes.Axes.legend`, such as:

                - loc: Location of the legend (e.g., 'upper right', 'lower left').
                - fontsize: Font size for the legend text.
                - ncol: Number of columns in the legend.
                - title: Title for the legend.

                See the Matplotlib documentation for more details.

        Returns:
            matplotlib.legend.Legend:
                The legend instance created for the Smith chart.
        """

        class SmithHandlerLine2D(HandlerLine2D):
            """
            Custom legend handler for `Line2D` objects in Smith charts.

            This class extends `matplotlib.legend_handler.HandlerLine2D` to provide
            customized rendering of legend entries for `Line2D` objects, especially
            those with marker modifications in Smith charts. It ensures that custom
            markers, such as start and end markers, are rendered correctly in the legend.
            """

            def create_artists(
                self,
                legend,
                orig_handle,
                xdescent,
                ydescent,
                width,
                height,
                fontsize,
                trans,
            ):
                """Creates the legend artist applying custom markers."""
                legline = HandlerLine2D.create_artists(
                    self,
                    legend,
                    orig_handle,
                    xdescent,
                    ydescent,
                    width,
                    height,
                    fontsize,
                    trans,
                )
                try:
                    n_points = len(orig_handle.get_xdata())
                except Exception:
                    n_points = legend.numpoints

                proxy_line = legline[0]
                # Grab the proxy data arrays.
                proxy_xdata = list(proxy_line.get_xdata())
                proxy_ydata = list(proxy_line.get_ydata())

                # Adjust the proxy so that its number of points reflects the
                # actual number of points in the original line.
                if n_points == 1 and len(proxy_xdata) > 1:
                    # If only one point was drawn, pass an array of length 1.
                    proxy_line.set_xdata(proxy_xdata[:1])
                    proxy_line.set_ydata(proxy_ydata[:1])
                elif n_points == 2 and len(proxy_xdata) > 2:
                    # If two points were drawn, pass an array of length 2.
                    proxy_line.set_xdata(proxy_xdata[:2])
                    proxy_line.set_ydata(proxy_ydata[:2])

                # Apply the hacked line drawing if needed.
                if hasattr(orig_handle, "markers_hacked"):
                    legend.axes.hack_linedraw(proxy_line, rotate_marker=True)

                return legline

        handles, labels = self.get_legend_handles_labels()
        seen_labels = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                seen_labels.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        return Axes.legend(
            self,
            unique_handles,
            unique_labels,
            handler_map={Line2D: SmithHandlerLine2D()},
            **kwargs,
        )

    def plot(self, *args, **kwargs):
        """
        Plot data on the Smith Chart.

        This method extends the functionality of :meth:`matplotlib.axes.Axes.plot` to
        support Smith Chart-specific features, including handling of complex data and
        additional keyword arguments for customization.

        Args:
            *args:
                Positional arguments for the data to plot. Supports real and complex
                data. Complex data should either be of type `complex` or a
                `numpy.ndarray` with `dtype=complex`.
            **kwargs:
                Keyword arguments for customization. Includes all arguments supported
                by :meth:`matplotlib.axes.Axes.plot`, along with the following:

                datatype (str, optional):
                    Specifies the input data format. Must be one of:
                    - `S_PARAMETER` ('S'): Scattering parameters.
                    - `Z_PARAMETER` ('Z'): Impedance.
                    - `Y_PARAMETER` ('Y'): Admittance.
                    Defaults to `Z_PARAMETER`.

                interpolate (bool or int, optional):
                    If `True`, interpolates the given data linearly with a default step size.
                    If an integer, specifies the number of interpolation steps.
                    Defaults to `False`.

                equipoints (bool or int, optional):
                    If `True`, interpolates the data to equidistant points. If an integer,
                    specifies the number of equidistant points. Cannot be used with
                    `interpolate`. Defaults to `False`.

                markerhack (bool, optional):
                    Enables manipulation of the start and end markers of the line.
                    Defaults to `False`.

                rotate_marker (bool, optional):
                    If `markerhack` is enabled, rotates the end marker in the direction
                    of the corresponding path. Defaults to `False`.

        Returns:
            list[matplotlib.lines.Line2D]:
                A list of line objects representing the plotted data.

        Raises:
            ValueError: If `datatype` is not one of `S_PARAMETER`, `Z_PARAMETER`, or `Y_PARAMETER`.
            ValueError: If both `interpolate` and `equipoints` are enabled.
            ValueError: If `interpolate` is specified with a non-positive value.

        Examples:
            Plot impedance data on a Smith Chart:

            >>> import matplotlib.pyplot as plt
            >>> import pysmithchart
            >>> ZL = [30 + 30j, 50 + 50j, 100 + 100j]
            >>> plt.subplot(1, 1, 1, projection="smith")
            >>> plt.plot(ZL, "b", marker="o", markersize=10, datatype=pysmithchart.Z_PARAMETER)
            >>> plt.show()
        """
        datatype = kwargs.pop("datatype", self._get_key("plot.default.datatype"))
        if datatype not in [S_PARAMETER, Z_PARAMETER, Y_PARAMETER]:
            raise ValueError(f"Invalid datatype: {datatype}. Must be S_PARAMETER, Z_PARAMETER, or Y_PARAMETER")

        new_args = ()
        for arg in args:
            if not isinstance(arg, (str, np.ndarray)):
                if isinstance(arg, Number):
                    arg = np.array([arg], dtype=complex)
                elif isinstance(arg, Iterable):
                    arg = np.array(arg, dtype=complex)

            if isinstance(arg, np.ndarray) and arg.dtype in [complex, np.complex128]:
                new_args += utils.z_to_xy(arg)
            else:
                new_args += (arg,)

        if "zorder" not in kwargs:
            kwargs["zorder"] = self._current_zorder
            self._current_zorder += 0.001

        interpolate = kwargs.pop("interpolate", False)
        equipoints = kwargs.pop("equipoints", False)
        kwargs.setdefault("marker", self._get_key("plot.marker.default"))
        markerhack = kwargs.pop("markerhack", self._get_key("plot.marker.hack"))
        rotate_marker = kwargs.pop("rotate_marker", self._get_key("plot.marker.rotate"))

        if interpolate:
            if equipoints > 0:
                raise ValueError("Interpolation is not available with equidistant markers")
            interpolation = self._get_key("plot.default.interpolation")
            if interpolation < 0:
                raise ValueError("Interpolation is only for positive values possible!")
            if "markevery" in kwargs:
                mark = kwargs["markevery"]
                if isinstance(mark, Iterable):
                    mark = np.asarray(mark) * (interpolate + 1)
                else:
                    mark *= interpolate + 1
                kwargs["markevery"] = mark

        lines = Axes.plot(self, *new_args, **kwargs)

        for line in lines:
            cdata = utils.xy_to_z(line.get_data())

            if datatype == S_PARAMETER:
                z = self.moebius_inv_z(cdata)
            elif datatype == Y_PARAMETER:
                z = 1 / cdata
            else:
                z = cdata

            if self._normalize and datatype == Z_PARAMETER:
                z /= self._get_key("axes.impedance")

            line.set_data(utils.z_to_xy(z))

            if interpolate or equipoints:
                z = self.moebius_z(*line.get_data())
                if len(z) > 1:
                    spline, t0 = splprep(utils.z_to_xy(z), s=0)  # pylint: disable=unbalanced-tuple-unpacking
                    ilen = (interpolate + 1) * (len(t0) - 1) + 1
                    if equipoints == 1:
                        t = np.linspace(0, 1, ilen)
                    elif equipoints > 1:
                        t = np.linspace(0, 1, equipoints)
                    else:
                        t = np.zeros(ilen)
                        t[0], t[1:] = (
                            t0[0],
                            np.concatenate(
                                [np.linspace(i0, i1, interpolate + 2)[1:] for i0, i1 in zip(t0[:-1], t0[1:])]
                            ),
                        )
                    z = self.moebius_inv_z(*splev(t, spline))
                    line.set_data(utils.z_to_xy(z))
            if markerhack:
                self.hack_linedraw(line, rotate_marker)
        return lines

    def grid(
        self,
        visible=None,
        which="major",
        axis=None,
        dividers=None,
        threshold=None,
        **kwargs,
    ):
        """
        Draw gridlines on the Smith chart, with optional customization for style and behavior.

        This method overrides the default grid functionality in Matplotlib to use arcs
        instead of straight lines. The grid consists of major and minor components, which
        can be drawn in either a standard or "fancy" style. Fancy grids dynamically adjust
        spacing and length based on specified parameters.

        The "fancy" grid mode is only valid when `axis='both'`.

        Keyword arguments like `linestyle`, `linewidth`, `color`, and `alpha`
        can be used to customize the grid appearance.

        The `zorder` of the gridlines defaults to the Smith chart's settings
        unless explicitly overridden.

        Args:
            visible (bool, optional):
                Enables or disables the selected grid. Defaults to the current state.
            which (str, optional):
                Specifies which gridlines to draw:
                - `'major'`: Major gridlines only.
                - `'minor'`: Minor gridlines only.
                - `'both'`: Both major and minor gridlines.
                Defaults to `'major'`.
            axis (bool, optional):
                If `True`, draws the grid in a "fancy" style with dynamic spacing
                and length adjustments. Defaults to `None`, which uses the standard style.
            dividers (list[int], optional):
                Adaptive divisions for the minor fancy grid. Only applicable when `axis=True`.
                Has no effect on major or standard grids.
            threshold (float or tuple[float, float], optional):
                Specifies the threshold for dynamically adapting grid spacing and
                line length. Can be a single float for both axes or a tuple for
                individual axis thresholds.
            **kwargs:
                Additional keyword arguments passed to the gridline creator. Note that
                gridlines are created as `matplotlib.patches.Patch` objects, so not all
                properties from `matplotlib.lines.Line2D` are supported.

        See Also:
            - `matplotlib.axes.Axes.grid`: The base grid function being overridden.
            - `matplotlib.patches.Patch`: The class used to create the gridlines.
        """
        assert which in ["both", "major", "minor"]
        assert axis in [None, False, True]

        def get_kwargs(grid):
            kw = kwargs.copy()
            kw.setdefault("zorder", self._get_key("grid.zorder"))
            kw.setdefault("alpha", self._get_key("grid.alpha"))
            for key in ["linestyle", "linewidth", "color"]:
                if grid == "minor" and key == "linestyle":
                    if "linestyle" not in kw:
                        kw.setdefault("dash_capstyle", self._get_key("grid.minor.capstyle"))
                        kw.setdefault("dashes", self._get_key("grid.minor.dashes"))
                else:
                    kw.setdefault(key, self._get_key("grid.%s.%s" % (grid, key)))
            return kw

        def check_fancy(yticks):
            """
            Checks if the imaginary axis ticks are symmetric about zero.

            This property is required for "fancy" minor grid styling.

            Args:
                yticks: Array or list of tick values for the imaginary axis.

            Returns:
                The upper half of the `yticks` array (non-negative values).
            """
            len_y = (len(yticks) - 1) // 2
            if not (len(yticks) % 2 == 1 and (yticks[len_y:] + yticks[len_y::-1] < SC_EPSILON).all()):
                s = "Fancy minor grid is only supported for zero-symmetric imaginary grid. "
                s += "--- e.g., ImagMaxNLocator"
                raise ValueError(s)
            return yticks[len_y:]

        def split_threshold(threshold):
            if isinstance(threshold, tuple):
                thr_x, thr_y = threshold
            else:
                thr_x = thr_y = threshold
            assert thr_x > 0 and thr_y > 0
            return (thr_x / 1000, thr_y / 1000)

        def add_arc(ps, p0, p1, grid, arc_type):
            """
            Add an arc to the Smith Chart.

            Parameters:
                ps (tuple): Starting point of the arc in parameterized space.
                p0 (tuple): One endpoint of the arc.
                p1 (tuple): The other endpoint of the arc.
                grid (str): Specifies whether the arc is part of the "major" or "minor" grid.
                            Must be one of ["major", "minor"].
                arc_type (str): Specifies the type of the arc, either "real" or "imag" for
                            real or imaginary components.

            Side Effects:
                Appends the created arc to the appropriate list:
                - `_majorarcs` if `grid` is "major".
                - `_minorarcs` if `grid` is "minor".

            Notes:
                The `param` variable, which holds the styling parameters for the gridline
                (e.g., z-order, color, etc.), is defined in the enclosing scope.
            """
            assert grid in ["major", "minor"]
            assert arc_type in ["real", "imag"]
            assert p0 != p1
            if grid == "major":
                arcs = self._majorarcs
                if arc_type == "real":
                    param["color"] = self._get_key("grid.major.color.x")
                else:
                    param["color"] = self._get_key("grid.major.color.y")
            else:
                arcs = self._minorarcs
                if arc_type == "real":
                    param["color"] = self._get_key("grid.minor.color.x")
                else:
                    param["color"] = self._get_key("grid.minor.color.y")
                param["zorder"] -= 1e-09
            arcs.append(
                (
                    arc_type,
                    (ps, p0, p1),
                    self._add_gridline(ps, p0, p1, arc_type, **param),
                )
            )

        def draw_major_nonfancy():
            xticks = self.xaxis.get_majorticklocs()
            yticks = self.yaxis.get_majorticklocs()
            xticks = np.round(xticks, 7)
            yticks = np.round(yticks, 7)
            for xs in xticks:
                if xs < SC_NEAR_INFINITY:
                    add_arc(xs, -SC_NEAR_INFINITY, SC_INFINITY, "major", "real")
            for ys in yticks:
                if abs(ys) < SC_NEAR_INFINITY:
                    add_arc(ys, 0, SC_INFINITY, "major", "imag")

        def draw_minor_nonfancy():
            xticks = self.xaxis.get_minor_locator()()
            yticks = self.yaxis.get_minor_locator()()
            xticks = np.round(xticks, 7)
            yticks = np.round(yticks, 7)
            for xs in xticks:
                if xs < SC_NEAR_INFINITY:
                    add_arc(xs, -SC_NEAR_INFINITY, SC_INFINITY, "minor", "real")
            for ys in yticks:
                if abs(ys) < SC_NEAR_INFINITY:
                    add_arc(ys, 0, SC_INFINITY, "minor", "imag")

        def draw_major_fancy(threshold):
            xticks = np.sort(self.xaxis.get_majorticklocs())
            yticks = np.sort(self.yaxis.get_majorticklocs())
            assert len(xticks) > 0 and len(yticks) > 0
            yticks = check_fancy(yticks)
            if threshold is None:
                threshold = self._get_key("grid.major.fancy.threshold")
            thr_x, thr_y = split_threshold(threshold)
            add_arc(yticks[0], 0, SC_INFINITY, "major", "imag")
            tmp_yticks = yticks.copy()
            for xs in xticks:
                k = 1
                while k < len(tmp_yticks):
                    y0, y1 = tmp_yticks[k - 1 : k + 1]
                    if abs(self.moebius_z(xs, y0) - self.moebius_z(xs, y1)) < thr_x:
                        add_arc(y1, 0, xs, "major", "imag")
                        add_arc(-y1, 0, xs, "major", "imag")
                        tmp_yticks = np.delete(tmp_yticks, k)
                    else:
                        k += 1
            for i in range(1, len(yticks)):
                y0, y1 = yticks[i - 1 : i + 1]
                k = 1
                while k < len(xticks):
                    x0, x1 = xticks[k - 1 : k + 1]
                    if abs(self.moebius_z(x0, y1) - self.moebius_z(x1, y1)) < thr_y:
                        add_arc(x1, -y0, y0, "major", "real")
                        xticks = np.delete(xticks, k)
                    else:
                        k += 1

        def draw_minor_fancy(threshold, dividers):
            xticks = np.sort(self.xaxis.get_majorticklocs())
            yticks = np.sort(self.yaxis.get_majorticklocs())
            assert len(xticks) > 0 and len(yticks) > 0
            yticks = check_fancy(yticks)
            if dividers is None:
                dividers = self._get_key("grid.minor.fancy.dividers")
            assert len(dividers) > 0
            dividers = np.sort(dividers)
            if threshold is None:
                threshold = self._get_key("grid.minor.fancy.threshold")
            thr_x, thr_y = split_threshold(threshold)
            len_x, len_y = (len(xticks) - 1, len(yticks) - 1)
            d_mat = np.ones((len_x, len_y, 2), dtype=int)
            for i in range(len_x):
                for k in range(len_y):
                    x0, x1 = xticks[i : i + 2]
                    y0, y1 = yticks[k : k + 2]
                    xm = self.real_interp1d([x0, x1], 2)[1]
                    ym = self.imag_interp1d([y0, y1], 2)[1]
                    x_div = y_div = dividers[0]
                    for div in dividers[1:]:
                        if abs(self.moebius_z(x1 - (x1 - x0) / div, ym) - self.moebius_z(x1, ym)) > thr_x:
                            x_div = div
                        else:
                            break
                    for div in dividers[1:]:
                        if abs(self.moebius_z(xm, y1) - self.moebius_z(xm, y1 - (y1 - y0) / div)) > thr_y:
                            y_div = div
                        else:
                            break
                    d_mat[i, k] = [x_div, y_div]
            d_mat[:-1, 0, 0] = list(map(np.max, zip(d_mat[:-1, 0, 0], d_mat[1:, 0, 0])))
            idx = np.searchsorted(xticks, self.moebius_inv_z(0)) + 1
            idy = np.searchsorted(yticks, self.moebius_inv_z(1j).imag)
            if idx > idy:
                for d in range(idy):
                    delta = idx - idy + d
                    d_mat[delta, : d + 1] = d_mat[:delta, d] = d_mat[delta, 0]
            else:
                for d in range(idx):
                    delta = idy - idx + d
                    d_mat[: d + 1, delta] = d_mat[d, :delta] = d_mat[d, 0]
            x_lines, y_lines = ([], [])
            for i in range(len_x):
                x0, x1 = xticks[i : i + 2]
                for k in range(len_y):
                    y0, y1 = yticks[k : k + 2]
                    x_div, y_div = d_mat[i, k]
                    for xs in np.linspace(x0, x1, x_div + 1)[1:]:
                        x_lines.append([xs, y0, y1])
                        x_lines.append([xs, -y1, -y0])
                    for ys in np.linspace(y0, y1, y_div + 1)[1:]:
                        y_lines.append([ys, x0, x1])
                        y_lines.append([-ys, x0, x1])
            x_lines = np.round(np.array(x_lines), 7)
            y_lines = np.round(np.array(y_lines), 7)
            for tp, lines in [("real", x_lines), ("imag", y_lines)]:
                lines = np.array([[ps, min(p0, p1), max(p0, p1)] for ps, p0, p1 in lines])
                for tq, (qs, q0, q1), _ in self._majorarcs:
                    if tp == tq:
                        overlaps = (abs(lines[:, 0] - qs) < SC_EPSILON) & (lines[:, 2] > q0) & (lines[:, 1] < q1)
                        lines[overlaps] = np.nan
                lines = lines[~np.isnan(lines[:, 0])]
                lines = lines[np.lexsort((lines[:, 1], lines[:, 0]))]
                ps, p0, p1 = lines[0]
                for qs, q0, q1 in lines[1:]:
                    if ps != qs or not np.isclose(p1, q0, atol=SC_EPSILON):
                        add_arc(ps, p0, p1, "minor", tp)
                        ps, p0, p1 = (qs, q0, q1)
                    else:
                        p1 = q1

        if axis is None:
            fancy_major = self._get_key("grid.major.fancy")
            fancy_minor = self._get_key("grid.minor.fancy")
        else:
            fancy_major = fancy_minor = axis

        if "axis" in kwargs and kwargs["axis"] != "both":
            raise ValueError("Only 'both' is a supported value for 'axis'")

        # draw major grid lines
        if which in ["both", "major"]:
            for _, _, arc in self._majorarcs:
                arc.remove()
            self._majorarcs = []

            if visible:
                param = get_kwargs("major")
                if fancy_major:
                    draw_major_fancy(threshold)
                else:
                    draw_major_nonfancy()

        if which in ["both", "minor"]:
            for _, _, arc in self._minorarcs:
                arc.remove()
            self._minorarcs = []

            if visible:
                param = get_kwargs("minor")
                if fancy_minor:
                    draw_minor_fancy(threshold, dividers)
                else:
                    draw_minor_nonfancy()

    def hack_linedraw(self, line, rotate_marker):
        """
        Draw lines with different markers for start and end points.

        Modify the draw method of a `matplotlib.lines.Line2D` object to use
        different markers at the start and end points, optionally rotating the
        end marker to align with the path direction.

        This method customizes the appearance of lines by replacing the default
        marker behavior with dynamic start and end markers. It supports rotation
        of the end marker to follow the line's direction and ensures intermediate
        points retain the original marker style.

        Args:
            line (matplotlib.lines.Line2D):
                The line object to be modified.
            rotate_marker (bool):
                If `True`, the end marker is rotated to align with the tangent
                of the line's path. If `False`, the marker remains unrotated.

        Implementation Details:
            1. A nested `new_draw` method replaces the `Line2D.draw` method. This
               handles drawing start and end markers separately from intermediate points.
            2. If `rotate_marker` is enabled, the end marker is rotated to align
               with the path's direction using an affine transformation.
            3. The original `Line2D.draw` method is restored after rendering.
        """

        def to_marker_style(marker):
            if marker is None:
                return MarkerStyle("o")
            if isinstance(marker, MarkerStyle):
                return marker
            return MarkerStyle(marker)

        start_marker = self._get_key("plot.marker.start")
        end_marker = self._get_key("plot.marker.end")
        start = to_marker_style(start_marker)
        end = to_marker_style(end_marker)
        assert isinstance(line, Line2D)

        def new_draw(self_line, renderer):
            """
            Custom draw method for the line, allowing marker rotation and manipulation.

            Args:
                self_line (Line2D): The line object to draw.
                renderer: The renderer instance used to draw the line.
            """

            def new_draw_markers(
                _self_renderer,
                gc,
                _marker_path,
                _marker_trans,
                path,
                trans,
                rgbFace=None,
            ):
                """
                Custom draw method for markers on the line.

                Args:
                    self_renderer: Renderer for the markers.
                    gc: Graphics context.
                    marker_path: The path of the marker.
                    _marker_trans: (Unused) Transformation for the marker path.
                    path: Path for the line.
                    trans: Transformation for the path.
                    rgbFace: Fill color for the marker.
                """
                # pylint: disable=protected-access
                line_vertices = self_line._get_transformed_path().get_fully_transformed_path().vertices
                # pylint: enable=protected-access
                vertices = path.vertices

                if len(vertices) == 1:
                    # line with single point
                    line_set = [[to_marker_style(line.get_marker()), vertices]]
                else:
                    end_rot = to_marker_style(end)
                    if rotate_marker:
                        dx, dy = np.array(line_vertices[-1]) - np.array(line_vertices[-2])
                        end_rot._transform += Affine2D().rotate(np.arctan2(dy, dx) - np.pi / 2)

                    if len(vertices) == 2:
                        # line with two points
                        line_set = [[start, vertices[0:1]], [end_rot, vertices[1:2]]]
                    else:
                        # line with three or more points
                        line_set = [
                            [start, vertices[0:1]],
                            [to_marker_style(line.get_marker()), vertices[1:-1]],
                            [end_rot, vertices[-1:]],
                        ]

                for marker, points in line_set:
                    marker = to_marker_style(marker)
                    transform = marker.get_transform() + Affine2D().scale(self_line.get_markersize())
                    old_draw_markers(gc, marker.get_path(), transform, Path(points), trans, rgbFace)

            old_draw_markers = renderer.draw_markers
            renderer.draw_markers = MethodType(new_draw_markers, renderer)
            old_draw(renderer)
            renderer.draw_markers = old_draw_markers

        default_marker = to_marker_style(line.get_marker())
        if default_marker:
            start = to_marker_style(start)
            end = to_marker_style(end)
            if rotate_marker is None:
                rotate_marker = self._get_key("plot.marker.rotate")
            old_draw = line.draw
            line.draw = MethodType(new_draw, line)
            line.markers_hacked = True

    def _add_gridline(self, ps, p0, p1, arc_type, **kwargs):
        """
        Add a gridline to the Smith chart for the specified arc_type.

        This method creates and adds a gridline (real or imaginary) as a `matplotlib.lines.Line2D`
        object. Gridlines for the real axis are vertical lines (circles for constant resistance),
        while gridlines for the imaginary axis are horizontal lines (circles for constant reactance).

        For `arc_type='real'`, the gridline is drawn as a vertical line at the position `ps`.
        The start and end points of the line are defined by `p0` and `p1`.

        For `arc_type='imag'`, the gridline is drawn as a horizontal line at the position `ps`.
        The line spans between `p0` and `p1`.

        The `_interpolation_steps` property is set for efficient rendering,
        distinguishing between "x_gridline" and "y_gridline" types.

        Args:
            ps (float): The axis value for the gridline:
                - For `arc_type='real'`, this represents the resistance value.
                - For `arc_type='imag'`, this represents the reactance value.
            p0 (float): The start point of the gridline.
            p1 (float): The end point of the gridline.
            arc_type (str): The type of gridline to add. Must be either:
                - `'real'`: Gridline for the real axis (constant resistance).
                - `'imag'`: Gridline for the imaginary axis (constant reactance).
            **kwargs:
                Additional keyword arguments passed to the `matplotlib.lines.Line2D`
                constructor. These can be used to customize the gridline's appearance
                (e.g., color, linestyle, linewidth).
        """
        assert arc_type in ["real", "imag"]
        if arc_type == "real":
            assert ps >= 0
            line = Line2D(2 * [ps], [p0, p1], **kwargs)
            line.get_path()._interpolation_steps = "x_gridline"  # pylint: disable=protected-access
        else:
            assert 0 <= p0 < p1
            line = Line2D([p0, p1], 2 * [ps], **kwargs)
            if abs(ps) > SC_EPSILON:
                line.get_path()._interpolation_steps = "y_gridline"  # pylint: disable=protected-access
        return self.add_artist(line)
