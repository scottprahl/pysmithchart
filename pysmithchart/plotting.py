"""Plotting functionality for SmithAxes."""

from collections.abc import Iterable
from numbers import Number

import numpy as np
from matplotlib.axes import Axes
from matplotlib import _color_data
from scipy.interpolate import splprep, splev

from pysmithchart.constants import IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, REFLECTANCE_DOMAIN, ABSOLUTE_DOMAIN
from pysmithchart import utils


class PlottingMixin:
    """Mixin class providing plotting methods for SmithAxes."""

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

                domain (str, optional):
                    Specifies the input data format. Must be one of:
                    - `REFLECTANCE_DOMAIN` ('S'): Scattering parameters (reflection coefficient).
                      Values are converted via inverse Möbius: z = (1+S)/(1-S).
                      Warning issued if |S| > 1.
                    - `IMPEDANCE_DOMAIN` ('Z'): Impedance in Ohms (always normalized by Z₀).
                      Values are divided by characteristic impedance Z₀.
                    - `ABSOLUTE_DOMAIN` ('A'): Absolute/unnormalized coordinates.
                      Values are plotted as-is without any transformation.
                    - `ADMITTANCE_DOMAIN` ('Y'): Admittance (converted to impedance).
                    Defaults to `IMPEDANCE_DOMAIN`.

                interpolate (bool or int, optional):
                    If `True`, interpolates the given data linearly with a default step size.
                    If an integer, specifies the number of interpolation steps.
                    Defaults to `False`.

                equipoints (bool or int, optional):
                    If `True`, interpolates the data to equidistant points. If an integer,
                    specifies the number of equidistant points. Cannot be used with
                    `interpolate`. Defaults to `False`.

                arrow (str, bool, or dict, optional):
                    Add directional arrow(s) to the curve.
                    - None/False: No arrows (default)
                    - True/'end': Arrow at end of line
                    - 'start': Arrow at start of line
                    - 'both': Arrows at both ends
                    - dict: {'position': 'end'/'start'/'both', 'style': '->', 'size': 15}

        Returns:
            list[matplotlib.lines.Line2D]:
                A list of line objects representing the plotted data.

        Raises:
            ValueError: If `domain` is not one of the valid domain constants.
            UserWarning: If `domain` is `REFLECTANCE_DOMAIN` and |S| > 1 (point outside Smith chart).
            ValueError: If both `interpolate` and `equipoints` are enabled.
            ValueError: If `interpolate` is specified with a non-positive value.

        Examples:
            Plot impedance data on a Smith Chart:

            >>> import matplotlib.pyplot as plt
            >>> import pysmithchart
            >>> ZL = [30 + 30j, 50 + 50j, 100 + 100j]
            >>> plt.subplot(1, 1, 1, projection="smith")
            >>> plt.plot(ZL, "b", marker="o", markersize=10, domain=pysmithchart.IMPEDANCE_DOMAIN)
            >>> plt.show()

            Plot with arrow showing direction:

            >>> ZL = [30 + 30j, 50 + 50j, 100 + 100j]
            >>> plt.subplot(1, 1, 1, projection="smith")
            >>> plt.plot(ZL, "r-", arrow='end', linewidth=2)
            >>> plt.show()
        """
        domain = kwargs.pop("domain", self._get_key("plot.default.domain"))
        domain = utils.validate_domain(domain)
        arrow = kwargs.pop("arrow", None)  # Extract arrow parameter

        # Parse arguments into x, y pairs and other args (like format strings)
        new_args = ()
        i = 0
        while i < len(args):
            arg = args[i]

            # If it's a string (format specifier), pass through
            if isinstance(arg, str):
                new_args += (arg,)
                i += 1
                continue

            # Check if it's a complex number or array of complex numbers
            is_complex = False
            if isinstance(arg, Number):
                is_complex = np.iscomplexobj(arg)
            elif isinstance(arg, Iterable):
                try:
                    arr = np.asarray(arg)
                    is_complex = np.iscomplexobj(arr)
                except (ValueError, TypeError):
                    pass

            if is_complex:
                # Handle complex input: convert to array and split into x, y
                if isinstance(arg, Number):
                    arg = np.array([arg])
                else:
                    arg = np.asarray(arg)
                new_args += utils.z_to_xy(arg)
                i += 1
            else:
                # Not complex - check if next arg could be y-values
                if i + 1 < len(args) and not isinstance(args[i + 1], str):
                    # We have two consecutive non-string args - treat as x, y
                    x_arg = arg
                    y_arg = args[i + 1]

                    # Convert to arrays
                    if isinstance(x_arg, Number):
                        x_arr = np.array([x_arg])
                    else:
                        x_arr = np.asarray(x_arg)

                    if isinstance(y_arg, Number):
                        y_arr = np.array([y_arg])
                    else:
                        y_arr = np.asarray(y_arg)

                    new_args += (x_arr, y_arr)
                    i += 2
                else:
                    # Single real number or array - treat as x with y=0
                    if isinstance(arg, Number):
                        x_arr = np.array([arg])
                    else:
                        x_arr = np.asarray(arg)
                    y_arr = np.zeros_like(x_arr)
                    new_args += (x_arr, y_arr)
                    i += 1

        if "zorder" not in kwargs:
            kwargs["zorder"] = self._current_zorder
            self._current_zorder += 0.001

        interpolate = kwargs.pop("interpolate", False)
        equipoints = kwargs.pop("equipoints", False)

        # Only set default marker if no format string in args and no marker in kwargs
        has_format_string = any(isinstance(arg, str) and arg for arg in new_args)
        if not has_format_string:
            kwargs.setdefault("marker", self._get_key("plot.marker.default"))

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

            # Apply unified domain transformation
            x_transformed, y_transformed = self._apply_domain_transform(cdata, domain=domain, warn_s_parameter=True)

            line.set_data(x_transformed, y_transformed)

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

        # Add arrows if requested
        if arrow and lines:
            for line in lines:
                self._add_arrows_to_line(line, arrow)

        return lines

    def scatter(self, x, y=None, *args, domain=None, **kwargs):
        """
        Create a scatter plot on the Smith Chart.

        Args:
            x: X coordinates (real part) or complex impedance/admittance values.
            y: Y coordinates (imaginary part). Ignored if x is complex.
            *args: Optional format string (not commonly used with scatter, use c= and marker= instead).
            domain: Data format (REFLECTANCE_DOMAIN, IMPEDANCE_DOMAIN, ABSOLUTE_DOMAIN, ADMITTANCE_DOMAIN).
            **kwargs: Additional arguments passed to matplotlib.axes.Axes.scatter (s, c, marker, etc.).

        Returns:
            PathCollection: The scatter plot collection.

        Examples:
            >>> # Recommended: use keyword arguments
            >>> ax.scatter(50+25j, s=100, c='red', marker='o')

            >>> # Also works: format string (converted to kwargs)
            >>> ax.scatter(50+25j, 'ro', s=100)

        Note:
            Unlike plot(), scatter() primarily uses keyword arguments. If you provide a format
            string like 'ro', it will be parsed for color and marker, but size must still be
            specified with s=.
        """
        # Parse format string if provided
        if args and isinstance(args[0], str):
            fmt = args[0]
            # Parse format string for color and marker

            # Extract color
            for color_key in _color_data.TABLEAU_COLORS.keys():
                if color_key[4] in fmt:  # 'tab:blue' -> 'b'
                    kwargs.setdefault("c", color_key[4])
                    break
            # Common single-letter colors
            color_map = {
                "r": "red",
                "g": "green",
                "b": "blue",
                "c": "cyan",
                "m": "magenta",
                "y": "yellow",
                "k": "black",
                "w": "white",
            }
            for char in fmt:
                if char in color_map:
                    kwargs.setdefault("c", char)
                    break

            # Extract marker
            marker_chars = ".ov^<>123478spP*hH+xXDd|_"
            for char in fmt:
                if char in marker_chars:
                    kwargs.setdefault("marker", char)
                    break

        # Get domain
        if domain is None:
            domain = self._get_key("plot.default.domain")
        domain = utils.validate_domain(domain)
        x_plot, y_plot = self._apply_domain_transform(x, y, domain=domain, warn_s_parameter=True)

        # Set zorder
        if "zorder" not in kwargs:
            kwargs["zorder"] = self._current_zorder
            self._current_zorder += 0.001

        # Call matplotlib scatter with separate x and y
        return Axes.scatter(self, x_plot, y_plot, **kwargs)

    def _add_arrows_to_line(self, line, arrow=None):
        """
        Add arrows to a plotted line.

        This is a helper method used by plot functions to add directional arrows
        to curves on the Smith chart.

        Args:
            line (matplotlib.lines.Line2D): The line object to add arrows to.
            arrow (str, bool, or dict, optional): Arrow specification.
                - None or False: No arrows (default)
                - True or 'end': Arrow at end of line
                - 'start': Arrow at start of line
                - 'both': Arrows at both ends
                - dict: Full control with keys:
                    - 'position': 'start', 'end', or 'both' (default: 'end')
                    - 'style': matplotlib arrowstyle (default: '->')
                    - 'size': mutation_scale for arrow size (default: 15)
                    - 'offset': number of points from end to use for arrow direction (default: 1)

        Returns:
            list: List of annotation objects created for the arrows.

        Examples:
            >>> lines = ax.plot([1+1j, 2+2j], 'r-')
            >>> ax._add_arrows_to_line(lines[0], arrow='end')

            >>> lines = ax.plot([1+1j, 2+2j], 'b-')
            >>> ax._add_arrows_to_line(lines[0], arrow={'position': 'both', 'size': 20})
        """
        if not arrow:
            return []

        # Get line data - these are already in the transformed (display) coordinates
        x, y = line.get_data()

        # Need at least 2 points for an arrow
        if len(x) < 2:
            return []

        # Parse arrow parameter
        if arrow is True or arrow == "end":
            arrow_spec = {"position": "end", "style": "->", "size": 15, "offset": 1}
        elif isinstance(arrow, str):
            arrow_spec = {"position": arrow, "style": "->", "size": 15, "offset": 1}
        elif isinstance(arrow, dict):
            arrow_spec = {
                "position": arrow.get("position", "end"),
                "style": arrow.get("style", "->"),
                "size": arrow.get("size", 15),
                "offset": arrow.get("offset", 1),
            }
        else:
            return []

        # Extract arrow properties
        position = arrow_spec["position"]
        style = arrow_spec["style"]
        size = arrow_spec["size"]
        offset = arrow_spec["offset"]

        # Get visual properties from the line
        color = line.get_color()
        lw = line.get_linewidth()

        # Arrow properties
        arrow_props = dict(arrowstyle=style, lw=lw, color=color, mutation_scale=size)

        annotations = []

        # Add arrow at start
        # The key fix: use the line's transform (which is self.transData for Smith chart)
        # This ensures arrows are drawn in the same coordinate system as the line
        if position in ["start", "both"]:
            if len(x) > offset:
                ann = Axes.annotate(
                    self,
                    "",
                    xy=(x[offset], y[offset]),
                    xytext=(x[0], y[0]),
                    xycoords="data",
                    textcoords="data",
                    arrowprops=arrow_props,
                )
                annotations.append(ann)

        # Add arrow at end
        if position in ["end", "both"]:
            if len(x) > offset:
                ann = Axes.annotate(
                    self,
                    "",
                    xy=(x[-1], y[-1]),
                    xytext=(x[-(offset + 1)], y[-(offset + 1)]),
                    xycoords="data",
                    textcoords="data",
                    arrowprops=arrow_props,
                )
                annotations.append(ann)

        return annotations

    def text(self, x, y=None, s=None, domain=None, **kwargs):
        """
        Add text to the Smith chart at the specified coordinates.

        Args:
            x (float or complex): Real part of the coordinate, or complex impedance value.
                If complex, y parameter is ignored and s becomes the second argument.
            y (float or str, optional): Imaginary part (if x is real), or text string (if x is complex).
            s (str, optional): The text string to display (if x and y are real).
            domain (str, optional): Coordinate type (IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, REFLECTANCE_DOMAIN).
                Default: Uses plot.default.domain from smith_params.
            **kwargs: Additional matplotlib text parameters including:
                - transform: Coordinate transform (default uses data coordinates with Smith chart transformation).
                  If you specify transform=ax.transAxes or another non-data transform, coordinates will be
                  used as-is without Smith chart transformation.
                - All standard matplotlib text properties (fontsize, color, ha, va, etc.)

        Returns:
            matplotlib.text.Text: The created text object.

        Examples:
            >>> # Text at impedance coordinates (default behavior)
            >>> ax.text(50, 25, "Point A")  # Real and imaginary parts
            >>> ax.text(50+25j, "Point A")  # Complex impedance

            >>> # Text in axes coordinates (0-1 range, no transformation)
            >>> ax.text(0.5, 0.95, "Title", transform=ax.transAxes, ha='center')

            >>> # Text with styling
            >>> ax.text(75+50j, "Load", fontsize=12, color='red', ha='left')
        """
        # Extract transform from kwargs if present
        transform = kwargs.pop("transform", None)

        # Handle complex input: text(complex, string, ...)
        if np.iscomplexobj(x):
            if y is None:
                raise ValueError("When x is complex, y must be the text string")
            # x is complex, y is the string, s is actually in kwargs or None
            if s is not None:
                # User passed text(complex, something, something_else)
                # This is ambiguous, but likely: text(z, text, domain=...)
                # Put s back into kwargs as domain if it's a valid domain
                if s in [REFLECTANCE_DOMAIN, IMPEDANCE_DOMAIN, ABSOLUTE_DOMAIN, ADMITTANCE_DOMAIN]:
                    domain = s
                s = y  # y is the text string
            else:
                s = y  # y is the text string
            # Split complex into real and imaginary
            x, y = np.real(x), np.imag(x)
        elif y is None:
            raise ValueError("Must provide both x and y coordinates, or a complex coordinate")
        elif s is None:
            raise ValueError("Must provide text string")

        # Check if we should apply Smith chart transformation
        if self._should_transform_coordinates(transform):
            # Get default domain if not specified
            if domain is None:
                domain = self._get_key("plot.default.domain")

            # Transform coordinates using the helper method
            x_transformed, y_transformed = self._transform_coordinates(x, y, domain)

            # Call parent with transformed coordinates
            # Don't pass transform parameter - let matplotlib use default (transData)
            return super(PlottingMixin, self).text(x_transformed, y_transformed, s, **kwargs)
        else:
            # User specified a non-data transform, use coordinates as-is
            return super(PlottingMixin, self).text(x, y, s, transform=transform, **kwargs)

    def annotate(
        self,
        text,
        xy,
        xytext=None,
        xycoords="data",
        textcoords=None,
        domain=None,
        domain_text=None,
        arrowprops=None,
        annotation_clip=None,
        **kwargs,
    ):
        """
        Add an annotation (text with optional arrow) to the Smith chart.

        Args:
            text (str): The text of the annotation.
            xy (tuple): The point (x, y) to annotate.
            xytext (tuple, optional): Position (x, y) for the text. If None, text is placed at xy.
            xycoords (str or Transform, optional): Coordinate system for xy.
                Default is 'data'. Can be 'data', 'axes', 'figure', or a Transform.
                Only 'data' coordinates are transformed according to domain.
            textcoords (str or Transform, optional): Coordinate system for xytext.
                Defaults to xycoords value.
            domain (str, optional): Coordinate type for xy (IMPEDANCE, ADMITTANCE, or REFLECTION domain).
                Only used when xycoords is 'data' or not specified.
            domain_text (str, optional): Coordinate type for xytext.
                Only used when textcoords is 'data'. Defaults to domain value.
            arrowprops (dict, optional): Arrow properties.
            annotation_clip (bool, optional): Whether to clip annotation.
            **kwargs: Additional matplotlib annotate parameters.

        Returns:
            matplotlib.text.Annotation: The annotation object.
        """
        # Determine if we should transform xy coordinates
        if self._should_transform_coordinates(xycoords):
            # Get default domain if not specified
            if domain is None:
                domain = self._get_key("plot.default.domain")

            # Transform xy coordinates (the point being annotated)
            xy_transformed = self._transform_coordinates(xy[0], xy[1], domain)
        else:
            # xycoords is not 'data', use coordinates as-is
            xy_transformed = xy

        # Handle xytext coordinates if provided
        if xytext is not None:
            # If textcoords not specified, it defaults to xycoords
            if textcoords is None:
                textcoords = xycoords

            # Determine if we should transform xytext coordinates
            if self._should_transform_coordinates(textcoords):
                # If domain_text not specified, use same as domain
                if domain_text is None:
                    domain_text = domain if domain is not None else self._get_key("plot.default.domain")

                # Transform xytext coordinates
                xytext_transformed = self._transform_coordinates(xytext[0], xytext[1], domain_text)
            else:
                # textcoords is not 'data', use coordinates as-is
                xytext_transformed = xytext
        else:
            xytext_transformed = None

        # Call parent annotate with transformed coordinates
        return super(PlottingMixin, self).annotate(
            text,
            xy=xy_transformed,
            xytext=xytext_transformed,
            xycoords=xycoords,
            textcoords=textcoords,
            arrowprops=arrowprops,
            annotation_clip=annotation_clip,
            **kwargs,
        )

    def legend(self, *args, **kwargs):
        """
        Create and display a legend for the Smith chart, filtering duplicate entries.

        This method filters out duplicate legend labels, keeping only the first occurrence.

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
        # Get handles and labels, filtering out duplicates
        handles, labels = self.get_legend_handles_labels()
        seen_labels = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                seen_labels.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)

        return Axes.legend(self, unique_handles, unique_labels, **kwargs)

    def plot_constant_resistance(
        self, resistance, *args, reactance_range=None, domain=None, num_points=500, arrow=None, **kwargs
    ):
        """
        Plot a constant resistance circle on the Smith chart.

        Args:
            resistance (float): The resistance value to plot.
                - For IMPEDANCE_DOMAIN: Value in Ohms (will be normalized by Z₀)
                - For ABSOLUTE_DOMAIN: Normalized value (used as-is, typically r = R/Z₀)
            *args: Optional format string (e.g., 'r-', 'b--', 'go').
            reactance_range (tuple, optional): The (min, max) reactance range to plot.
                If None, draws a complete circle. If specified, draws an arc between
                the min and max reactance values.
            domain (str, optional): Domain for the data (IMPEDANCE_DOMAIN or ABSOLUTE_DOMAIN).
                Default: IMPEDANCE_DOMAIN (values in Ohms, normalized by Z₀).
            num_points (int, optional): Number of points to use for the circle (default: 200).
            arrow (str, bool, or dict, optional): Add directional arrow(s) to the curve.
                - None/False: No arrows (default)
                - True/'end': Arrow at end
                - 'start': Arrow at start
                - 'both': Arrows at both ends
                - dict: {'position': 'end'/'start'/'both', 'style': '->', 'size': 15}
            **kwargs: Additional keyword arguments passed to plot() (e.g., color, linestyle, label).

        Returns:
            list[matplotlib.lines.Line2D]: The plotted line objects.

        Examples:
            >>> # Plot 50Ω constant resistance circle (normalized by Z₀=50Ω)
            >>> ax.plot_constant_resistance(50, 'r-', label='R = 50Ω')

            >>> # With arrow showing direction
            >>> ax.plot_constant_resistance(50, 'r-', arrow='end', label='R = 50Ω')

            >>> # Plot normalized r=1.0 circle using absolute domain
            >>> ax.plot_constant_resistance(1.0, 'b-', domain=ABSOLUTE_DOMAIN, label='r = 1.0')

            >>> # Plot arc with limited reactance range and both arrows
            >>> ax.plot_constant_resistance(75, 'g--', reactance_range=(-100, 100), arrow='both')

        Notes:
            On a Smith chart, constant resistance forms a circle. The circle is parametrized
            by varying the reactance from -∞ to +∞. For a complete circle, the function uses
            angular parametrization. For a partial arc, it uses the specified reactance range.
        """
        # Default to IMPEDANCE_DOMAIN if not specified
        if domain is None:
            domain = IMPEDANCE_DOMAIN

        if reactance_range is None:
            # Draw complete circle using angular parametrization
            theta = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, num_points)

            # Use tangent to span from -large to +large reactance
            if domain == ABSOLUTE_DOMAIN:
                X = 10 * resistance * np.tan(theta)
            else:  # IMPEDANCE_DOMAIN
                Z0 = self._get_key("axes.Z0")
                X = 10 * max(Z0, resistance) * np.tan(theta)

            Z = resistance + 1j * X
        else:
            # Draw arc with specified reactance range
            X = np.linspace(reactance_range[0], reactance_range[1], num_points)
            Z = resistance + 1j * X

        # Plot the circle with optional format string and arrow
        if args:
            return self.plot(Z, *args, domain=domain, arrow=arrow, **kwargs)
        else:
            return self.plot(Z, domain=domain, arrow=arrow, **kwargs)

    def plot_constant_reactance(
        self, reactance, *args, resistance_range=None, domain=None, num_points=200, arrow=None, **kwargs
    ):
        """
        Plot a constant reactance arc on the Smith chart.

        Args:
            reactance (float): The reactance value to plot.
                - For IMPEDANCE_DOMAIN: Value in Ohms (will be normalized by Z₀)
                - For ABSOLUTE_DOMAIN: Normalized value (used as-is, typically x = X/Z₀)
                Positive for inductive, negative for capacitive.
            *args: Optional format string (e.g., 'r-', 'b--', 'go').
            resistance_range (tuple, optional): The (min, max) resistance range to plot.
                If None, automatically determines range to show the full arc.
            domain (str, optional): Domain for the data (IMPEDANCE_DOMAIN or ABSOLUTE_DOMAIN).
                Default: IMPEDANCE_DOMAIN (values in Ohms, normalized by Z₀).
            num_points (int, optional): Number of points to use for the arc (default: 200).
            arrow (str, bool, or dict, optional): Add directional arrow(s) to the curve.
                - None/False: No arrows (default)
                - True/'end': Arrow at end
                - 'start': Arrow at start
                - 'both': Arrows at both ends
                - dict: {'position': 'end'/'start'/'both', 'style': '->', 'size': 15}
            **kwargs: Additional keyword arguments passed to plot() (e.g., color, linestyle, label).

        Returns:
            list[matplotlib.lines.Line2D]: The plotted line objects.

        Examples:
            >>> # Plot +50Ω constant reactance arc (inductive)
            >>> ax.plot_constant_reactance(50, 'r-', label='X = +50Ω (inductive)')

            >>> # Plot normalized x=1.0 using absolute domain
            >>> ax.plot_constant_reactance(1.0, 'b-', domain=ABSOLUTE_DOMAIN, label='x = 1.0')

            >>> # Plot with custom resistance range
            >>> ax.plot_constant_reactance(75, 'g--', resistance_range=(0, 200))

        Notes:
            On a Smith chart, constant reactance forms circular arcs. The arcs are parametrized
            by varying the resistance from 0 to ∞. Positive reactance (inductive) appears in the
            upper half of the chart, negative reactance (capacitive) in the lower half.
        """
        # Default to IMPEDANCE_DOMAIN if not specified
        if domain is None:
            domain = IMPEDANCE_DOMAIN

        # Determine resistance range if not specified
        if resistance_range is None:
            if domain == ABSOLUTE_DOMAIN:
                # For normalized/absolute, use range that covers most of the chart
                resistance_range = (0.01, 10)
            else:  # IMPEDANCE_DOMAIN
                # For absolute values in Ohms, use range based on Z0
                Z0 = self._get_key("axes.Z0")
                resistance_range = (0.01, 10 * Z0)

        # Generate points along constant reactance
        R = np.linspace(resistance_range[0], resistance_range[1], num_points)
        Z = R + 1j * reactance

        # Plot the circle with optional format string
        if args:
            lines = self.plot(Z, *args, domain=domain, **kwargs)
        else:
            lines = self.plot(Z, domain=domain, **kwargs)

        # Add arrows if requested
        if arrow and lines:
            self._add_arrows_to_line(lines[0], arrow)

        return lines

    def plot_constant_conductance(
        self, conductance, *args, susceptance_range=None, domain=None, num_points=500, arrow=None, **kwargs
    ):
        """
        Plot a constant conductance circle on the Smith chart (admittance chart).

        Constant conductance forms a circle on an admittance Smith chart, just as
        constant resistance forms a circle on an impedance Smith chart.

        Args:
            conductance (float): The conductance value to plot.
                - For ADMITTANCE_DOMAIN: Value in Siemens (will be normalized by Y₀=1/Z₀)
                - For ABSOLUTE_DOMAIN: Normalized value (used as-is, typically g = G×Z₀)
            *args: Optional format string (e.g., 'r-', 'b--', 'go').
            susceptance_range (tuple, optional): The (min, max) susceptance range to plot.
                If None, draws a complete circle.
            domain (str, optional): Domain for the data (ADMITTANCE_DOMAIN or ABSOLUTE_DOMAIN).
                Default: ADMITTANCE_DOMAIN (values in Siemens, normalized by Y₀).
            num_points (int, optional): Number of points to use for the circle (default: 200).
            arrow (str, bool, or dict, optional): Add directional arrow(s) to the curve.
                - None/False: No arrows (default)
                - True/'end': Arrow at end
                - 'start': Arrow at start
                - 'both': Arrows at both ends
                - dict: {'position': 'end'/'start'/'both', 'style': '->', 'size': 15}
            **kwargs: Additional keyword arguments passed to plot() (e.g., color, linestyle, label).

        Returns:
            list[matplotlib.lines.Line2D]: The plotted line objects.

        Examples:
            >>> # Plot 0.02S constant conductance circle (Y₀=1/50Ω=0.02S)
            >>> ax.plot_constant_conductance(0.02, 'r-', label='G = 0.02S')

            >>> # Plot normalized g=1.0 circle using absolute domain
            >>> ax.plot_constant_conductance(1.0, 'b-', domain=ABSOLUTE_DOMAIN, label='g = 1.0')

            >>> # Plot with custom susceptance range
            >>> ax.plot_constant_conductance(0.02, 'g--', susceptance_range=(-0.05, 0.05))

        Notes:
            On an admittance Smith chart, constant conductance forms a circle, just like
            constant resistance on an impedance chart. The circle is parametrized by varying
            susceptance from -∞ to +∞.
        """
        # Default to ADMITTANCE_DOMAIN if not specified
        if domain is None:
            domain = ADMITTANCE_DOMAIN

        if susceptance_range is None:
            # Draw complete circle using angular parametrization
            theta = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, num_points)

            # Use tangent to span from -large to +large susceptance
            if domain == ABSOLUTE_DOMAIN:
                B = 10 * conductance * np.tan(theta)
            else:  # ADMITTANCE_DOMAIN
                Z0 = self._get_key("axes.Z0")
                Y0 = 1 / Z0
                B = 10 * max(Y0, conductance) * np.tan(theta)

            Y = conductance + 1j * B
        else:
            # Draw arc with specified susceptance range
            B = np.linspace(susceptance_range[0], susceptance_range[1], num_points)
            Y = conductance + 1j * B

        # Plot the circle with optional format string
        if args:
            lines = self.plot(Y, *args, domain=domain, **kwargs)
        else:
            lines = self.plot(Y, domain=domain, **kwargs)

        # Add arrows if requested
        if arrow and lines:
            self._add_arrows_to_line(lines[0], arrow)

        return lines

    def plot_constant_susceptance(
        self, susceptance, *args, conductance_range=None, domain=None, num_points=200, arrow=None, **kwargs
    ):
        """
        Plot a constant susceptance arc on the Smith chart (admittance chart).

        Constant susceptance forms an arc on an admittance Smith chart, just as
        constant reactance forms an arc on an impedance Smith chart.

        Args:
            susceptance (float): The susceptance value to plot.
                - For ADMITTANCE_DOMAIN: Value in Siemens (will be normalized by Y₀=1/Z₀)
                - For ABSOLUTE_DOMAIN: Normalized value (used as-is, typically b = B×Z₀)
                Positive for capacitive, negative for inductive.
            *args: Optional format string (e.g., 'r-', 'b--', 'go').
            conductance_range (tuple, optional): The (min, max) conductance range to plot.
                If None, automatically determines range to show the full arc.
            domain (str, optional): Domain for the data (ADMITTANCE_DOMAIN or ABSOLUTE_DOMAIN).
                Default: ADMITTANCE_DOMAIN (values in Siemens, normalized by Y₀).
            num_points (int, optional): Number of points to use for the arc (default: 200).
            arrow (str, bool, or dict, optional): Add directional arrow(s) to the curve.
                - None/False: No arrows (default)
                - True/'end': Arrow at end
                - 'start': Arrow at start
                - 'both': Arrows at both ends
                - dict: {'position': 'end'/'start'/'both', 'style': '->', 'size': 15}
            **kwargs: Additional keyword arguments passed to plot() (e.g., color, linestyle, label).

        Returns:
            list[matplotlib.lines.Line2D]: The plotted line objects.

        Examples:
            >>> # Plot +0.02S constant susceptance arc (capacitive)
            >>> ax.plot_constant_susceptance(0.02, 'r-', label='B = +0.02S (capacitive)')

            >>> # Plot -0.02S constant susceptance arc (inductive)
            >>> ax.plot_constant_susceptance(-0.02, 'b-', label='B = -0.02S (inductive)')

            >>> # Plot normalized b=1.0 arc
            >>> ax.plot_constant_susceptance(1.0, 'g-', domain=ABSOLUTE_DOMAIN, label='b = 1.0')

            >>> # Plot with custom conductance range
            >>> ax.plot_constant_susceptance(0.02, 'g--', conductance_range=(0, 0.05))

        Notes:
            On an admittance Smith chart, constant susceptance forms circular arcs. The arcs
            are parametrized by varying conductance from 0 to ∞. Positive susceptance (capacitive)
            appears in the upper half, negative susceptance (inductive) in the lower half.

            Note: The sign convention is opposite to reactance - positive susceptance is capacitive,
            while positive reactance is inductive.
        """
        if domain is None:
            domain = ADMITTANCE_DOMAIN

        # Determine conductance range if not specified
        if conductance_range is None:
            if domain == ABSOLUTE_DOMAIN:
                # For normalized/absolute, use range that covers most of the chart
                conductance_range = (0.01, 10)
            else:  # ADMITTANCE_DOMAIN
                # For absolute values in Siemens, use range based on Y0
                Z0 = self._get_key("axes.Z0")
                Y0 = 1 / Z0
                conductance_range = (0.01, 10 * Y0)

        # Generate points along constant susceptance
        G = np.linspace(conductance_range[0], conductance_range[1], num_points)
        Y = G + 1j * susceptance

        # Plot the circle with optional format string
        if args:
            lines = self.plot(Y, *args, domain=domain, **kwargs)
        else:
            lines = self.plot(Y, domain=domain, **kwargs)

        # Add arrows if requested
        if arrow and lines:
            self._add_arrows_to_line(lines[0], arrow)

        return lines

    def plot_vswr(self, vswr, *args, angle_range=None, num_points=200, arrow=None, **kwargs):
        """
        Plot a constant VSWR circle on the Smith chart.

        A constant VSWR circle represents all impedances with the same voltage standing
        wave ratio. The circle is centered at the chart center with radius |Γ|.

        Args:
            vswr (float): The VSWR value to plot. Must be >= 1.0.
                VSWR = 1.0 is a perfect match (center point).
                VSWR = ∞ is the outer edge of the Smith chart.
            *args: Optional format string (e.g., 'r-', 'b--', 'go').
            angle_range (tuple, optional): The (start_angle, end_angle) in degrees to plot.
                If None, plots the full circle (0° to 360°).
                Angles are measured counterclockwise from the positive real axis.
            num_points (int, optional): Number of points to use for the circle (default: 200).
            arrow (str, bool, or dict, optional): Add directional arrow(s) to the curve.
                - None/False: No arrows (default)
                - True/'end': Arrow at end
                - 'start': Arrow at start
                - 'both': Arrows at both ends
                - dict: {'position': 'end'/'start'/'both', 'style': '->', 'size': 15}
            **kwargs: Additional keyword arguments passed to plot() (e.g., color, linestyle, label).

        Returns:
            list[matplotlib.lines.Line2D]: The plotted line objects.

        Raises:
            ValueError: If vswr < 1.0.

        Examples:
            >>> # Plot VSWR = 2.0 circle
            >>> ax.plot_vswr(2.0, 'r-', label='VSWR = 2.0')

            >>> # Plot partial arc from 0° to 180°
            >>> ax.plot_vswr(1.5, 'b--', angle_range=(0, 180))

            >>> # Plot VSWR = 3.0 with custom styling
            >>> ax.plot_vswr(3.0, color='green', linestyle='--', linewidth=2)

        Notes:
            The relationship between VSWR and reflection coefficient magnitude is:
            |Γ| = (VSWR - 1) / (VSWR + 1)
        """
        if vswr < 1.0:
            raise ValueError(f"VSWR must be >= 1.0, got {vswr}")

        # Calculate reflection coefficient magnitude from VSWR
        # |Γ| = (VSWR - 1) / (VSWR + 1)
        gamma_mag = (vswr - 1) / (vswr + 1)

        # Determine angle range
        if angle_range is None:
            angle_range = (0, 360)

        # Generate points around the circle
        angles = np.linspace(np.radians(angle_range[0]), np.radians(angle_range[1]), num_points)

        # Create reflection coefficients on the circle
        gamma = gamma_mag * np.exp(1j * angles)

        # Plot the circle with optional format string
        if args:
            lines = self.plot(gamma, *args, domain=REFLECTANCE_DOMAIN, **kwargs)
        else:
            lines = self.plot(gamma, domain=REFLECTANCE_DOMAIN, **kwargs)

        # Add arrows if requested
        if arrow and lines:
            self._add_arrows_to_line(lines[0], arrow)

        return lines

    def plot_rotation_path(self, Z_start, Z_end, *args, domain=None, num_points=100, arrow=None, **kwargs):
        """
        Plot a physically realizable impedance matching path.

        For impedances at the same VSWR: Draws a single arc along the constant-VSWR circle.
        For impedances at different VSWR: Draws a two-step path:
            Step 1: Rotate along constant-VSWR circle (transmission line)
            Step 2: Move toward center (reactive element)

        Args:
            Z_start: Starting impedance (complex number).
            Z_end: Ending impedance (complex number).
            *args: Optional format string (e.g., 'r-', 'b--').
            domain: Domain for the impedances (IMPEDANCE_DOMAIN, ABSOLUTE_DOMAIN, etc.).
                   Default: IMPEDANCE_DOMAIN.
            num_points (int): Number of points for smooth path (default: 100).
            arrow (str, bool, or dict, optional): Add directional arrow(s).
                - For single arc (same VSWR): Arrow added to the arc
                - For two-step path: Arrows added to both steps
                - None/False: No arrows (default)
                - True/'end': Arrow at end of each segment
                - 'start': Arrow at start
                - 'both': Arrows at both ends
                - dict: {'position': 'end'/'start'/'both', 'style': '->', 'size': 15}
            **kwargs: Additional plot arguments (color, linestyle, label, etc.).

        Returns:
            list: List of line objects. Single line for same VSWR, two lines otherwise.

        Examples:
            >>> # Same VSWR - single arc with arrow
            >>> ax.plot_rotation_path(75+50j, 100+50j, 'r-', arrow='end', label='Rotation')

            >>> # Different VSWR - two-step path with arrows
            >>> ax.plot_rotation_path(75+50j, 50+0j, 'b--', arrow='end', label='Matching')

        Notes:
            For same VSWR: Represents traveling along a lossless transmission line.
            For different VSWR: Represents a matching network with transmission line + reactive element.
        """
        # Default domain
        if domain is None:
            domain = IMPEDANCE_DOMAIN

        # Convert to complex if needed
        if not isinstance(Z_start, complex):
            if hasattr(Z_start, "__iter__"):
                Z_start = complex(Z_start[0], Z_start[1])
            else:
                Z_start = complex(Z_start, 0)

        if not isinstance(Z_end, complex):
            if hasattr(Z_end, "__iter__"):
                Z_end = complex(Z_end[0], Z_end[1])
            else:
                Z_end = complex(Z_end, 0)

        # Transform to normalized impedance
        Z0 = self._get_key("axes.Z0")

        if domain == IMPEDANCE_DOMAIN:
            z_start_norm = Z_start / Z0
            z_end_norm = Z_end / Z0
        elif domain == ABSOLUTE_DOMAIN:
            z_start_norm = Z_start
            z_end_norm = Z_end
        elif domain == ADMITTANCE_DOMAIN:
            z_start_norm = (1 / Z_start) / Z0
            z_end_norm = (1 / Z_end) / Z0
        elif domain == REFLECTANCE_DOMAIN:
            z_start_norm = utils.moebius_inverse_transform(Z_start, norm=1)
            z_end_norm = utils.moebius_inverse_transform(Z_end, norm=1)
        else:
            z_start_norm = Z_start
            z_end_norm = Z_end

        # Convert to reflection coefficients
        gamma_start = utils.moebius_transform(z_start_norm, norm=1)
        gamma_end = utils.moebius_transform(z_end_norm, norm=1)

        mag_start = np.abs(gamma_start)
        mag_end = np.abs(gamma_end)

        # Check if they're on the same VSWR circle (within tolerance)
        vswr_tolerance = 0.01  # 1% tolerance
        same_vswr = np.abs(mag_start - mag_end) < vswr_tolerance

        fmt_str = args[0] if args else None
        fmt_kwargs = kwargs.copy()

        if same_vswr:
            # CASE 1: Same VSWR - single arc along constant-VSWR circle
            avg_mag = 0.5 * (mag_start + mag_end)

            angle_start = np.angle(gamma_start)
            angle_end = np.angle(gamma_end)

            # Find shortest arc
            angle_diff = angle_end - angle_start
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            angles = np.linspace(angle_start, angle_start + angle_diff, num_points)
            gamma_path = avg_mag * np.exp(1j * angles)

            # Plot single arc with arrow
            if fmt_str:
                return self.plot(gamma_path, fmt_str, domain=REFLECTANCE_DOMAIN, arrow=arrow, **fmt_kwargs)
            else:
                return self.plot(gamma_path, domain=REFLECTANCE_DOMAIN, arrow=arrow, **fmt_kwargs)

        else:
            # CASE 2: Different VSWR - two-step matching path

            # Intermediate point: rotate to real axis on start VSWR circle
            angle_start = np.angle(gamma_start)
            angle_end = np.angle(gamma_end)

            # Choose which real axis crossing is closer to end point
            if np.abs(angle_end - 0) < np.abs(angle_end - np.pi):
                angle_intermediate = 0.0  # Positive real axis
            else:
                angle_intermediate = np.pi  # Negative real axis

            gamma_intermediate = mag_start * np.exp(1j * angle_intermediate)

            # STEP 1: Rotate along constant VSWR from start to real axis
            angle_diff = angle_intermediate - angle_start
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            angles_step1 = np.linspace(angle_start, angle_start + angle_diff, num_points // 2)
            gamma_step1 = mag_start * np.exp(1j * angles_step1)

            # STEP 2: Move from intermediate point toward center (radial)
            t = np.linspace(0, 1, num_points // 2)
            gamma_step2 = gamma_intermediate * (1 - t) + gamma_end * t

            lines = []

            # Plot step 1 (transmission line rotation) with arrow
            if fmt_str:
                lines.extend(self.plot(gamma_step1, fmt_str, domain=REFLECTANCE_DOMAIN, arrow=arrow, **fmt_kwargs))
            else:
                lines.extend(self.plot(gamma_step1, domain=REFLECTANCE_DOMAIN, arrow=arrow, **fmt_kwargs))

            # Plot step 2 (reactive element) with arrow - remove label to avoid duplicate
            fmt_kwargs_2 = fmt_kwargs.copy()
            fmt_kwargs_2.pop("label", None)

            if fmt_str:
                lines.extend(self.plot(gamma_step2, fmt_str, domain=REFLECTANCE_DOMAIN, arrow=arrow, **fmt_kwargs_2))
            else:
                lines.extend(self.plot(gamma_step2, domain=REFLECTANCE_DOMAIN, arrow=arrow, **fmt_kwargs_2))

            return lines
