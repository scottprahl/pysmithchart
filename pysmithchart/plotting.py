"""Plotting functionality for SmithAxes."""

from collections.abc import Iterable
from numbers import Number
from types import MethodType

import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from scipy.interpolate import splprep, splev

from pysmithchart.constants import Z_PARAMETER, Y_PARAMETER, S_PARAMETER, A_PARAMETER
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

                datatype (str, optional):
                    Specifies the input data format. Must be one of:
                    - `S_PARAMETER` ('S'): Scattering parameters (reflection coefficient).
                      Values are converted via inverse Möbius: z = (1+S)/(1-S).
                      Warning issued if |S| > 1.
                    - `Z_PARAMETER` ('Z'): Impedance in Ohms (always normalized by Z₀).
                      Values are divided by characteristic impedance Z₀.
                    - `A_PARAMETER` ('A'): Absolute/unnormalized coordinates.
                      Values are plotted as-is without any transformation.
                    - `Y_PARAMETER` ('Y'): Admittance (converted to impedance).
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
            ValueError: If `datatype` is not one of `S_PARAMETER`, `Z_PARAMETER`, `A_PARAMETER`, or `Y_PARAMETER`.
            UserWarning: If `datatype` is `S_PARAMETER` and |S| > 1 (point outside Smith chart).
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
        if datatype not in [S_PARAMETER, Z_PARAMETER, A_PARAMETER, Y_PARAMETER]:
            raise ValueError(f"Invalid datatype: {datatype}. Must be S_PARAMETER, Z_PARAMETER, A_PARAMETER, or Y_PARAMETER")

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
                # S-parameters: Check magnitude and warn if > 1
                s_magnitude = np.abs(cdata)
                if np.any(s_magnitude > 1):
                    import warnings
                    warnings.warn(
                        f"S-parameter magnitude |S| > 1 detected (max: {np.max(s_magnitude):.3f}). "
                        "Points outside the unit circle will not be visible on the Smith chart.",
                        UserWarning
                    )
                # Apply inverse Möbius with norm=1 (always normalized)
                # z = (1 + S) / (1 - S)
                z = self.moebius_inv_z(cdata, normalize=True)
                
            elif datatype == Z_PARAMETER:
                # Z-parameters: Always normalize by Z₀
                z = cdata / self._get_key("axes.impedance")
                
            elif datatype == A_PARAMETER:
                # A-parameters: Use as-is, no transformation
                z = cdata
                
            elif datatype == Y_PARAMETER:
                # Y-parameters: Convert to impedance, then normalize
                # Y is absolute admittance in Siemens, convert to absolute Z, then normalize
                z = (1 / cdata) / self._get_key("axes.impedance")
            else:
                z = cdata

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

    def text(self, x, y, s, datatype=None, transform=None, **kwargs):
        """
        Add text to the Smith chart at the specified coordinates.

        Args:
            x (float): Real part of the coordinate (or axes coordinate if transform is specified).
            y (float): Imaginary part of the coordinate (or axes coordinate if transform is specified).
            s (str): The text string to display.
            datatype (str, optional): Coordinate type (Z_PARAMETER, Y_PARAMETER, S_PARAMETER).
                Only used if transform is not specified or is self.transData.
            transform (Transform, optional): The coordinate transform. If not specified or is
                self.transData, coordinates will be transformed according to datatype.
                If specified as ax.transAxes or another non-data transform, coordinates
                will be used as-is without Smith chart transformation.
            **kwargs: Additional matplotlib text parameters.

        Returns:
            matplotlib.text.Text: The created text object.
        """
        # Check if we should apply Smith chart transformation
        # Handle 'transform' kwarg which may also be in kwargs dict
        if transform is None and "transform" in kwargs:
            transform = kwargs["transform"]

        if self._should_transform_coordinates(transform):
            # Get default datatype if not specified
            if datatype is None:
                datatype = self._get_key("plot.default.datatype")

            # Validate datatype
            if datatype not in [S_PARAMETER, Z_PARAMETER, Y_PARAMETER, A_PARAMETER]:
                raise ValueError(f"Invalid datatype: {datatype}. Must be S_PARAMETER, Z_PARAMETER, Y_PARAMETER, or A_PARAMETER")

            # Transform coordinates using the helper method
            x_transformed, y_transformed = self._transform_coordinates(x, y, datatype)

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
        datatype=None,
        datatype_text=None,
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
                Only 'data' coordinates are transformed according to datatype.
            textcoords (str or Transform, optional): Coordinate system for xytext.
                Defaults to xycoords value.
            datatype (str, optional): Coordinate type for xy (Z, Y, or S parameter).
                Only used when xycoords is 'data' or not specified.
            datatype_text (str, optional): Coordinate type for xytext.
                Only used when textcoords is 'data'. Defaults to datatype value.
            arrowprops (dict, optional): Arrow properties.
            annotation_clip (bool, optional): Whether to clip annotation.
            **kwargs: Additional matplotlib annotate parameters.

        Returns:
            matplotlib.text.Annotation: The annotation object.
        """
        # Determine if we should transform xy coordinates
        if self._should_transform_coordinates(xycoords):
            # Get default datatype if not specified
            if datatype is None:
                datatype = self._get_key("plot.default.datatype")

            # Validate datatype for xy
            if datatype not in [S_PARAMETER, Z_PARAMETER, Y_PARAMETER, A_PARAMETER]:
                raise ValueError(f"Invalid datatype: {datatype}. Must be S_PARAMETER, Z_PARAMETER, Y_PARAMETER, or A_PARAMETER")

            # Transform xy coordinates (the point being annotated)
            xy_transformed = self._transform_coordinates(xy[0], xy[1], datatype)
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
                # If datatype_text not specified, use same as datatype
                if datatype_text is None:
                    datatype_text = datatype if datatype is not None else self._get_key("plot.default.datatype")

                # Validate datatype_text
                if datatype_text not in [S_PARAMETER, Z_PARAMETER, Y_PARAMETER, A_PARAMETER]:
                    raise ValueError(f"Invalid datatype_text: {datatype_text}. Must be S_PARAMETER, Z_PARAMETER, Y_PARAMETER, or A_PARAMETER")

                # Transform xytext coordinates
                xytext_transformed = self._transform_coordinates(xytext[0], xytext[1], datatype_text)
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
