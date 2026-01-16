"""Grid drawing functionality for SmithAxes."""

import numpy as np
from matplotlib.lines import Line2D

from pysmithchart.constants import SC_EPSILON, SC_INFINITY, SC_NEAR_INFINITY, SC_TWICE_INFINITY


class GridMixin:
    """Mixin class providing grid drawing methods for SmithAxes."""

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

        return self.add_line(line)
