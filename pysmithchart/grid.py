"""Grid drawing functionality using constant value plotting functions."""

import numpy as np

from pysmithchart.constants import Y_DOMAIN, NORM_Z_DOMAIN
from pysmithchart.constants import SC_EPSILON, SC_NEAR_INFINITY
from pysmithchart.utils import choose_minor_divider


class GridMixin:
    """Mixin class providing grid drawing methods for SmithAxes."""

    def grid(self, grid="impedance", **kwargs):
        """
        Draw gridlines on the Smith chart.

        The grid is controlled by configuration parameters:
        - grid.Z.major.enable / grid.Z.minor.enable (impedance)
        - grid.Y.major.enable / grid.Y.minor.enable (admittance)
        - grid.fancy (enables adaptive clipping for both grids)

        Args:
            grid (str): 'impedance', 'admittance', or 'both' (default: 'both')
            **kwargs: Styling parameters that override configuration
        """
        assert grid in ["impedance", "admittance", "both"]

        fancy = self._get_key("grid.fancy")
        draw_impedance = grid in ["impedance", "both"]
        draw_admittance = grid in ["admittance", "both"]

        # Draw impedance grids
        if draw_impedance:
            if self._get_key("grid.Z.major.enable"):
                self._draw_impedance_major(fancy, **kwargs)
            if self._get_key("grid.Z.minor.enable"):
                self._draw_impedance_minor(fancy, **kwargs)

        # Draw admittance grids
        if draw_admittance:
            if self._get_key("grid.Y.major.enable"):
                self._draw_admittance_major(fancy, **kwargs)
            if self._get_key("grid.Y.minor.enable"):
                self._draw_admittance_minor(fancy, **kwargs)

    # ========== IMPEDANCE DRAWING ==========

    def _draw_impedance_major(self, fancy, **kwargs):
        """Draw major impedance gridlines.

        Args:
            fancy: If True, use adaptive clipping based on Möbius distance
            **kwargs: Style overrides
        """
        style = self._get_grid_style("impedance", "major", **kwargs)
        style.setdefault("marker", "")

        # Get major ticks and ranges (domain-agnostic, returns ABSOLUTE values)
        xticks = np.sort(self.xaxis.get_majorticklocs())
        yticks = np.sort(self.yaxis.get_majorticklocs())
        threshold = self._get_key("grid.major.threshold")

        if fancy:
            r_ranges, x_ranges = self._compute_major_ranges(xticks, yticks, threshold)
        else:
            r_ranges = [None] * len(xticks)
            x_ranges = [None] * len(yticks)

        # Draw resistance circles with clipped reactance ranges
        for r, x_range in zip(xticks, r_ranges):
            if 0 <= r < SC_NEAR_INFINITY:
                self.plot_constant_resistance(r, range=x_range, **style)

        # Draw reactance circles with clipped resistance ranges
        for x, r_range in zip(yticks, x_ranges):
            if abs(x) < SC_NEAR_INFINITY:
                self.plot_constant_reactance(x, range=r_range, **style)

    def _draw_impedance_minor(self, fancy, **kwargs):
        """Draw minor impedance gridlines with nice spacing.

        Args:
            fancy: If True, use adaptive clipping based on Möbius distance
            **kwargs: Style overrides
        """
        style = self._get_grid_style("impedance", "minor", **kwargs)
        style.setdefault("marker", "")

        # Get impedance minor grid parameters
        real_divs = self._get_key("grid.Z.minor.real.divisions")
        imag_divs = self._get_key("grid.Z.minor.imag.divisions")
        threshold = self._get_key("grid.minor.threshold")

        # Compute minor tick values (domain-agnostic, returns ABSOLUTE values)
        xt_major = np.sort(self.xaxis.get_majorticklocs())
        yt_major = np.sort(self.yaxis.get_majorticklocs())

        if len(xt_major) == 0 or len(yt_major) == 0:
            return

        x_minor, y_minor = self._compute_minor_ticks(xt_major, yt_major, real_divs, imag_divs, threshold)

        if fancy:
            # First compute major ranges
            major_r_ranges, major_x_ranges = self._compute_major_ranges(
                xt_major, yt_major, self._get_key("grid.major.threshold")
            )
            # Then inherit those ranges for minor gridlines
            r_ranges, x_ranges = self._inherit_major_ranges(
                x_minor, y_minor, xt_major, yt_major, major_r_ranges, major_x_ranges
            )
        else:
            r_ranges = [None] * len(x_minor)
            x_ranges = [None] * len(y_minor)

        # Draw minor gridlines with inherited clipping from major
        for r, x_range in zip(x_minor, r_ranges):
            if 0 <= r < SC_NEAR_INFINITY:
                self.plot_constant_resistance(r, range=x_range, **style)

        for x, r_range in zip(y_minor, x_ranges):
            if abs(x) < SC_NEAR_INFINITY:
                self.plot_constant_reactance(x, range=r_range, **style)

    # ========== ADMITTANCE DRAWING ==========

    def _draw_admittance_major(self, fancy, **kwargs):
        """Draw major admittance gridlines.

        Args:
            fancy: If True, use adaptive clipping based on Möbius distance
            **kwargs: Style overrides
        """
        style = self._get_grid_style("admittance", "major", **kwargs)
        style.setdefault("marker", "")

        Z0 = self._get_key("axes.Z0")
        Y0 = 1 / Z0

        # Get major ticks and ranges (same tick computation, domain-agnostic)
        xticks = np.sort(self.xaxis.get_majorticklocs())
        yticks = np.sort(self.yaxis.get_majorticklocs())

        if fancy:
            threshold = self._get_key("grid.major.threshold")
            g_ranges, b_ranges = self._compute_major_ranges(xticks, yticks, threshold)
        else:
            g_ranges = [None] * len(xticks)
            b_ranges = [None] * len(yticks)

        # Draw conductance circles with clipped susceptance ranges
        for g, b_range in zip(xticks, g_ranges):
            if g > 1e-10:
                self.plot_constant_conductance(g, range=b_range, **style)

        # Draw susceptance circles with clipped conductance ranges
        for b, g_range in zip(yticks, b_ranges):
            self.plot_constant_susceptance(b, range=g_range, **style)

    def _draw_admittance_minor(self, fancy, **kwargs):
        """Draw minor admittance gridlines with nice spacing.

        Args:
            fancy: If True, use adaptive clipping based on Möbius distance
            **kwargs: Style overrides
        """
        style = self._get_grid_style("admittance", "minor", **kwargs)
        style.setdefault("marker", "")

        # Get admittance minor grid parameters
        real_divs = self._get_key("grid.Y.minor.real.divisions")
        imag_divs = self._get_key("grid.Y.minor.imag.divisions")
        threshold = self._get_key("grid.minor.threshold")

        # Compute minor tick values (domain-agnostic, returns ABSOLUTE values)
        xt_major = np.sort(self.xaxis.get_majorticklocs())
        yt_major = np.sort(self.yaxis.get_majorticklocs())

        if len(xt_major) == 0 or len(yt_major) == 0:
            return

        g_minor, b_minor = self._compute_minor_ticks(xt_major, yt_major, real_divs, imag_divs, threshold)

        if fancy:
            # First compute major ranges
            major_g_ranges, major_b_ranges = self._compute_major_ranges(
                xt_major, yt_major, self._get_key("grid.major.threshold")
            )
            # Then inherit those ranges for minor gridlines
            g_ranges, b_ranges = self._inherit_major_ranges(
                g_minor, b_minor, xt_major, yt_major, major_g_ranges, major_b_ranges
            )
        else:
            g_ranges = [None] * len(g_minor)
            b_ranges = [None] * len(b_minor)

        # Draw minor gridlines with inherited clipping from major
        for g, b_range in zip(g_minor, g_ranges):
            if g > 1e-10:
                self.plot_constant_conductance(g, range=b_range, **style)

        for b, g_range in zip(b_minor, b_ranges):
            self.plot_constant_susceptance(b, range=g_range, **style)

    # ========== HELPER METHODS (DOMAIN-AGNOSTIC) ==========

    def _compute_major_ranges(self, xticks, yticks, threshold):
        """Compute clipping ranges for major gridlines in fancy mode.

        This is domain-agnostic - it works on ABSOLUTE tick values and returns
        ABSOLUTE ranges. Works for both impedance and admittance.

        Args:
            xticks: Major real-axis tick values (sorted, ABSOLUTE)
            yticks: Major imaginary-axis tick values (sorted, ABSOLUTE)
            threshold: Möbius distance threshold (single value or tuple)

        Returns:
            tuple: (real_ranges, imag_ranges) where each is a list of ranges
                   Range is either None (full circle) or (min, max) tuple
        """
        thr_x, thr_y = self._split_threshold(threshold)

        # Check if yticks are symmetric (required for fancy)
        try:
            yticks_pos = self._check_fancy(yticks)
        except ValueError:
            # Fall back to no clipping
            return [None] * len(xticks), [None] * len(yticks)

        # Initialize: all circles drawn fully
        real_ranges = [None] * len(xticks)
        imag_ranges = [None] * len(yticks)

        # X=0 line (real axis) always drawn fully
        imag_ranges[len(yticks) // 2] = None  # Center index is X=0

        # Clip imaginary circles at real values
        tmp_yticks = yticks_pos.copy()
        for i, r in enumerate(xticks):
            if r >= SC_NEAR_INFINITY:
                continue
            k = 1
            while k < len(tmp_yticks):
                x0, x1 = tmp_yticks[k - 1 : k + 1]
                # Check Möbius distance
                if abs(self.moebius_z(r, x0) - self.moebius_z(r, x1)) < thr_x:
                    # Clip this imaginary circle at this real value
                    idx_pos = np.where(np.abs(yticks - x1) < SC_EPSILON)[0]
                    idx_neg = np.where(np.abs(yticks - (-x1)) < SC_EPSILON)[0]
                    if len(idx_pos) > 0:
                        imag_ranges[idx_pos[0]] = (0, r)
                    if len(idx_neg) > 0:
                        imag_ranges[idx_neg[0]] = (0, r)
                    tmp_yticks = np.delete(tmp_yticks, k)
                else:
                    k += 1

        # Clip real circles at imaginary values
        for i in range(1, len(yticks_pos)):
            x0, x1 = yticks_pos[i - 1 : i + 1]
            k = 1
            tmp_xticks = xticks.copy()
            while k < len(tmp_xticks):
                r0, r1 = tmp_xticks[k - 1 : k + 1]
                if abs(self.moebius_z(r0, x1) - self.moebius_z(r1, x1)) < thr_y:
                    # Clip this real circle at this imaginary range
                    idx = np.where(np.abs(xticks - r1) < SC_EPSILON)[0]
                    if len(idx) > 0:
                        real_ranges[idx[0]] = (-x0, x0)
                    tmp_xticks = np.delete(tmp_xticks, k)
                else:
                    k += 1

        return real_ranges, imag_ranges

    def _inherit_major_ranges(self, x_minor, y_minor, xt_major, yt_major, major_real_ranges, major_imag_ranges):
        """Compute clipping ranges for minor gridlines by inheriting from major gridlines.

        For each minor tick, find the nearest major ticks on either side and use
        the most restrictive clipping range (minimum of the max values).

        Args:
            x_minor: Minor real-axis tick values (sorted, ABSOLUTE)
            y_minor: Minor imaginary-axis tick values (sorted, ABSOLUTE)
            xt_major: Major real-axis tick values (sorted, ABSOLUTE)
            yt_major: Major imaginary-axis tick values (sorted, ABSOLUTE)
            major_real_ranges: Clipping ranges for major real gridlines
            major_imag_ranges: Clipping ranges for major imaginary gridlines

        Returns:
            tuple: (real_ranges, imag_ranges) for minor gridlines
        """
        real_ranges = []
        imag_ranges = []

        # For each minor real tick, inherit range from nearest major real ticks
        for r_minor in x_minor:
            # Find major ticks on either side
            idx_before = np.where(xt_major <= r_minor)[0]
            idx_after = np.where(xt_major >= r_minor)[0]

            range_before = major_real_ranges[idx_before[-1]] if len(idx_before) > 0 else None
            range_after = major_real_ranges[idx_after[0]] if len(idx_after) > 0 else None

            # Use the most restrictive range (smallest absolute max value)
            if range_before is None and range_after is None:
                real_ranges.append(None)
            elif range_before is None:
                real_ranges.append(range_after)
            elif range_after is None:
                real_ranges.append(range_before)
            else:
                # Both have ranges - take minimum of the max values
                # Ranges are symmetric: (-x, x) so we just compare the positive value
                max_before = abs(range_before[1]) if range_before[1] != 0 else abs(range_before[0])
                max_after = abs(range_after[1]) if range_after[1] != 0 else abs(range_after[0])
                min_max = min(max_before, max_after)
                real_ranges.append((-min_max, min_max))

        # For each minor imaginary tick, inherit range from nearest major imaginary ticks
        # Note: y_minor and yt_major contain both positive and negative values, already sorted
        for y_minor_val in y_minor:
            # Find major ticks on either side (in the SORTED array, not by absolute value)
            idx_before = np.where(yt_major <= y_minor_val)[0]
            idx_after = np.where(yt_major >= y_minor_val)[0]

            range_before = major_imag_ranges[idx_before[-1]] if len(idx_before) > 0 else None
            range_after = major_imag_ranges[idx_after[0]] if len(idx_after) > 0 else None

            # Use the most restrictive range (smallest max value)
            if range_before is None and range_after is None:
                imag_ranges.append(None)
            elif range_before is None:
                imag_ranges.append(range_after)
            elif range_after is None:
                imag_ranges.append(range_before)
            else:
                # Both have ranges - take minimum of the max values
                max_before = range_before[1]
                max_after = range_after[1]
                min_max = min(max_before, max_after)
                imag_ranges.append((0, min_max))

        return real_ranges, imag_ranges

    def _compute_minor_ticks(self, xt_major, yt_major, real_divs, imag_divs, threshold):
        """Compute minor tick positions with nice spacing.

        This is domain-agnostic - it works on ABSOLUTE major tick values and returns
        ABSOLUTE minor tick values. Works for both impedance and admittance.

        Args:
            xt_major: Major real-axis tick values (sorted, ABSOLUTE)
            yt_major: Major imaginary-axis tick values (sorted, ABSOLUTE)
            real_divs: Max divisions for real axis (or None for automatic)
            imag_divs: Max divisions for imaginary axis (or None for automatic)
            threshold: Integer that determines tick spacing

        Returns:
            tuple: (x_minor, y_minor) arrays of minor tick positions (ABSOLUTE)
        """
        # Check if yticks are symmetric
        try:
            yt_pos = self._check_fancy(yt_major)
        except ValueError:
            # Fall back to simple locator
            x_minor = self.xaxis.minor.locator()
            y_minor = self.yaxis.minor.locator()
            return x_minor, y_minor

        dividers = np.array([1, 2, 3, 5])
        thr_x, thr_y = self._split_threshold(threshold)

        if real_divs is None:
            x_dividers = list(dividers)
        else:
            x_dividers = [d for d in dividers if d <= real_divs] or [real_divs]

        if imag_divs is None:
            y_dividers = list(dividers)
        else:
            y_dividers = [d for d in dividers if d <= imag_divs] or [imag_divs]

        # Use midpoint for distance calculations
        ym = self.imag_interp1d([yt_pos[0], yt_pos[-1]], 2)[1]
        xm = self.real_interp1d([xt_major[0], xt_major[-1]], 2)[1]

        x_minor = []
        for x0, x1 in zip(xt_major[:-1], xt_major[1:]):
            if x1 >= SC_NEAR_INFINITY:
                continue
            div = choose_minor_divider(
                x0,
                x1,
                x_dividers,
                thr_x,
                map_func=lambda x, ym=ym: self.moebius_z(x, ym),
                max_divisions=real_divs,
            )
            if div > 1:
                x_minor.extend(np.linspace(x0, x1, div + 1)[1:-1])

        y_minor = []
        for y0, y1 in zip(yt_pos[:-1], yt_pos[1:]):
            div = choose_minor_divider(
                y0,
                y1,
                y_dividers,
                thr_y,
                map_func=lambda y, xm=xm: self.moebius_z(xm, y),
                max_divisions=imag_divs,
            )
            if div > 1:
                y_minor.extend(np.linspace(y0, y1, div + 1)[1:-1])

        x_minor = np.unique(np.round(np.asarray(x_minor), 7))
        y_minor = np.unique(np.round(np.asarray(y_minor), 7))

        # Add negative y values
        y_minor_full = []
        for y in y_minor:
            y_minor_full.append(y)
            if abs(y) > SC_EPSILON:
                y_minor_full.append(-y)

        return x_minor, np.array(y_minor_full)

    def _check_fancy(self, yticks):
        """Check if imaginary axis ticks are symmetric about zero."""
        len_y = (len(yticks) - 1) // 2
        if not (len(yticks) % 2 == 1 and (yticks[len_y:] + yticks[len_y::-1] < SC_EPSILON).all()):
            raise ValueError("Fancy grid requires zero-symmetric imaginary ticks")
        return yticks[len_y:]

    def _split_threshold(self, threshold):
        """Split threshold into x and y components and convert to Möbius distance.

        Args:
            threshold: Either a legacy numeric value (will be divided by 1000),
                      a tuple of legacy values, or a string like "2mm" for physical distance.
                      Can also be a tuple of strings like ("2mm", "1.5mm").

        Returns:
            tuple: (thr_x, thr_y) in Möbius distance units
        """

        def convert_threshold(thr):
            """Convert a single threshold value to Möbius distance."""
            if isinstance(thr, str) and thr.endswith("mm"):
                # Physical distance in millimeters
                mm_value = float(thr[:-2])
                # Convert mm to Möbius distance using current figure size
                return self._mm_to_moebius(mm_value)
            else:
                # Legacy numeric value - divide by 1000
                return thr / 1000

        if isinstance(threshold, tuple):
            thr_x = convert_threshold(threshold[0])
            thr_y = convert_threshold(threshold[1])
        else:
            thr_x = thr_y = convert_threshold(threshold)

        return (thr_x, thr_y)

    def _mm_to_moebius(self, mm):
        """Convert physical distance in mm to Möbius distance.

        Args:
            mm: Distance in millimeters

        Returns:
            float: Equivalent distance in Möbius space
        """
        # Get the bounding box of the axes in display coordinates (pixels)
        bbox = self.get_window_extent()

        # The Smith chart radius in display coordinates
        r = self._get_key("axes.radius")

        # Smith chart diameter in pixels (width of the circular chart area)
        # The chart goes from -r to +r in axes coordinates, which is 2*r
        # The axes coordinates go from 0 to 1, so we need the full axes width
        chart_diameter_pixels = bbox.width * 2 * r

        # Convert mm to inches (1 inch = 25.4 mm), then to pixels using DPI
        dpi = self.figure.dpi
        mm_to_pixels = dpi / 25.4
        distance_pixels = mm * mm_to_pixels

        # Convert pixels to fraction of chart diameter (this gives us data space distance)
        # In Möbius space, the chart has diameter 2 (from -1 to +1)
        moebius_distance = (distance_pixels / chart_diameter_pixels) * 2

        return moebius_distance

    def _get_grid_style(self, grid, level, **user_kwargs):
        """Get styling parameters for a grid."""
        if grid == "impedance":
            prefix = f"grid.Z.{level}"
        else:
            prefix = f"grid.Y.{level}"

        style = {}
        style["color"] = self._get_key(f"{prefix}.color")
        style["linestyle"] = self._get_key(f"{prefix}.linestyle")
        style["linewidth"] = self._get_key(f"{prefix}.linewidth")
        style["alpha"] = self._get_key(f"{prefix}.alpha")
        style["zorder"] = self._get_key("grid.zorder")

        if level == "minor":
            try:
                style["dashes"] = self._get_key(f"{prefix}.dashes")
                style["dash_capstyle"] = self._get_key(f"{prefix}.capstyle")
            except KeyError:
                pass

        style.update(user_kwargs)
        return style
