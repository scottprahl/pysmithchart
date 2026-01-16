"""Helper and utility methods for SmithAxes."""

from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.spines import Spine

from pysmithchart.constants import SC_TWICE_INFINITY


class HelpersMixin:
    """Mixin class providing helper methods for SmithAxes."""

    def _gen_axes_patch(self):
        """Generate the patch used to draw the Smith chart axes."""
        r = self._get_key("axes.radius") + 0.015
        c = self._get_key("grid.major.color.x")
        return Circle((0.5, 0.5), r, edgecolor=c)

    def _gen_axes_spines(self, locations=None, offset=0.0, units="inches"):
        """Generate the spines for the circular Smith chart axes."""
        spine = Spine.circular_spine(self, (0.5, 0.5), self._get_key("axes.radius"))
        spine.set_edgecolor(self._get_key("grid.major.color.x"))
        return {self.name: spine}

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
