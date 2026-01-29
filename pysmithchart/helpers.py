"""Helper and utility methods for SmithAxes."""

import warnings
from numbers import Number
from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.spines import Spine

from pysmithchart import utils
from pysmithchart.constants import SC_TWICE_INFINITY
from pysmithchart.constants import R_DOMAIN, Z_DOMAIN, NORM_Z_DOMAIN, Y_DOMAIN, NORM_Y_DOMAIN

# Only export the mixin class, not imported symbols
__all__ = ["HelpersMixin"]


class HelpersMixin:
    """Mixin class providing helper methods for SmithAxes."""

    def _apply_domain_transform(self, x, y=None, domain=None, warn_s_parameter=True):
        """
        Apply domain transformation to convert input coordinates to normalized impedance.

        This unified function handles all domain transformations for plot(), scatter(),
        text(), and annotate() methods.

        Args:
            x: Real part(s), or complex impedance/admittance/S-parameter value(s).
               Can be scalar, array, or complex.
            y: Imaginary part(s). Ignored if x is complex. If None and x is real, y defaults to 0.
            domain: One of R_DOMAIN, Z_DOMAIN, NORM_Z_DOMAIN, Y_DOMAIN.
                    If None, uses plot.default.domain from scParams.
            warn_s_parameter: If True, warn when |S| > 1 (default: True).
                             Set to False to suppress warnings.

        Returns:
            tuple: (x_transformed, y_transformed) in normalized impedance space,
                   ready for plotting. Both are numpy arrays or scalars matching input type.

        Examples:
            >>> # Scalar impedance
            >>> x, y = ax._apply_domain_transform(50, 25, domain=Z_DOMAIN)

            >>> # Array of complex S-parameters
            >>> x, y = ax._apply_domain_transform([0.5+0.3j, -0.2-0.1j], domain=R_DOMAIN)

            >>> # Scalar complex admittance
            >>> x, y = ax._apply_domain_transform(0.02+0.01j, domain=Y_DOMAIN)
        """
        # Get default domain if not specified
        if domain is None:
            domain = self._get_key("plot.default.domain")

        # Validate domain
        domain = utils.validate_domain(domain)

        # Track if input was scalar for proper output format
        is_scalar_input = isinstance(x, Number)

        # Handle complex input vs separate x, y
        is_complex = False
        if isinstance(x, Number):
            is_complex = np.iscomplexobj(x)
        elif isinstance(x, Iterable):
            try:
                arr = np.asarray(x)
                is_complex = np.iscomplexobj(arr)
            except (ValueError, TypeError):
                pass

        if is_complex:
            # Complex input - convert to complex array
            if isinstance(x, Number):
                cdata = np.array([x])
            else:
                cdata = np.asarray(x)
        else:
            # Separate x and y inputs
            if isinstance(x, Number):
                x_arr = np.array([x])
            else:
                x_arr = np.asarray(x)

            if y is None:
                y_arr = np.zeros_like(x_arr)
            elif isinstance(y, Number):
                y_arr = np.array([y])
            else:
                y_arr = np.asarray(y)

            # Suppress warnings for inf/nan arithmetic (expected in edge cases)
            # Suppress warnings for inf/nan arithmetic (expected in edge cases)
            with np.errstate(invalid="ignore", divide="ignore"):
                cdata = x_arr + 1j * y_arr

        # Handle special cases: infinity maps to edges of Smith chart
        # Real(Z) = inf or |Z| = inf → right edge at (1, 0) in normalized space
        is_inf = np.isinf(np.real(cdata)) | np.isinf(np.abs(cdata))

        # Apply domain transformation (suppress inf/nan warnings)
        with np.errstate(invalid="ignore", divide="ignore"):
            if domain == R_DOMAIN:
                # S-parameters: Check magnitude and warn if > 1
                if warn_s_parameter:
                    s_magnitude = np.abs(cdata)
                    if np.any(s_magnitude > 1):
                        warnings.warn(
                            f"S-parameter magnitude |S| > 1 detected (max: {np.max(s_magnitude):.3f}). "
                            "Points outside the unit circle will not be visible on the Smith chart.",
                            UserWarning,
                        )
                # Apply inverse Möbius: z = (1 + S) / (1 - S)
                z = self.moebius_inv_z(cdata, normalize=True)

            elif domain == Z_DOMAIN:
                # Z-parameters in ohms: Normalize by Z₀
                z = cdata / self._get_key("axes.Z0")

            elif domain == NORM_Z_DOMAIN:
                # Already normalized, use as-is
                z = cdata

            elif domain == Y_DOMAIN:
                # Y-parameters in Siemens: Normalize by 1/Z₀
                y_norm = cdata * self._get_key("axes.Z0")
                z = np.conjugate(1.0 / y_norm)

            elif domain == NORM_Y_DOMAIN:
                # A-parameters: Already normalized, use as-is
                z = np.conjugate(1.0 / cdata)

            else:
                # Should never reach here due to validation above
                z = cdata

        # Map infinity values to right edge of Smith chart (1, 0)
        # This represents open circuit (infinite impedance)
        if np.any(is_inf):
            if np.isscalar(z):
                if is_inf:
                    z = 1.0 + 0j
            else:
                z = np.where(is_inf, 1.0 + 0j, z)

        # Convert to x, y coordinates
        x_transformed, y_transformed = utils.z_to_xy(z)

        # Return scalars if input was scalar
        if is_scalar_input:
            return float(x_transformed[0]), float(y_transformed[0])

        return x_transformed, y_transformed

    def _gen_axes_patch(self):
        """Generate the patch used to draw the Smith chart axes."""
        r = self._get_key("axes.radius") + 0.015
        if not self._get_key("grid.outer.enable"):
            # Return an invisible patch while preserving the circular clipping.
            return Circle((0.5, 0.5), r, edgecolor="none", facecolor="none")

        return Circle(
            (0.5, 0.5),
            r,
            edgecolor=self._get_key("grid.outer.color"),
            linestyle=self._get_key("grid.outer.linestyle"),
            linewidth=self._get_key("grid.outer.linewidth"),
            alpha=self._get_key("grid.outer.alpha"),
            facecolor="none",
        )

    def _gen_axes_spines(self):
        """Generate the spines for the circular Smith chart axes."""
        spine = Spine.circular_spine(self, (0.5, 0.5), self._get_key("axes.radius"))
        if self._get_key("grid.outer.enable"):
            spine.set_edgecolor(self._get_key("grid.outer.color"))
            spine.set_linestyle(self._get_key("grid.outer.linestyle"))
            spine.set_linewidth(self._get_key("grid.outer.linewidth"))
            spine.set_alpha(self._get_key("grid.outer.alpha"))
        else:
            spine.set_edgecolor("none")
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
