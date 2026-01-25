"""This module contains the implementation for formatters."""

from matplotlib.ticker import Formatter
from .constants import SC_EPSILON, SC_NEAR_INFINITY

__all__ = ["RealFormatter", "ImagFormatter"]


class RealFormatter(Formatter):
    """
    Formatter for the real axis of a Smith chart.

    This formatter formats tick values for the real axis by printing numbers
    as floats, removing trailing zeros and unnecessary decimal points.
    Special cases include returning an empty string '' for values near zero.

    Args:
        axes (SmithAxes):
            The parent `SmithAxes` instance associated with this formatter.

    Raises:
        AssertionError: If `axes` is not an instance of `SmithAxes`.

    Example:
        >>> formatter = RealFormatter(axes)
        >>> print(formatter(0.1))  # "0.1"
        >>> print(formatter(0))    # ""
    """

    def __init__(self, axes, *args, **kwargs):
        """
        Initialize the RealFormatter.

        Args:
            axes (SmithAxes):
                The parent `SmithAxes` instance.
            *args:
                Additional positional arguments passed to `Formatter`.
            **kwargs:
                Additional keyword arguments passed to `Formatter`.
        """
        super().__init__(*args, **kwargs)
        self.axes = axes

    def __call__(self, x, pos=None):
        """
        Format the given tick value.

        Args:
            x (float):
                The tick value to format.
            pos (int, optional):
                The position of the tick value (ignored in this formatter).

        Returns:
            str: The formatted tick value as a string, or `''` for values near zero.
        """
        if x < SC_EPSILON or x > SC_NEAR_INFINITY:
            return ""
        return ("%f" % x).rstrip("0").rstrip(".")


class ImagFormatter(RealFormatter):
    """
    Formatter for the imaginary axis of a Smith chart.

    This formatter formats tick values for the imaginary axis by printing numbers
    as floats, removing trailing zeros and unnecessary decimal points, and appending
    "j" to indicate imaginary values. Special cases include:

        - `''` (empty string) for negative infinity.
        - `'0'` for values near zero, ensuring `-0` is not displayed.

    Args:
        axes (SmithAxes):
            The parent `SmithAxes` instance associated with this formatter.
    """

    def __call__(self, x, pos=None):
        """
        Format the given tick value for the imaginary axis.

        Args:
            x (float):
                The tick value to format.
            pos (int, optional):
                The position of the tick value (ignored in this formatter).

        Returns:
            str: The formatted tick value as a string, with special handling for:
                - `''` (empty string) for negative infinity.
                - `'∞'` (UTF-8 infinity symbol) for positive infinity.
                - `'0'` for values near zero.
                - Appended "j" for imaginary values.
        """
        if x < -SC_NEAR_INFINITY:
            return ""
        if x > SC_NEAR_INFINITY:
            return "∞ "
        if abs(x) < SC_EPSILON:
            return "0"
        return ("%f" % x).rstrip("0").rstrip(".") + "j"
