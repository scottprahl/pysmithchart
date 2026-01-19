"""This module contains the implementation for locators."""

from matplotlib.ticker import AutoMinorLocator, Locator
import numpy as np

from .constants import SC_EPSILON, SC_INFINITY
from .utils import ang_to_c

__all__ = ["MajorXLocator", "MajorYLocator", "MinorLocator"]


class MajorXLocator(Locator):
    """
    A locator for the real (resistance/X) axis on the Smith chart.

    Attributes:
        axes (SmithAxes): The parent Smith chart axes to which this locator applies.
        steps (int): The maximum number of divisions for the real axis.
        precision (int): The maximum number of significant decimals for tick rounding.
        ticks (list): The calculated tick positions.
    """

    def __init__(self, axes, n, precision=None):
        """Initialize the MajorXLocator."""
        super().__init__()

        self.axes = axes
        self.steps = n
        self.precision = precision if precision is not None else axes._get_key("grid.locator.precision")

        if self.precision <= 0:
            raise ValueError("`precision` must be greater than 0.")

        self.ticks = None

    def __call__(self):
        """Compute or return cached tick values."""
        if self.ticks is None:
            self.ticks = self.tick_values(0, SC_INFINITY)
        return self.ticks

    def nice_round(self, num, down=True):
        """
        Round a number to a nicely rounded value based on precision.

        The rounding behavior adapts dynamically to ensure ticks are visually
        consistent across different scales.

        Args:
            num (float): The number to round.
            down (bool, optional): Whether to round down. Defaults to `True`.

        Returns: A nicely rounded value.
        """
        exp = np.ceil(np.log10(np.abs(num) + SC_EPSILON))
        if exp < 1:
            exp += 1
        norm = 10 ** (-(exp - self.precision))
        num_normed = num * norm
        if num_normed < 3.3:
            norm *= 2
        elif num_normed > 50:
            norm /= 10
        if not 1 < num_normed % 10 < 9:
            if abs(num_normed % 10 - 1) < SC_EPSILON:
                num -= 0.5 / norm
            f_round = np.round
        else:
            f_round = np.floor if down else np.ceil
        return f_round(np.round(num * norm, 1)) / norm

    def tick_values(self, vmin, vmax):
        """
        Compute the tick values for the real axis.

        Includes the center value as a mandatory tick and dynamically
        adjusts spacing to ensure evenly distributed ticks.

        Args:
            vmin (float): The minimum value of the axis.
            vmax (float): The maximum value of the axis.

        Returns: he computed tick values for the real axis.
        """
        tmin, tmax = (self.transform(vmin), self.transform(vmax))
        mean = self.transform(self.nice_round(self.invert(0.5 * (tmin + tmax))))
        result = [tmin, tmax, mean]
        d0 = abs(tmin - tmax) / (self.steps + 1)
        for sgn, side, end in [[1, False, tmax], [-1, True, tmin]]:
            d, d0 = (d0, None)
            last = mean
            while True:
                new = last + d * sgn
                if self.out_of_range(new) or abs(end - new) < d / 2:
                    break
                new = self.transform(self.nice_round(self.invert(new), side))
                d = abs(new - last)
                if d0 is None:
                    d0 = d
                last = new
                result.append(last)
        return np.sort(self.invert(np.array(result)))

    def out_of_range(self, x):
        """Check if a value is outside the valid range for the real axis."""
        return abs(x) > 1

    def transform(self, x):
        """Apply the Möbius transformation to a value."""
        return self.axes.moebius_z(x)

    def invert(self, x):
        """Apply the inverse Möbius transformation to a value."""
        return self.axes.moebius_inv_z(x)


class MajorYLocator(MajorXLocator):
    """
    Locator for the imaginary (reactance/Y) axis of a Smith chart.

    This class generates evenly spaced, nicely rounded tick values for the imaginary
    axis of a Smith chart. It extends the `MajorXLocator` class and adapts it for
    handling reactance values.
    """

    def __init__(self, axes, n, precision=None):
        """Initialize the MajorYLocator."""
        super().__init__(axes, n // 2, precision)

    def __call__(self):
        """Compute or return cached tick values for the imaginary axis."""
        if self.ticks is None:
            tmp = self.tick_values(0, SC_INFINITY)
            self.ticks = np.concatenate((-tmp[:0:-1], tmp))
        return self.ticks

    def out_of_range(self, x):
        """Check if a value is outside the valid range for the imaginary axis."""
        return not 0 <= x <= np.pi

    def transform(self, x):
        """Apply the Möbius transformation to a value on the imaginary axis."""
        return np.pi - np.angle(self.axes.moebius_z(x * 1j))

    def invert(self, x):
        """Apply the inverse Möbius transformation to a value."""
        return np.imag(-self.axes.moebius_inv_z(ang_to_c(np.pi + np.array(x))))


class MinorLocator(AutoMinorLocator):
    """
    Minor tick locator for Smith chart axes.

    This locator generates evenly spaced minor ticks between major tick values.
    It supports both fixed and automatic modes:

    - Fixed mode (n=int): Uses the same number of divisions for all intervals
    - Automatic mode (n=None): Adapts divisions per interval based on spacing

    Attributes:
        ndivs (int or None): The number of divisions between major tick intervals,
                            or None for automatic mode.
        _ticks (numpy.ndarray or None): Cached array of computed minor tick values.

    Args:
        n (int or None, optional):
            The number of divisions between major tick values.
            - If an integer: use exactly that many divisions for all intervals.
            - If None: automatically compute divisions per interval based on spacing.
            Defaults to None (automatic).
    """

    def __init__(self, n=None):
        """
        Initialize the MinorLocator.

        Args:
            n (int or None, optional):
                The number of divisions between major tick values.
                - If an integer: use exactly that many divisions for all intervals.
                - If None: automatically compute divisions per interval based on spacing.
                Must be a positive integer if provided. Defaults to None (automatic).
        """
        if n is not None:
            assert isinstance(n, int) and n > 0
        super().__init__(n=n)
        self._ticks = None

    def tick_values(self, vmin, vmax):
        """
        Call parent to find tick values.

        This doesn't get used.
        """

    def __call__(self):
        """Compute and return minor tick positions.

        Ticks are recomputed on every call to ensure they stay synchronized
        with the current ndivs setting and major tick locations.

        If ndivs is None (automatic mode), divisions are computed adaptively
        per interval to maintain uniform spacing within each major interval.
        """
        locs = self.axis.get_majorticklocs()

        if self.ndivs is None:
            # Automatic mode: compute divisions per interval based on span
            minor_ticks = []
            for p0, p1 in zip(locs[:-1], locs[1:]):
                span = abs(p1 - p0)
                # Choose divisions based on span magnitude
                # Smaller spans get more divisions for consistent visual density
                if span < 0.1:
                    n = 5
                elif span < 0.5:
                    n = 4
                elif span < 2.0:
                    n = 3
                else:
                    n = 2
                # Generate uniform divisions within this interval
                interval_ticks = np.linspace(p0, p1, n + 1)[1:-1]
                minor_ticks.extend(interval_ticks)
            self._ticks = np.array(minor_ticks)
        else:
            # Fixed mode: use same divisions for all intervals
            self._ticks = np.hstack([np.linspace(p0, p1, self.ndivs + 1)[1:-1] for p0, p1 in zip(locs[:-1], locs[1:])])

        return self._ticks

    def get_ticklocs(self):
        """Return the computed minor tick locations without filtering."""
        return self._ticks
