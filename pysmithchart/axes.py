"""Main SmithAxes class assembled from modular components.

This module provides the main SmithAxes class which extends matplotlib's Axes
to create Smith Charts. The implementation is split across multiple mixin classes
for better organization and maintainability.
"""

from matplotlib.axes import Axes

from pysmithchart.core import AxesCore
from pysmithchart.transforms import TransformMixin
from pysmithchart.grid import GridMixin
from pysmithchart.plotting import PlottingMixin
from pysmithchart.helpers import HelpersMixin

__all__ = ["SmithAxes"]


class SmithAxes(AxesCore, TransformMixin, GridMixin, PlottingMixin, HelpersMixin, Axes):
    """
    A subclass of :class:`matplotlib.axes.Axes` specialized for rendering Smith Charts.

    This class implements a fully automatic Smith Chart with support for impedance
    normalization, custom grid configurations, and flexible marker handling. Default
    parameters (e.g., grid settings, marker styles, and plot defaults) are defined in
    :mod:`pysmithchart.constants`.

    The implementation is organized into several mixin classes:

    - AxesCore: Initialization and configuration
    - TransformMixin: Coordinate transformations (Möbius, etc.)
    - GridMixin: Grid drawing functionality
    - PlottingMixin: Plotting, text, annotation methods
    - HelpersMixin: Utility and helper methods

    Note:
        Parameter changes (such as grid updates) may not take effect immediately.
        To reset the chart, use the :meth:`clear` method.

    Examples:
        Create a simple Smith chart:

        >>> import matplotlib.pyplot as plt
        >>> import pysmithchart
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(1, 1, 1, projection='smith')
        >>> ax.plot([0.5+0.5j, 0.3+0.8j])
        >>> plt.show()
    """

    # The name attribute identifies this projection
    name = "smith"
