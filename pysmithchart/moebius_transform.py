"""This module contains the implementation for moebius transform."""

from collections.abc import Iterable

from matplotlib.patches import Arc
from matplotlib.path import Path
from matplotlib.transforms import Transform, Affine2D
import numpy as np

from .utils import z_to_xy
from .constants import SC_INFINITY

__all__ = ["MoebiusTransform", "InvertedMoebiusTransform"]


class BaseMoebiusTransform(Transform):
    """Abstract class to work around circular imports."""

    #: The number of input dimensions (always 2).
    input_dims = 2
    #: The number of output dimensions (always 2).
    output_dims = 2
    #: Whether the transform is separable (always False).
    is_separable = False

    def __init__(self, axes, *args, **kwargs):
        """Initialize the inverse polar translation transformation."""
        super().__init__(*args, **kwargs)
        self.axes = axes

    def inverted(self):
        """To be implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class MoebiusTransform(BaseMoebiusTransform):
    """
    Transform points and paths into Smith chart data space.

    This class implements the Möbius transformation required to map Cartesian
    coordinates to the Smith chart's complex data space. It supports point transformations
    and path transformations for visualizing data on the Smith chart.
    """

    def transform_non_affine(self, values):
        """
        Apply the non-affine Möbius transformation to input data.

        Args:
            values (array-like): The input data to transform. Can be a single point
                (x, y) or an iterable of points.

        Returns:
            list or tuple: The transformed points in Smith chart data space.
        """

        def moebius_xy(_xy):
            return z_to_xy(self.axes.moebius_z(*_xy))

        if isinstance(values[0], Iterable):
            return list(map(moebius_xy, values))
        return moebius_xy(values)

    def transform_path_non_affine(self, path):
        """
        Transform a path using the Möbius transformation.

        This uses path._interpolation identify if the is a x or y gridline.

        This method generates arcs based on the Möbius transformation.

        The method supports linear interpolation (linetype=1) for non-gridline paths.

        Args:
            path (matplotlib.path.Path): The input path to transform.

        Returns:
            matplotlib.path.Path: The transformed path in Smith chart data space.
        """
        vertices = path.vertices
        codes = path.codes
        linetype = path._interpolation_steps  # pylint: disable=protected-access
        if linetype in ["x_gridline", "y_gridline"]:
            assert len(vertices) == 2
            x, y = np.array(list(zip(*vertices)))
            z = self.axes.moebius_z(x, y)
            if linetype == "x_gridline":
                assert x[0] == x[1]
                zm = 0.5 * (1 + self.axes.moebius_z(x[0]))
            else:
                assert y[0] == y[1]
                if self.axes._normalize:  # pylint: disable=protected-access
                    scale = 1j
                else:
                    scale = 1j * self.axes._get_key("axes.impedance")  # pylint: disable=protected-access
                zm = 1 + scale / y[0]
            d = 2 * abs(zm - 1)

            # For y_gridlines, calculate the full arc span
            # The arc should go from the leftmost point to the rightmost point on the circle
            if linetype == "y_gridline":
                # The circle is centered at zm with radius d/2
                # It intersects the real axis (Im=0) at two points
                # Left intersection: zm - d/2, Right intersection: zm + d/2
                # But we want angles measured from center zm
                # The full arc should span from one side to the other
                # For a reactance circle, this is typically from -180° to 0° or 0° to 180°
                # depending on sign, but let's calculate based on actual geometry

                # The arc goes from the leftmost to rightmost point
                # In S-parameter space, the leftmost point is at x=0 (which maps to s=-1)
                # Calculate where the line at x=0 intersects our circle
                z_left = self.axes.moebius_z(0, y[0])  # Left edge
                # For right edge, we want to go as far as possible
                # The circle might complete before reaching infinity
                # Calculate the rightmost intersection: where the circle intersects Re(z)=1 (right edge)
                # or where it completes (Im(z)=0)

                # Simpler approach: use a very large x value to approximate infinity
                z_right = self.axes.moebius_z(SC_INFINITY, y[0])

                ang0, ang1 = np.angle([z_left - zm, z_right - zm], deg=True) % 360
            else:
                # For x_gridlines, use the transformed endpoints as before
                ang0, ang1 = np.angle(z - zm, deg=True) % 360

            reverse = ang0 > ang1
            if reverse:
                ang0, ang1 = (ang1, ang0)
            arc = Arc(
                z_to_xy(zm),
                d,
                d,
                theta1=ang0,
                theta2=ang1,
                transform=Affine2D(),
            )
            arc._path = Path.arc(ang0, ang1)  # pylint: disable=protected-access
            arc_path = arc.get_patch_transform().transform_path(arc.get_path())
            if reverse:
                new_vertices = arc_path.vertices[::-1]
            else:
                new_vertices = arc_path.vertices
            new_codes = arc_path.codes
        elif linetype == 1:
            new_vertices = self.transform_non_affine(vertices)
            new_codes = codes
        else:
            raise NotImplementedError("Value for 'path_interpolation' cannot be interpreted.")
        return Path(new_vertices, new_codes)

    def inverted(self):
        """
        Return the inverse Möbius transformation.

        This method provides the inverse transformation for mapping points or paths
        back from the Smith chart's data space to Cartesian coordinates.

        Returns:
            SmithAxes.InvertedMoebiusTransform: The inverted transformation instance.

        Example:
            >>> transform = MoebiusTransform(axes)
            >>> inverted_transform = transform.inverted()
        """
        return InvertedMoebiusTransform(self.axes)


class InvertedMoebiusTransform(BaseMoebiusTransform):
    """
    Perform the inverse transformation for points and paths in Smith chart data space.

    This class implements the inverse Möbius transformation, which maps points and paths
    from the Smith chart's data space back to Cartesian coordinates. It is typically used
    as the inverse of the `MoebiusTransform` class.
    """

    def transform_non_affine(self, values):
        """
        Apply the non-affine inverse Möbius transformation to input data.

        Args:
            values (array-like):
                The input data to transform, given as a list of (x, y) points.

        Returns:
            list: The transformed points, mapped from the Smith chart data space
            back to Cartesian coordinates.
        """

        def _moebius_inv_xy(_xy):
            return z_to_xy(self.axes.moebius_inv_z(*_xy))

        return list(map(_moebius_inv_xy, values))

    def inverted(self):
        """
        Return the forward Möbius transformation.

        This method provides the forward Möbius transformation to map points
        from Cartesian coordinates to Smith chart data space.

        Returns:
            SmithAxes.MoebiusTransform: The forward Möbius transformation instance.

        Example:
            >>> inverted_transform = InvertedMoebiusTransform(axes)
            >>> forward_transform = inverted_transform.inverted()
        """
        return MoebiusTransform(self.axes)
