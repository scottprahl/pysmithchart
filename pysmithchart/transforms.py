"""Transform-related functionality for SmithAxes."""

import numpy as np
from matplotlib.transforms import Affine2D, BboxTransformTo

from pysmithchart.moebius_transform import MoebiusTransform
from pysmithchart.constants import SC_TWICE_INFINITY


class TransformMixin:
    """Mixin class providing transform-related methods for SmithAxes."""

    def _should_transform_coordinates(self, coord_system):
        """
        Determine if coordinates should be transformed based on the coordinate system.

        This is a unified helper to check whether we should apply Smith chart transformations.
        Only 'data' coordinates should be transformed; all other coordinate systems
        (axes, figure, etc.) should be left alone.

        Args:
            coord_system (str or Transform): The coordinate system specification.
                Can be 'data', 'axes', 'figure', or a Transform object.

        Returns:
            bool: True if coordinates should be transformed, False otherwise.
        """
        # If it's a string, check if it's 'data'
        if isinstance(coord_system, str):
            return coord_system == "data"

        # If it's a Transform object, check if it's transData
        # (or None, which defaults to transData)
        if coord_system is None:
            return True

        # Check if it's the data transform
        return coord_system is self.transData

    def _transform_coordinates(self, x, y, domain):
        """
        Transform coordinates from specified domain to impedance space.

        This method delegates to the unified _apply_domain_transform() function
        for consistent coordinate transformation across all plotting methods.

        Args:
            x (float or array): Real part of coordinate(s).
            y (float or array): Imaginary part of coordinate(s).
            domain (str): Coordinate type (Z_DOMAIN, Y_DOMAIN, R_DOMAIN, NORM_Z_DOMAIN).

        Returns:
            tuple: (x_impedance, y_impedance) in impedance space.
        """
        # Delegate to unified transformation function
        # Suppress S-parameter warnings for text/annotate (warn_s_parameter=False)
        return self._apply_domain_transform(x, y, domain=domain, warn_s_parameter=False)

    def _set_lim_and_transforms(self):
        """
        Configure the axis limits and transformation pipelines for the chart.

        This method defines and applies a series of transformations to map data
        space, Möbius space, axes space, and drawing space.

        Transformations:
            - `transProjection`: Maps data space to Möbius space using a Möbius transformation.
            - `transAffine`: Scales and translates Möbius space to fit axes space.
            - `transDataToAxes`: Combines `transProjection` and `transAffine` to map data space to axes space.
            - `transAxes`: Maps axes space to drawing space using the bounding box (`bbox`).
            - `transMoebius`: Combines `transAffine` and `transAxes` to map Möbius space to drawing space.
            - `transData`: Combines `transProjection` and `transMoebius` as data-to-drawing-space transform.

        X-axis transformations:
            - `_xaxis_pretransform`: Scales and centers the x-axis based on axis limits.
            - `_xaxis_transform`: Combines `_xaxis_pretransform` and `transData` for full x-axis mapping.
            - `_xaxis_text1_transform`: Adjusts x-axis label positions.

        Y-axis transformations:
            - `_yaxis_stretch`: Scales the y-axis based on axis limits.
            - `_yaxis_correction`: Applies additional translation to the y-axis for label adjustments.
            - `_yaxis_transform`: Combines `_yaxis_stretch` and `transData` for full y-axis mapping.
            - `_yaxis_text1_transform`: Combines `_yaxis_stretch` and `_yaxis_correction` for y label position
        """
        r = self._get_key("axes.radius")
        self.transProjection = MoebiusTransform(self)
        self.transAffine = Affine2D().scale(r, r).translate(0.5, 0.5)
        self.transDataToAxes = self.transProjection + self.transAffine
        self.transAxes = BboxTransformTo(self.bbox)
        self.transMoebius = self.transAffine + self.transAxes
        self.transData = self.transProjection + self.transMoebius
        self._xaxis_pretransform = Affine2D().scale(1, 2 * SC_TWICE_INFINITY).translate(0, -SC_TWICE_INFINITY)
        self._xaxis_transform = self._xaxis_pretransform + self.transData
        self._xaxis_text1_transform = Affine2D().scale(1.0, 0.0) + self.transData
        self._yaxis_stretch = Affine2D().scale(SC_TWICE_INFINITY, 1.0)
        self._yaxis_correction = self.transData + Affine2D().translate(*self._get_key("axes.ylabel.correction")[:2])
        self._yaxis_transform = self._yaxis_stretch + self.transData
        self._yaxis_text1_transform = self._yaxis_stretch + self._yaxis_correction

    def get_xaxis_transform(self, which="grid"):
        """
        Get the transformation for the x-axis.

        Args:
            which (str): Specifies which gridlines the transformation is for.
                Defaults to "grid".

        Returns:
            Transform: The transformation object for the x-axis.
        """
        assert which in ["tick1", "tick2", "grid"]
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad_points):
        """
        Get the transformation for text on the first x-axis.

        Args:
            pad_points (float): Padding in points.

        Returns:
            tuple: A tuple containing the transformation and text alignment information.
        """
        return self._xaxis_text1_transform, "center", "center"

    def get_yaxis_transform(self, which="grid"):
        """
        Get the transformation for the y-axis.

        Args:
            which (str): Specifies which gridlines the transformation is for.
                Defaults to "grid".

        Returns:
            Transform: The transformation object for the y-axis.
        """
        assert which in ["tick1", "tick2", "grid"]
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad_points):
        """
        Get the transformation for text on the first y-axis.

        Args:
            pad_points (float): Padding in points.

        Returns:
            tuple: A tuple containing the transformation and text alignment information.
        """
        from pysmithchart.polar_transform import PolarTranslate

        if hasattr(self, "yaxis") and len(self.yaxis.majorTicks) > 0:
            font_size = self.yaxis.majorTicks[0].label1.get_size()
        else:
            font_size = self._get_key("font.size")
        offset = self._get_key("axes.ylabel.correction")[2]
        return (
            self._yaxis_text1_transform + PolarTranslate(self, pad=pad_points + offset, font_size=font_size),
            "center",
            "center",
        )

    def moebius_z(self, *args, normalize=None):
        """
        Apply the Möbius transformation to impedance values.

        Converts impedance values (Z-parameters) to reflection coefficients (S-parameters)
        using the Möbius transformation. Handles both single values and arrays.

        Args:
            *args: Either a single complex number/array or separate real and imaginary parts.
            normalize (bool, optional): Whether to apply normalization. If None, uses
                the axes' normalization setting.

        Returns:
            complex or ndarray: The transformed value(s) in S-parameter space.

        Examples:
            >>> z = 50 + 50j  # Impedance
            >>> s = ax.moebius_z(z)  # Convert to S-parameter
        """
        from pysmithchart import utils

        if normalize is None:
            normalize = True

        # Parse arguments to get z
        if len(args) == 1:
            z = args[0]
        elif len(args) == 2:
            z = args[0] + 1j * args[1]
        else:
            raise ValueError("Invalid number of arguments")

        # Determine normalization value
        z0 = self._get_key("axes.Z0")
        norm = 1 if normalize else z0

        # Call canonical implementation from utils
        return utils.moebius_transform(z, norm=norm)

    def moebius_inv_z(self, *args, normalize=None):
        """
        Apply the inverse Möbius transformation to reflection coefficients.

        Converts reflection coefficients (S-parameters) back to impedance values
        (Z-parameters) using the inverse Möbius transformation. Handles both single
        values and arrays.

        Args:
            *args: Either a single complex number/array or separate real and imaginary parts.
            normalize (bool, optional): Whether to apply normalization. If None, uses
                the axes' normalization setting.

        Returns:
            complex or ndarray: The transformed value(s) in Z-parameter space.

        Examples:
            >>> s = 0.2 + 0.3j  # Reflection coefficient
            >>> z = ax.moebius_inv_z(s)  # Convert to impedance
        """
        from pysmithchart import utils

        if normalize is None:
            normalize = True

        # Parse arguments to get s
        if len(args) == 1:
            s = args[0]
        elif len(args) == 2:
            s = args[0] + 1j * args[1]
        else:
            raise ValueError("Invalid number of arguments")

        # Determine normalization value
        z0 = self._get_key("axes.Z0")
        norm = 1 if normalize else z0

        # Call canonical implementation from utils
        return utils.moebius_inverse_transform(s, norm=norm)

    def real_interp1d(self, x, steps):
        """
        Interpolate a vector of real values with evenly spaced points.

        This method interpolates the given real values such that, after applying a Möbius
        transformation with an imaginary part of 0, the resulting points are evenly spaced.

        The result is mapped back to the original space using the inverse Möbius transformation.

        Args:
            x (iterable): Real values to interpolate.
            steps (int): Interpolation steps between two points.

        Returns: Interpolated real values.
        """
        from matplotlib.cbook import simple_linear_interpolation as linear_interpolation

        return self.moebius_inv_z(linear_interpolation(self.moebius_z(np.array(x)), steps))

    def imag_interp1d(self, y, steps):
        """
        Interpolate a vector of imaginary values with evenly spaced points.

        This method interpolates the given imaginary values such that, after applying
        a Möbius transformation with a real part of 0, the resulting points are evenly spaced.

        The result is mapped back to the original space using the inverse Möbius transformation.

        Args:
            y (iterable): Imaginary values to interpolate.
            steps (int): Interpolation steps between two points.

        Returns: Interpolated imaginary values.
        """
        from matplotlib.cbook import simple_linear_interpolation as linear_interpolation
        from pysmithchart import utils

        angs = np.angle(self.moebius_z(np.array(y) * 1j)) % (2 * np.pi)
        i_angs = linear_interpolation(angs, steps)
        return np.imag(self.moebius_inv_z(utils.ang_to_c(i_angs)))
