"""
Pytest tests for SmithAxes scatter() method with domain support.

This test suite validates the scatter plotting functionality with different
datatypes and input formats on Smith charts.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PathCollection

matplotlib.use("Agg")

import sys

sys.path.insert(0, "/home/claude")

from pysmithchart import IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, REFLECTANCE_DOMAIN, ABSOLUTE_DOMAIN


class TestSmithAxesScatter:
    """Test suite for SmithAxes.scatter() method."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_scatter_method_exists(self, smith_axes):
        """Test that scatter() method exists."""
        assert hasattr(smith_axes, "scatter")
        assert callable(smith_axes.scatter)

    def test_scatter_returns_path_collection(self, smith_axes):
        """Test that scatter returns a PathCollection."""
        collection = smith_axes.scatter([50 + 25j], domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)

    def test_scatter_complex_input_REFLECTION_DOMAIN(self, smith_axes):
        """Test scatter with complex impedance values."""
        ZL = [30 + 30j, 50 + 50j, 100 + 100j]
        collection = smith_axes.scatter(ZL, s=100, domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == len(ZL)

    def test_scatter_separate_xy_REFLECTION_DOMAIN(self, smith_axes):
        """Test scatter with separate x, y arrays."""
        x_vals = [30, 50, 100]
        y_vals = [30, 50, 100]
        collection = smith_axes.scatter(x_vals, y_vals, s=100, domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == len(x_vals)

    def test_scatter_single_point(self, smith_axes):
        """Test scatter with a single point."""
        collection = smith_axes.scatter(50 + 25j, s=150, domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == 1

    def test_scatter_REFLECTION_DOMAIN2(self, smith_axes):
        """Test scatter with S-parameters."""
        S = [0.5 + 0.3j, -0.2 - 0.1j, 0.1 + 0.4j]
        collection = smith_axes.scatter(S, s=100, c="blue", domain=REFLECTANCE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == len(S)

    def test_scatter_ADMITTANCE_DOMAIN(self, smith_axes):
        """Test scatter with Y-parameters (admittance)."""
        Y = [0.02 + 0.01j, 0.01 - 0.01j]
        collection = smith_axes.scatter(Y, s=100, domain=ADMITTANCE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == len(Y)

    def test_scatter_ABSOLUTE_DOMAIN(self, smith_axes):
        """Test scatter with A-parameters (absolute coordinates)."""
        A = [1.0 + 0.5j, 0.5 + 0.3j]
        collection = smith_axes.scatter(A, s=100, domain=ABSOLUTE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == len(A)

    def test_scatter_default_datatype(self, smith_axes):
        """Test that default domain is used when not specified."""
        collection = smith_axes.scatter([50 + 25j])
        assert isinstance(collection, PathCollection)

    def test_scatter_invalid_datatype(self, smith_axes):
        """Test that invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            smith_axes.scatter([50 + 25j], domain="INVALID")

    def test_scatter_with_color(self, smith_axes):
        """Test scatter with color specification."""
        collection = smith_axes.scatter([50 + 25j, 75 + 50j], s=100, c="red", domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)

    def test_scatter_with_colormap(self, smith_axes):
        """Test scatter with colormap."""
        points = [30 + 30j, 50 + 50j, 100 + 100j]
        colors = [0, 0.5, 1.0]
        collection = smith_axes.scatter(points, s=100, c=colors, cmap="viridis", domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)

    def test_scatter_with_varying_sizes(self, smith_axes):
        """Test scatter with varying marker sizes."""
        points = [30 + 30j, 50 + 50j, 100 + 100j]
        sizes = [50, 100, 150]
        collection = smith_axes.scatter(points, s=sizes, domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)

    def test_scatter_with_alpha(self, smith_axes):
        """Test scatter with alpha transparency."""
        collection = smith_axes.scatter([50 + 25j], s=150, alpha=0.5, domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)

    def test_scatter_with_edgecolors(self, smith_axes):
        """Test scatter with edge colors."""
        collection = smith_axes.scatter(
            [50 + 25j, 75 + 50j], s=100, c="blue", edgecolors="red", linewidths=2, domain=IMPEDANCE_DOMAIN
        )
        assert isinstance(collection, PathCollection)

    def test_scatter_with_marker_style(self, smith_axes):
        """Test scatter with different marker styles."""
        markers = ["o", "s", "^", "D", "*"]
        for i, marker in enumerate(markers):
            collection = smith_axes.scatter([50 + i * 10j], s=100, marker=marker, domain=IMPEDANCE_DOMAIN)
            assert isinstance(collection, PathCollection)

    def test_scatter_REFLECTION_DOMAIN2_warning_outside_chart(self, smith_axes):
        """Test warning when S-parameter magnitude > 1."""
        with pytest.warns(UserWarning, match="S-parameter magnitude"):
            smith_axes.scatter([1.5 + 0j], domain=REFLECTANCE_DOMAIN)

    def test_scatter_multiple_calls(self, smith_axes):
        """Test multiple scatter calls on same axes."""
        c1 = smith_axes.scatter([30 + 30j], s=100, c="red", domain=IMPEDANCE_DOMAIN)
        c2 = smith_axes.scatter([50 + 50j], s=100, c="blue", domain=IMPEDANCE_DOMAIN)
        c3 = smith_axes.scatter([100 + 100j], s=100, c="green", domain=IMPEDANCE_DOMAIN)

        assert isinstance(c1, PathCollection)
        assert isinstance(c2, PathCollection)
        assert isinstance(c3, PathCollection)

    def test_scatter_with_plot(self, smith_axes):
        """Test scatter used together with plot."""
        smith_axes.plot([30 + 30j, 100 + 100j], "b-", domain=IMPEDANCE_DOMAIN)
        collection = smith_axes.scatter([30 + 30j, 50 + 50j, 100 + 100j], s=100, c="red", domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)

    def test_scatter_matched_load(self, smith_axes):
        """Test scatter at matched load (center of chart)."""
        collection = smith_axes.scatter([50 + 0j], s=200, c="green", marker="*", domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)

    def test_scatter_numpy_array_input(self, smith_axes):
        """Test scatter with numpy array input."""
        Z_array = np.array([30 + 30j, 50 + 50j, 100 + 100j])
        collection = smith_axes.scatter(Z_array, s=100, domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == len(Z_array)

    def test_scatter_real_only_input(self, smith_axes):
        """Test scatter with real-only input (imaginary = 0)."""
        x_vals = [25, 50, 75, 100]
        collection = smith_axes.scatter(x_vals, s=100, domain=IMPEDANCE_DOMAIN)
        assert isinstance(collection, PathCollection)
        assert len(collection.get_offsets()) == len(x_vals)

    def test_scatter_zorder(self, smith_axes):
        """Test that scatter respects zorder."""
        c1 = smith_axes.scatter([30 + 30j], s=100, domain=IMPEDANCE_DOMAIN)
        c2 = smith_axes.scatter([50 + 50j], s=100, domain=IMPEDANCE_DOMAIN)

        assert c2.get_zorder() > c1.get_zorder()

    @pytest.mark.parametrize("domain", [IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, REFLECTANCE_DOMAIN, ABSOLUTE_DOMAIN])
    def test_scatter_all_datatypes_parametrized(self, smith_axes, domain):
        """Parametrized test for all datatypes."""
        # REFLECTANCE_DOMAIN with large impedance values will trigger warning
        if domain == REFLECTANCE_DOMAIN:
            with pytest.warns(UserWarning, match="S-parameter magnitude"):
                collection = smith_axes.scatter([50 + 25j], s=100, domain=domain)
        else:
            collection = smith_axes.scatter([50 + 25j], s=100, domain=domain)
        assert isinstance(collection, PathCollection)


class TestScatterTransformations:
    """Test coordinate transformations in scatter method."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_scatter_z_to_s_consistency(self, smith_axes):
        """Test that Z and S parameters place points consistently."""
        Z0 = 50
        ZL = 75 + 50j

        Gamma = (ZL - Z0) / (ZL + Z0)

        c1 = smith_axes.scatter([ZL], s=100, c="red", domain=IMPEDANCE_DOMAIN)
        c2 = smith_axes.scatter([Gamma], s=50, c="blue", domain=REFLECTANCE_DOMAIN)

        pos1 = c1.get_offsets()[0]
        pos2 = c2.get_offsets()[0]

        assert np.allclose(pos1, pos2, rtol=1e-10)

    def test_scatter_y_to_z_consistency(self, smith_axes):
        """Test that Y and Z parameters are consistent."""
        Z = 75 + 50j
        Y = 1 / Z

        c1 = smith_axes.scatter([Z], s=100, c="red", domain=IMPEDANCE_DOMAIN)
        c2 = smith_axes.scatter([Y], s=50, c="blue", domain=ADMITTANCE_DOMAIN)

        pos1 = c1.get_offsets()[0]
        pos2 = c2.get_offsets()[0]

        assert np.allclose(pos1, pos2, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
