"""
Pytest test suite for SmithAxes text() method.

This module tests the custom text() method implementation that properly
transforms impedance/admittance coordinates to Smith chart display space.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import sys

sys.path.insert(0, "/home/claude")

from pysmithchart import IMPEDANCE_DOMAIN, REFLECTANCE_DOMAIN


class TestSmithAxesTextMethod:
    """Test suite for SmithAxes.text() method."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_text_method_exists(self, smith_axes):
        """Test that the text() method exists on SmithAxes."""
        assert hasattr(smith_axes, "text")
        assert callable(smith_axes.text)

    def test_text_basic_call(self, smith_axes):
        """Test basic text() method call returns a Text object."""
        text_obj = smith_axes.text(50, 25, "Test Label")
        assert isinstance(text_obj, matplotlib.text.Text)

    def test_text_at_origin(self, smith_axes):
        """Test text placement at the origin (0, 0)."""
        text_obj = smith_axes.text(0, 0, "Origin")
        assert text_obj is not None
        # Check that text was added to the axes
        assert text_obj in smith_axes.texts

    def test_text_at_matched_load(self, smith_axes):
        """Test text placement at matched load (50Ω for default normalization)."""
        text_obj = smith_axes.text(50, 0, "Matched")
        assert text_obj is not None
        # The text should be transformed to the center of the Smith chart
        # After Moebius transform, 50+0j should map close to origin

    def test_text_with_positive_reactance(self, smith_axes):
        """Test text placement with positive (inductive) reactance."""
        text_obj = smith_axes.text(50, 50, "Inductive")
        assert text_obj is not None
        assert text_obj.get_text() == "Inductive"

    def test_text_with_negative_reactance(self, smith_axes):
        """Test text placement with negative (capacitive) reactance."""
        text_obj = smith_axes.text(50, -50, "Capacitive")
        assert text_obj is not None
        assert text_obj.get_text() == "Capacitive"

    def test_text_multiple_labels(self, smith_axes):
        """Test adding multiple text labels."""
        coords = [(25, 25), (50, 0), (75, 50), (100, -25)]
        labels = ["Point1", "Point2", "Point3", "Point4"]

        text_objects = []
        for (x, y), label in zip(coords, labels):
            text_obj = smith_axes.text(x, y, label)
            text_objects.append(text_obj)

        assert len(text_objects) == 4
        assert all(isinstance(obj, matplotlib.text.Text) for obj in text_objects)
        assert len(smith_axes.texts) >= 4

    def test_text_with_styling(self, smith_axes):
        """Test text with various styling options."""
        text_obj = smith_axes.text(
            50, 25, "Styled Text", fontsize=14, color="red", fontweight="bold", ha="center", va="bottom"
        )
        assert text_obj.get_fontsize() == 14
        assert text_obj.get_color() == "red"
        assert text_obj.get_weight() == "bold"
        assert text_obj.get_ha() == "center"
        assert text_obj.get_va() == "bottom"

    def test_text_with_bbox(self, smith_axes):
        """Test text with background box."""
        bbox_props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        text_obj = smith_axes.text(50, 0, "Boxed Text", bbox=bbox_props)
        assert text_obj.get_bbox_patch() is not None

    def test_text_with_rotation(self, smith_axes):
        """Test text with rotation."""
        text_obj = smith_axes.text(50, 25, "Rotated", rotation=45)
        assert text_obj.get_rotation() == 45

    def test_text_alignment_options(self, smith_axes):
        """Test all alignment options."""
        h_alignments = ["left", "center", "right"]
        v_alignments = ["top", "center", "bottom"]

        for ha in h_alignments:
            for va in v_alignments:
                text_obj = smith_axes.text(50, 25, f"{ha}-{va}", ha=ha, va=va)
                assert text_obj.get_ha() == ha
                assert text_obj.get_va() == va

    def test_text_with_complex_impedance(self, smith_axes):
        """Test text placement using complex impedance values."""
        z = 75 + 50j
        text_obj = smith_axes.text(z.real, z.imag, f"{z}")
        assert text_obj is not None
        assert text_obj.get_text() == f"{z}"

    def test_text_array_of_coordinates(self, smith_axes):
        """Test adding text at multiple coordinates from arrays."""
        x_coords = np.array([25, 50, 75, 100])
        y_coords = np.array([25, 0, 50, -25])

        for x, y in zip(x_coords, y_coords):
            text_obj = smith_axes.text(x, y, f"{x}+{y}j")
            assert text_obj is not None

    def test_text_transformation_applied(self, smith_axes):
        """Test that transformation is applied to coordinates."""
        # Create text at a known impedance
        x, y = 50, 25
        text_obj = smith_axes.text(x, y, "Test", domain=IMPEDANCE_DOMAIN)

        # Get the transformed position
        pos = text_obj.get_position()

        # For IMPEDANCE_DOMAIN, the transformation is:
        # z = 50 + 25j -> normalize by Z0 -> z_to_xy
        from pysmithchart.utils import z_to_xy

        z = 50 + 25j
        # IMPEDANCE_DOMAIN is always normalized by Z0 in the transform pipeline
        if smith_axes._get_key("axes.normalize"):
            z = z / smith_axes._get_key("axes.Z0")

        x_expected, y_expected = z_to_xy(z)

        # Check that the text is at the transformed position
        assert np.isclose(pos[0], x_expected, rtol=1e-5)
        assert np.isclose(pos[1], y_expected, rtol=1e-5)

    def test_text_transformation_REFLECTION_DOMAIN2(self, smith_axes):
        """Test that Moebius transformation is applied for REFLECTANCE_DOMAIN."""
        # Create text in S-parameter space (reflection coefficient)
        x, y = 0.5, 0.3
        text_obj = smith_axes.text(x, y, "Test", domain=REFLECTANCE_DOMAIN)

        # Get the transformed position
        pos = text_obj.get_position()

        # For REFLECTANCE_DOMAIN, transformation is: moebius_inv_z(s) -> z_to_xy
        from pysmithchart.utils import z_to_xy

        s = 0.5 + 0.3j
        z_impedance = smith_axes.moebius_inv_z(s)
        x_expected, y_expected = z_to_xy(z_impedance)

        # Check that the text is at the transformed position
        assert np.isclose(pos[0], x_expected, rtol=1e-5)
        assert np.isclose(pos[1], y_expected, rtol=1e-5)

    def test_text_uses_transdata_by_default(self, smith_axes):
        """Test that text uses transData transform by default."""
        text_obj = smith_axes.text(50, 25, "Test")
        assert text_obj.get_transform() == smith_axes.transData

    def test_text_custom_transform_override(self, smith_axes):
        """Test that custom transform can be provided."""
        custom_transform = smith_axes.transAxes
        text_obj = smith_axes.text(0.5, 0.5, "Center", transform=custom_transform)
        assert text_obj.get_transform() == custom_transform

    def test_text_with_zero_impedance(self, smith_axes):
        """Test text at zero impedance (short circuit)."""
        text_obj = smith_axes.text(0, 0, "Short")
        assert text_obj is not None

    def test_text_with_high_impedance(self, smith_axes):
        """Test text at high impedance values."""
        text_obj = smith_axes.text(500, 500, "High Z")
        assert text_obj is not None

    def test_text_with_pure_resistance(self, smith_axes):
        """Test text with pure resistance (zero reactance)."""
        resistances = [10, 25, 50, 100, 200]
        for r in resistances:
            text_obj = smith_axes.text(r, 0, f"{r}Ω")
            assert text_obj is not None

    def test_text_with_pure_reactance(self, smith_axes):
        """Test text with pure reactance (zero resistance)."""
        reactances = [-100, -50, 50, 100]
        for x in reactances:
            text_obj = smith_axes.text(0, x, f"{x}jΩ")
            assert text_obj is not None

    def test_text_string_formatting(self, smith_axes):
        """Test various string formats."""
        formats = [
            "Simple text",
            "Multi\nline\ntext",
            "Unicode: αβγδε",
            "Math: ∞ ± ∂",
            "Numbers: 123.456",
            r"$\LaTeX$ math",
        ]
        for i, fmt in enumerate(formats):
            text_obj = smith_axes.text(50 + i * 10, 25, fmt)
            assert text_obj.get_text() == fmt

    def test_text_empty_string(self, smith_axes):
        """Test text with empty string."""
        text_obj = smith_axes.text(50, 25, "")
        assert text_obj.get_text() == ""

    def test_text_with_plot_points(self, smith_axes):
        """Test text labels accompanying plot points."""
        # Plot some points
        impedances = [25 + 25j, 50 + 0j, 75 + 50j]
        smith_axes.plot(impedances, "o", domain=IMPEDANCE_DOMAIN, label="Points")

        # Add text labels at the same coordinates
        for z in impedances:
            text_obj = smith_axes.text(z.real, z.imag, f"{z}", ha="left", va="bottom")
            assert text_obj is not None

    def test_text_different_normalizations(self):
        """Test text with different impedance normalizations."""
        normalizations = [50, 75, 100]

        for norm in normalizations:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="smith", Z0=norm)

            # Add text at normalized impedance
            text_obj = ax.text(norm, 0, f"{norm}Ω")
            assert text_obj is not None

            plt.close(fig)

    def test_text_returns_text_object(self, smith_axes):
        """Test that text() returns a proper Text object that can be modified."""
        text_obj = smith_axes.text(50, 25, "Modifiable")

        # Modify the returned object
        text_obj.set_fontsize(16)
        text_obj.set_color("blue")
        text_obj.set_weight("bold")

        # Verify modifications
        assert text_obj.get_fontsize() == 16
        assert text_obj.get_color() == "blue"
        assert text_obj.get_weight() == "bold"

    def test_text_with_negative_coordinates(self, smith_axes):
        """Test text with negative real coordinates (invalid for impedance)."""
        # Negative resistance doesn't make physical sense but should not crash
        text_obj = smith_axes.text(-10, 25, "Invalid")
        assert text_obj is not None

    def test_text_zorder(self, smith_axes):
        """Test text z-order (layering)."""
        text_obj = smith_axes.text(50, 25, "Layered", zorder=10)
        assert text_obj.get_zorder() == 10

    def test_text_alpha_transparency(self, smith_axes):
        """Test text with transparency."""
        text_obj = smith_axes.text(50, 25, "Transparent", alpha=0.5)
        assert text_obj.get_alpha() == 0.5

    def test_text_font_family(self, smith_axes):
        """Test text with different font families."""
        families = ["serif", "sans-serif", "monospace"]
        for i, family in enumerate(families):
            text_obj = smith_axes.text(50 + i * 10, 25, family, family=family)
            assert text_obj is not None

    def test_text_clip_behavior(self, smith_axes):
        """Test text clipping behavior."""
        text_obj = smith_axes.text(50, 25, "Clipped", clip_on=True)
        assert text_obj.get_clip_on()

    @pytest.mark.parametrize(
        "x,y,label",
        [
            (25, 25, "Point A"),
            (50, 0, "Point B"),
            (75, 50, "Point C"),
            (100, -25, "Point D"),
            (0, 0, "Point E"),
        ],
    )
    def test_text_parametrized_coordinates(self, smith_axes, x, y, label):
        """Parametrized test for various coordinates."""
        text_obj = smith_axes.text(x, y, label)
        assert text_obj is not None
        assert text_obj.get_text() == label


class TestSmithAxesTextIntegration:
    """Integration tests for text() with other SmithAxes features."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_text_with_grid(self, smith_axes):
        """Test text rendering with grid enabled."""
        smith_axes.grid(True, which="both")
        text_obj = smith_axes.text(50, 25, "With Grid")
        assert text_obj is not None

    def test_text_with_legend(self, smith_axes):
        """Test text with legend present."""
        smith_axes.plot([50 + 25j], "o", domain=IMPEDANCE_DOMAIN, label="Data")
        smith_axes.legend()
        text_obj = smith_axes.text(50, 25, "With Legend")
        assert text_obj is not None

    def test_text_after_clear(self, smith_axes):
        """Test text addition after axes clear."""
        smith_axes.text(50, 25, "First")
        smith_axes.clear()
        text_obj = smith_axes.text(50, 25, "After Clear")
        assert text_obj is not None

    def test_text_persistence_across_draws(self, smith_axes):
        """Test that text persists across multiple draw calls."""
        text_obj = smith_axes.text(50, 25, "Persistent")
        fig = smith_axes.get_figure()

        # Draw multiple times
        fig.canvas.draw()
        fig.canvas.draw()

        # Text should still be there
        assert text_obj in smith_axes.texts

    def test_text_with_annotations(self, smith_axes):
        """Test text alongside annotations."""
        smith_axes.text(50, 25, "Text Label")
        smith_axes.annotate("Annotation", xy=(50, 25), xytext=(75, 50), arrowprops=dict(arrowstyle="->"))
        assert len(smith_axes.texts) >= 2  # Both text and annotation


class TestSmithAxesTextEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_text_with_nan_coordinates(self, smith_axes):
        """Test text with NaN coordinates."""
        text_obj = smith_axes.text(np.nan, np.nan, "NaN")
        assert text_obj is not None

    def test_text_with_inf_coordinates(self, smith_axes):
        """Test text with infinite coordinates."""
        text_obj = smith_axes.text(np.inf, np.inf, "Inf")
        assert text_obj is not None

    def test_text_with_very_large_coordinates(self, smith_axes):
        """Test text with very large coordinate values."""
        text_obj = smith_axes.text(1e6, 1e6, "Large")
        assert text_obj is not None

    def test_text_with_very_small_coordinates(self, smith_axes):
        """Test text with very small coordinate values."""
        text_obj = smith_axes.text(1e-6, 1e-6, "Small")
        assert text_obj is not None

    def test_text_numeric_string(self, smith_axes):
        """Test text with numeric values as strings."""
        text_obj = smith_axes.text(50, 25, 123.456)
        # Matplotlib should convert it to string
        assert isinstance(text_obj.get_text(), str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
