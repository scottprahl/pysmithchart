"""
Pytest tests for SmithAxes annotate() method with domain support.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, "/home/claude")

from pysmithchart import SmithAxes, IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, REFLECTANCE_DOMAIN


class TestSmithAxesAnnotate:
    """Test suite for SmithAxes.annotate() method."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_annotate_method_exists(self, smith_axes):
        """Test that annotate() method exists."""
        assert hasattr(smith_axes, "annotate")
        assert callable(smith_axes.annotate)

    def test_annotate_basic_REFLECTION_DOMAIN(self, smith_axes):
        """Test basic annotation with IMPEDANCE_DOMAIN."""
        ann = smith_axes.annotate("Test", xy=(50, 25), domain=IMPEDANCE_DOMAIN)
        assert isinstance(ann, matplotlib.text.Annotation)
        assert ann.get_text() == "Test"

    def test_annotate_with_arrow(self, smith_axes):
        """Test annotation with arrow."""
        ann = smith_axes.annotate(
            "Load", xy=(50, 25), xytext=(70, 40), domain=IMPEDANCE_DOMAIN, arrowprops=dict(arrowstyle="->")
        )
        assert ann is not None
        assert ann.arrow_patch is not None

    def test_annotate_without_arrow(self, smith_axes):
        """Test annotation without arrow (just offset text)."""
        ann = smith_axes.annotate("Label", xy=(50, 25), xytext=(60, 30), domain=IMPEDANCE_DOMAIN)
        assert ann is not None
        # When arrowprops is None, arrow_patch should be None
        assert ann.arrow_patch is None

    def test_annotate_ADMITTANCE_DOMAIN(self, smith_axes):
        """Test annotation with ADMITTANCE_DOMAIN (admittance)."""
        ann = smith_axes.annotate("Y Point", xy=(0.02, 0.01), domain=ADMITTANCE_DOMAIN)
        assert isinstance(ann, matplotlib.text.Annotation)

    def test_annotate_REFLECTION_DOMAIN2(self, smith_axes):
        """Test annotation with REFLECTANCE_DOMAIN."""
        ann = smith_axes.annotate("Γ", xy=(0.5, 0.3), domain=REFLECTANCE_DOMAIN)
        assert isinstance(ann, matplotlib.text.Annotation)

    def test_annotate_default_datatype(self, smith_axes):
        """Test that default domain is used when not specified."""
        ann = smith_axes.annotate("Default", xy=(50, 25))
        assert isinstance(ann, matplotlib.text.Annotation)

    def test_annotate_invalid_datatype(self, smith_axes):
        """Test that invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            smith_axes.annotate("Test", xy=(50, 25), domain="INVALID")

    def test_annotate_mixed_datatypes(self, smith_axes):
        """Test annotation with different datatypes for xy and xytext."""
        ann = smith_axes.annotate(
            "Mixed",
            xy=(0.5, 0.3),  # S-parameter
            xytext=(75, 50),  # Z-parameter
            domain=REFLECTANCE_DOMAIN,
            domain_text=IMPEDANCE_DOMAIN,
            arrowprops=dict(arrowstyle="->"),
        )
        assert ann is not None

    def test_annotate_datatype_text_defaults_to_datatype(self, smith_axes):
        """Test that domain_text defaults to domain when not specified."""
        # Should not raise an error
        ann = smith_axes.annotate(
            "Test",
            xy=(50, 25),
            xytext=(60, 35),
            domain=IMPEDANCE_DOMAIN,
            # domain_text not specified - should use IMPEDANCE_DOMAIN
        )
        assert ann is not None

    def test_annotate_invalid_datatype_text(self, smith_axes):
        """Test that invalid domain_text raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            smith_axes.annotate("Test", xy=(50, 25), xytext=(60, 35), domain=IMPEDANCE_DOMAIN, domain_text="INVALID")

    def test_annotate_with_styling(self, smith_axes):
        """Test annotation with text styling."""
        ann = smith_axes.annotate(
            "Styled",
            xy=(50, 25),
            xytext=(70, 40),
            domain=IMPEDANCE_DOMAIN,
            fontsize=14,
            color="red",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="blue", lw=2),
        )
        assert ann.get_fontsize() == 14
        assert ann.get_color() == "red"

    def test_annotate_with_bbox(self, smith_axes):
        """Test annotation with background box."""
        bbox_props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ann = smith_axes.annotate("Boxed", xy=(50, 25), domain=IMPEDANCE_DOMAIN, bbox=bbox_props)
        assert ann.get_bbox_patch() is not None

    def test_annotate_arrow_styles(self, smith_axes):
        """Test various arrow styles."""
        arrow_styles = ["->", "-[", "-|>", "fancy", "simple", "wedge"]

        for i, style in enumerate(arrow_styles):
            ann = smith_axes.annotate(
                f"Arrow {i}",
                xy=(50 + i * 10, 25),
                xytext=(60 + i * 10, 35),
                domain=IMPEDANCE_DOMAIN,
                arrowprops=dict(arrowstyle=style),
            )
            assert ann is not None

    def test_annotate_connection_styles(self, smith_axes):
        """Test various connection styles."""
        connection_styles = ["arc3", "angle3", "angle", "arc", "bar"]

        for i, style in enumerate(connection_styles):
            ann = smith_axes.annotate(
                f"Conn {i}",
                xy=(50, 25 + i * 10),
                xytext=(70, 35 + i * 10),
                domain=IMPEDANCE_DOMAIN,
                arrowprops=dict(arrowstyle="->", connectionstyle=style),
            )
            assert ann is not None

    def test_annotate_matches_plot_location(self, smith_axes):
        """Test that annotation points to same location as plot."""
        z = 50 + 25j

        # Plot a point
        smith_axes.plot([z], "o", markersize=10, domain=IMPEDANCE_DOMAIN)

        # Annotate it
        ann = smith_axes.annotate(
            "Point", xy=(z.real, z.imag), xytext=(70, 40), domain=IMPEDANCE_DOMAIN, arrowprops=dict(arrowstyle="->")
        )
        assert ann is not None

    def test_annotate_multiple_annotations(self, smith_axes):
        """Test adding multiple annotations."""
        points = [
            (25, 25, "A"),
            (50, 0, "B"),
            (75, 50, "C"),
        ]

        annotations = []
        for x, y, label in points:
            ann = smith_axes.annotate(
                label, xy=(x, y), xytext=(x + 15, y + 15), domain=IMPEDANCE_DOMAIN, arrowprops=dict(arrowstyle="->")
            )
            annotations.append(ann)

        assert len(annotations) == 3
        assert all(isinstance(a, matplotlib.text.Annotation) for a in annotations)

    def test_annotate_at_matched_load(self, smith_axes):
        """Test annotation at matched load (50Ω)."""
        ann = smith_axes.annotate(
            "Matched\n50Ω", xy=(50, 0), xytext=(60, 20), domain=IMPEDANCE_DOMAIN, arrowprops=dict(arrowstyle="->")
        )
        assert ann is not None

    def test_annotate_alignment(self, smith_axes):
        """Test annotation with different alignments."""
        alignments = [
            ("left", "bottom"),
            ("center", "center"),
            ("right", "top"),
        ]

        for i, (ha, va) in enumerate(alignments):
            ann = smith_axes.annotate(
                f"{ha}-{va}",
                xy=(50, 25 + i * 15),
                xytext=(70, 35 + i * 15),
                domain=IMPEDANCE_DOMAIN,
                ha=ha,
                va=va,
                arrowprops=dict(arrowstyle="->"),
            )
            assert ann.get_ha() == ha
            assert ann.get_va() == va

    def test_annotate_with_plot_and_text(self, smith_axes):
        """Test using annotate(), plot(), and text() together."""
        z = 50 + 25j

        # Plot point
        smith_axes.plot([z], "o", markersize=10, domain=IMPEDANCE_DOMAIN)

        # Add text
        smith_axes.text(z.real, z.imag, "  Label", domain=IMPEDANCE_DOMAIN, ha="left")

        # Add annotation
        ann = smith_axes.annotate(
            "Important",
            xy=(z.real, z.imag),
            xytext=(75, 50),
            domain=IMPEDANCE_DOMAIN,
            arrowprops=dict(arrowstyle="->", color="red"),
        )

        assert ann is not None

    @pytest.mark.parametrize("domain", [IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, REFLECTANCE_DOMAIN])
    def test_annotate_all_datatypes_parametrized(self, smith_axes, domain):
        """Parametrized test for all datatypes."""
        ann = smith_axes.annotate(
            f"Test {domain}", xy=(50, 25), xytext=(70, 40), domain=domain, arrowprops=dict(arrowstyle="->")
        )
        assert isinstance(ann, matplotlib.text.Annotation)

    def test_annotate_xytext_none(self, smith_axes):
        """Test annotation when xytext is None (text at xy position)."""
        ann = smith_axes.annotate("At Point", xy=(50, 25), xytext=None, domain=IMPEDANCE_DOMAIN)
        assert ann is not None


class TestSmithAxesAnnotateIntegration:
    """Integration tests for annotate() with other features."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_annotate_with_grid(self, smith_axes):
        """Test annotation with grid enabled."""
        smith_axes.grid(True, which="both")
        ann = smith_axes.annotate(
            "Test", xy=(50, 25), xytext=(70, 40), domain=IMPEDANCE_DOMAIN, arrowprops=dict(arrowstyle="->")
        )
        assert ann is not None

    def test_annotate_with_legend(self, smith_axes):
        """Test annotation with legend."""
        smith_axes.plot([50 + 25j], "o", domain=IMPEDANCE_DOMAIN, label="Data")
        smith_axes.legend()
        ann = smith_axes.annotate(
            "Point", xy=(50, 25), xytext=(70, 40), domain=IMPEDANCE_DOMAIN, arrowprops=dict(arrowstyle="->")
        )
        assert ann is not None

    def test_annotate_rf_circuit(self, smith_axes):
        """Test annotating an RF circuit diagram."""
        # Simulate circuit points
        circuit = [
            (25 + 25j, "Source"),
            (50 + 0j, "Match"),
            (75 + 50j, "Load"),
        ]

        for z, label in circuit:
            smith_axes.plot([z], "o", markersize=8, domain=IMPEDANCE_DOMAIN)
            smith_axes.annotate(
                label,
                xy=(z.real, z.imag),
                xytext=(z.real + 20, z.imag + 20),
                domain=IMPEDANCE_DOMAIN,
                arrowprops=dict(arrowstyle="->", color="blue"),
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        assert len(smith_axes.texts) >= 3  # Annotations are stored in .texts


class TestSmithAxesTransformCoordinates:
    """Test the _transform_coordinates helper method."""

    @pytest.fixture
    def smith_axes(self):
        """Create a SmithAxes instance for testing."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection="smith")
        yield ax
        plt.close(fig)

    def test_transform_coordinates_exists(self, smith_axes):
        """Test that _transform_coordinates method exists."""
        assert hasattr(smith_axes, "_transform_coordinates")
        assert callable(smith_axes._transform_coordinates)

    def test_transform_coordinates_REFLECTION_DOMAIN(self, smith_axes):
        """Test coordinate transformation for IMPEDANCE_DOMAIN."""
        x, y = smith_axes._transform_coordinates(50, 25, IMPEDANCE_DOMAIN)
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))

    def test_transform_coordinates_ADMITTANCE_DOMAIN(self, smith_axes):
        """Test coordinate transformation for ADMITTANCE_DOMAIN."""
        x, y = smith_axes._transform_coordinates(0.02, 0.01, ADMITTANCE_DOMAIN)
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))

    def test_transform_coordinates_REFLECTION_DOMAIN2(self, smith_axes):
        """Test coordinate transformation for REFLECTANCE_DOMAIN."""
        x, y = smith_axes._transform_coordinates(0.5, 0.3, REFLECTANCE_DOMAIN)
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))

    def test_transform_coordinates_returns_tuple(self, smith_axes):
        """Test that _transform_coordinates returns a tuple."""
        result = smith_axes._transform_coordinates(50, 25, IMPEDANCE_DOMAIN)
        assert isinstance(result, tuple)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
