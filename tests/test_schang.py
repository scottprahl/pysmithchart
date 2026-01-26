# pylint: disable=redefined-outer-name
"""
Test basic Smith chart functionality.

This test was forked from https://github.com/schang412/mpl-smithchart

Tests:
------
- `test_plot_point`: Verifies the ability to plot individual points
- `test_plot_s_param`: Tests the plotting of S-parameters on the Smith chart.
- `test_plot_labels`: Ensures plots with labels and legends render correctly.
- `test_plot_normalized_axes`: Normalized axes different impedances.
- `test_plot_grid_styles`: Examines various grid style configurations.
"""

from itertools import product
import os
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pysmithchart import R_DOMAIN


@pytest.fixture
def chart_dir(tmpdir):
    """
    Fixture to provide the directory for saving charts.

    - Locally: Saves charts in the `charts` folder within the `tests` directory.
    - On GitHub Actions: Uses the provided `tmpdir`.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        return tmpdir

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_chart_dir = os.path.join(script_dir, "charts")
    os.makedirs(local_chart_dir, exist_ok=True)
    return local_chart_dir


def s11_of_cap(freq):
    """Calculate S11 for a capacitor connected to ground."""
    return (1 - 1j * freq * 1e-9) / (1 + 1j * freq * 1e-9)


def s11_of_parallel_cap_res(freq, z0=50):
    """Calculate S11 for a capacitor in parallel with a resistor to ground."""
    s = 2j * np.pi * freq
    return (50 - z0 * (1 + s * 1e-9 * 50)) / (50 + z0 * (1 + s * 1e-9 * 50))


@pytest.mark.parametrize(
    "points",
    [
        [200 + 100j, 50.0, 50 - 10j],
    ],
)
def test_plot_points(chart_dir, points):
    """Test plotting a single point on the Smith chart."""
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    for point in points:
        plt.plot(point, marker="o", markersize=6, label=f"{point}")  # Plot each point
    plt.title("Plot of Three Points")
    plt.legend()
    plt.savefig(os.path.join(chart_dir, "schang_points.pdf"), format="pdf")
    plt.close()


@pytest.mark.filterwarnings("ignore:S-parameter magnitude")
def test_plot_s_param(chart_dir):
    """Test plotting S-parameters on the Smith chart."""
    freqs = np.logspace(0, 9, 200)
    s11 = s11_of_cap(freqs)
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(s11, markevery=1, domain=R_DOMAIN)
    plt.title("S-Parameters of a Capacitor")
    plt.savefig(os.path.join(chart_dir, "schang_cap.pdf"), format="pdf")
    plt.close()


@pytest.mark.filterwarnings("ignore:S-parameter magnitude")
def test_plot_labels(chart_dir):
    """Test plotting with labels and a legend."""
    freqs = np.logspace(0, 9, 200)
    s11 = s11_of_cap(freqs)
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(s11, markevery=1, domain=R_DOMAIN, label="s11")
    plt.legend()
    plt.title("S-Parameters with Labels and Legend")
    plt.savefig(os.path.join(chart_dir, "schang_labels.pdf"), format="pdf")
    plt.close()


def test_plot_normalized_axes(chart_dir):
    """Test plotting with normalized and non-normalized axes."""
    freqs = np.logspace(0, 9, 200)
    plt.figure(figsize=(18, 12)).set_layout_engine("tight")

    for i, (do_normalize_axes, impedance) in enumerate(product([True, False], [10, 50, 200])):
        s11 = s11_of_parallel_cap_res(freqs, z0=impedance)
        plt.subplot(
            2,
            3,
            i + 1,
            projection="smith",
            Z0=impedance,
        )
        plt.plot(s11, domain=R_DOMAIN)
        plt.title(f"Impedance: {impedance} Ω")

    plt.savefig(os.path.join(chart_dir, "schang_normalized_axes.pdf"), format="pdf")
    plt.close()


def test_plot_grid_styles(chart_dir):
    """Test plotting with various grid styles."""
    freqs = np.logspace(0, 9, 200)
    s11 = s11_of_parallel_cap_res(freqs)

    # Define the options for major, minor enable, and minor fancy
    fancy_options = [True, False]
    minor_enable_options = [True, False]

    # Generate all combinations
    combinations = product(fancy_options, minor_enable_options)

    # Iterate through the combinations
    plt.figure(figsize=(12, 12))
    for i, (fancy, minor_enable) in enumerate(combinations):
        sc = {"grid.fancy": fancy, "grid.Z.minor.enable": minor_enable}
        plt.subplot(2, 2, i + 1, projection="smith", **sc)
        major_str = "fancy" if fancy else "standard"
        plt.plot(s11, domain=R_DOMAIN)
        plt.title(f"Grid: {major_str}")

    plt.suptitle("Grid Style Variations")
    plt.savefig(os.path.join(chart_dir, "schang_grid_styles.pdf"), format="pdf")
    plt.close()
