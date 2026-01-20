# pylint: disable=redefined-outer-name
"""
Tests for Smith chart plotting functionality using `pysmithchart`.

This test suite validates various features of the `pysmithchart` library by
generating and saving Smith chart plots. Each test ensures specific use cases,
such as plotting impedance circles, VSWR, or frequency ranges, are rendered
correctly. The generated plots are saved as PDF files.

Test Functions:
    - test_transformer_circle: Test for plotting transformer impedance circles.
    - test_empty_smith_chart: Test for rendering an empty Smith chart with a grid.
    - test_minor_grid_colors: Test for verifying minor grid colors on the chart.
    - test_plot_single_load: Test for plotting a single load impedance.
    - test_vswr_circle: Test for plotting VSWR circles with labeled points.
    - test_frequency_range: Test for visualizing an RLC frequency range.
    - test_stub_design: Test for plotting stub designs with constant resistance and
      SWR circles.
"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pysmithchart import REFLECTANCE_DOMAIN


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


def test_empty_smith_chart(chart_dir):
    """Test for plotting an empty Smith chart."""
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith", **{"grid.major.color": "blue"})
    plt.title("Completely blue grid")

    image_path = os.path.join(chart_dir, "simple_blue.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_minor_grid_colors(chart_dir):
    """Test for verifying minor grid colors on the Smith chart."""
    plt.figure(figsize=(6, 6))
    params = {
        "grid.major.color.x": "blue",
        "grid.major.color.y": "red",
        "grid.minor.enable": True,
        "grid.minor.color.x": "yellow",
        "grid.minor.color.y": "orange",
    }
    plt.subplot(1, 1, 1, projection="smith", **params)
    plt.title("Ugly grid colors test")

    image_path = os.path.join(chart_dir, "simple_minor_colors.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_plot_single_scalar(chart_dir):
    """Test for plotting a single load on the Smith chart."""
    ZL = 75 + 50j
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(ZL, "b", linestyle=None, markersize=4, label="75+50j")
    plt.title("75 + 50j as data")
    plt.legend()

    image_path = os.path.join(chart_dir, "simple_scalar_point.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_plot_single_array(chart_dir):
    """Test for plotting a pair of points."""
    ZL = 75 + 50j
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot([ZL, 2 * ZL], color="b", marker="o", markersize=4, label="two points")
    plt.title("[75 + 50j, 150 + 100j] as data")
    plt.legend()

    image_path = os.path.join(chart_dir, "simple_array_point.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


# def test_plot_single_pair(chart_dir):
#     """Test for plotting a single load on the Smith chart."""
#     ZL = 75 + 50j
#     plt.figure(figsize=(6, 6))
#     plt.subplot(1, 1, 1, projection="smith")
#     print(np.array([[ZL.real, ZL.imag]]).shape)
#     plt.plot([[ZL.real], [ZL.imag]], color="b", marker="o", markersize=10)
#     plt.title('[[ZL.real], [ZL.imag]] as data')
#     image_path = os.path.join(chart_dir, "simple_array_point_reals.pdf")
#     plt.savefig(image_path, format="pdf")
#     plt.close()


def test_vswr_circle(chart_dir):
    """Test for plotting VSWR circle on the Smith chart."""
    Z0 = 50
    ZL = 30 + 30j

    Gamma = (ZL - Z0) / (ZL + Z0)
    lam = np.linspace(0, 0.5, 26)
    theta = 2 * np.pi * lam
    Gamma_d = Gamma * np.exp(-2j * theta)
    z = (1 + Gamma_d) / (1 - Gamma_d)
    Zd = z * Z0

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(ZL, "b", marker="o", markersize=10)
    plt.plot(Zd, "r", linestyle="", marker="o", markersize=5)

    bdict = {"facecolor": "cyan", "edgecolor": "none"}
    for i in [0, 5, 10, 15, 20]:
        plt.text(Zd[i].real / 50, Zd[i].imag / 50, " %.2fλ" % lam[i], bbox=bdict)

    image_path = os.path.join(chart_dir, "simple_vswr.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_frequency_range(chart_dir):
    """Test for plotting RLC frequency range on the Smith chart."""
    R = 50
    L = 20e-9
    C = 2e-12
    f = np.linspace(2, 20, 10) * 100e6
    omega = 2 * np.pi * f

    ZL = R - 1j / (omega * C) + 1j * omega * L

    bdict = {"facecolor": "cyan", "edgecolor": "none"}
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(ZL, "b", marker="o", markersize=10, linestyle="")
    for i in [0, 3, 5, 9]:
        x = ZL[i].real / 50
        y = ZL[i].imag / 50
        s = " %.0fMHz   " % (f[i] / 1e6)
        plt.text(x, y, s, ha="right", va="center", fontsize=10, bbox=bdict)
    plt.title("RLC Series Load, (50Ω, 2pF, 20nH)\nf=200-2000MHz")
    image_path = os.path.join(chart_dir, "simple_RLC_frequency.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_transformer_circle(chart_dir):
    """Test for plotting transformer circle on the Smith chart."""
    Z0 = 50
    ZL = 30 + 30j
    Gamma = (ZL - Z0) / (ZL + Z0)

    angle = 2 * np.pi / 8
    Gamma_prime = Gamma * np.exp(-2j * angle)
    z = (1 + Gamma_prime) / (1 - Gamma_prime)
    Zf = z * Z0

    angle = 2 * np.pi / 8 * np.linspace(0.1, 0.9, 9)
    Gamma_prime = Gamma * np.exp(-2j * angle)

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(ZL, "b", marker="o", markersize=4)
    plt.plot(Zf, "r", marker="o", markersize=4)
    plt.plot(Gamma_prime, "k", ls="", domain=REFLECTANCE_DOMAIN, marker="o", markersize=4)
    plt.title("Rotating point by λ/8")
    plt.legend()
    image_path = os.path.join(chart_dir, "simple_eighth.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_stub_design(chart_dir):
    """Test for plotting stub design with SWR and constant resistance circles."""
    Z0 = 50
    ZL = 100 + 50j

    lam = np.linspace(0, 0.5, 101)
    Gamma = (ZL - Z0) / (ZL + Z0)
    Gamma_prime = Gamma * np.exp(-2j * 2 * np.pi * lam)
    z = (1 + Gamma_prime) / (1 - Gamma_prime)
    Zd = z * Z0

    ZR = 50 + np.linspace(-1e4, 1e4, 1000) * 1j

    bdict = {"facecolor": "cyan", "edgecolor": "none"}
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(Zd, "r", marker="", label="constant |Γ|")
    plt.text(Zd[25].real / 50, Zd[25].imag / 50, " %.3fλ" % lam[25], bbox=bdict)
    plt.plot(ZR, "g", marker=None, label="constant 50Ω")
    plt.plot([ZL], "b", marker="o", markersize=5)
    plt.title("Stub design")
    plt.legend()

    image_path = os.path.join(chart_dir, "simple_stub.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()
