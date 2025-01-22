import sys
import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pysmithchart import SmithAxes

# Utility function
def cs(z, N=5):
    """Convert complex number to string for printing."""
    if z.imag < 0:
        form = "(%% .%df - %%.%dfj)" % (N, N)
    else:
        form = "(%% .%df + %%.%dfj)" % (N, N)
    return form % (z.real, abs(z.imag))


@pytest.fixture
def setup_chart_dir(tmp_path):
    """Fixture to set up a temporary directory for saving charts."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_dir = os.path.join(script_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    return chart_dir


def test_transformer_circle(setup_chart_dir):
    """Test for plotting transformer circle on the Smith chart."""
    Z0 = 50
    ZL = 30 + 30j

    Gamma = (ZL - Z0) / (ZL + Z0)
    Gamma_prime = Gamma * np.exp(-2j * 2 * np.pi / 8)
    z = (1 + Gamma_prime) / (1 - Gamma_prime)
    Zd = z * Z0

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1, projection="smith", grid_major_enable=True)
    plt.plot(ZL, "b", marker="o", markersize=10, datatype=SmithAxes.Z_PARAMETER)
    plt.plot(Zd, "r", marker="o", markersize=5, datatype=SmithAxes.Z_PARAMETER)
    image_path = os.path.join(setup_chart_dir, "lambda_over_eight.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_empty_smith_chart(setup_chart_dir):
    """Test for plotting an empty Smith chart."""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1, projection="smith")
    SmithAxes.update_scParams(reset=True, instance=ax, grid_minor_color="blue")
    image_path = os.path.join(setup_chart_dir, "plain_smith.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_minor_grid_colors():
    """Test for verifying minor grid colors on the Smith chart."""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1, projection="smith")
    SmithAxes.update_scParams(
        reset=True,
        instance=ax,
        grid_major_color_x="blue",
        grid_minor_color_x="blue",
        grid_major_color_y="orange",
        grid_minor_color_y="orange",
    )
    plt.close()


def test_plot_single_load(setup_chart_dir):
    """Test for plotting a single load on the Smith chart."""
    ZL = 75 + 50j
    Z0 = 50
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1, projection="smith")
    SmithAxes.update_scParams(instance=ax, reset=True, grid_major_enable=True)
    plt.plot([ZL], color="b", marker="o", markersize=10, datatype=SmithAxes.Z_PARAMETER)
    image_path = os.path.join(setup_chart_dir, "one_point.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_vswr_circle(setup_chart_dir):
    """Test for plotting VSWR circle on the Smith chart."""
    Z0 = 50
    ZL = 30 + 30j

    Gamma = (ZL - Z0) / (ZL + Z0)
    lam = np.linspace(0, 0.5, 26)
    theta = 2 * np.pi * lam
    Gamma_d = Gamma * np.exp(-2j * theta)
    z = (1 + Gamma_d) / (1 - Gamma_d)
    Zd = z * Z0

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1, projection="smith", grid_major_enable=True)
    plt.plot(ZL, "b", marker="o", markersize=10, datatype=SmithAxes.Z_PARAMETER)
    plt.plot(Zd, "r", linestyle="", marker="o", markersize=5, datatype=SmithAxes.Z_PARAMETER)
    for i in [0, 5, 10, 15, 20]:
        plt.text(Zd[i].real / 50, Zd[i].imag / 50, " %.2fλ" % lam[i], bbox=dict(facecolor="cyan", edgecolor="none"))
    image_path = os.path.join(setup_chart_dir, "vswr.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_frequency_range(setup_chart_dir):
    """Test for plotting RLC frequency range on the Smith chart."""
    R = 50
    L = 20e-9
    C = 2e-12
    f = np.linspace(2, 20, 10) * 100e6
    omega = 2 * np.pi * f

    Z0 = 50
    ZL = R + 1 / (1j * omega * C) + 1j * omega * L

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1, projection="smith", grid_major_enable=True)
    plt.plot(ZL, "b", marker="o", markersize=10, linestyle="", datatype=SmithAxes.Z_PARAMETER)
    for i in [0, 3, 5, 9]:
        plt.text(ZL[i].real / 50, ZL[i].imag / 50, " %.0fMHz" % (f[i] / 1e6), bbox=dict(facecolor="cyan", edgecolor="none"))
    plt.title("RLC Series Load, R=50Ω, C=2pF, L=20nH")
    image_path = os.path.join(setup_chart_dir, "RLC_frequency.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()


def test_stub_design(setup_chart_dir):
    """Test for plotting stub design with SWR and constant resistance circles."""
    Z0 = 50
    ZL = 100 + 50j

    lam = np.linspace(0, 0.5, 101)
    Gamma = (ZL - Z0) / (ZL + Z0)
    Gamma_prime = Gamma * np.exp(-2j * 2 * np.pi * lam)
    z = (1 + Gamma_prime) / (1 - Gamma_prime)
    Zd = z * Z0

    ZR = 50 + np.linspace(-1e4, 1e4, 1000) * 1j

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1, projection="smith", grid_major_enable=True)
    plt.plot([ZL], "b", marker="o", markersize=10, datatype=SmithAxes.Z_PARAMETER)
    plt.plot(Zd, "r", marker="", datatype=SmithAxes.Z_PARAMETER)
    plt.text(Zd[25].real / 50, Zd[25].imag / 50, " %.3fλ" % lam[25], bbox=dict(facecolor="cyan", edgecolor="none"))
    plt.plot(ZR, "g", marker=None, datatype=SmithAxes.Z_PARAMETER)
    image_path = os.path.join(setup_chart_dir, "stub.pdf")
    plt.savefig(image_path, format="pdf")
    plt.close()
