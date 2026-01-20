# pylint: disable=redefined-outer-name
"""
Tests for Smith chart functionality using `pysmithchart`.

This file contains unit tests to validate the behavior of the `pysmithchart` library.
Specifically, it includes tests for plotting elements like VSWR (Voltage Standing
Wave Ratio) circles on the Smith chart and saving them as PDF files.

This is heavily modified from https://github.com/soerenbnoergaard/pySmithPlot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pytest

from pysmithchart import REFLECTANCE_DOMAIN, ADMITTANCE_DOMAIN
from pysmithchart.utils import calc_gamma, calc_load


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


def test_vswr_circle_z(chart_dir):
    """Test plotting a VSWR circle on the Smith chart."""
    # Create VSWR circle
    Gamma = 0.5 * np.exp(2j * np.pi * np.linspace(0, 1, 40))
    ZL = -(Gamma + 1) / (Gamma - 1)

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith", Z0=1)

    plt.plot(ZL[:-1], "k", ls="", label="VSWR Circle")
    plt.plot(1 + 0j, "b", marker="o", label="$1+0j$")
    plt.plot(1 + 1j, "r", marker="o", label="$1+1j$")
    plt.plot(0.5 - 0.5j, "g", marker="o", label="$0.5-0.5j$")

    plt.title("Plotting using Z-Parameters")
    plt.legend()
    output_path = os.path.join(chart_dir, "vswr_circle_z.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()


def test_vswr_circle_s(chart_dir):
    """Test plotting a VSWR circle on the Smith chart."""
    Z0 = 50

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")

    Gamma = 0.5 * np.exp(2j * np.pi * np.linspace(0, 1, 40))
    plt.plot(Gamma[:-1], "k", domain=REFLECTANCE_DOMAIN, label="VSWR Circle")

    ZL = Z0 * (1 + 0j)
    Gamma = calc_gamma(Z0, ZL)
    plt.plot(Gamma, "b", domain=REFLECTANCE_DOMAIN, marker="o", label="$1+0j$")

    ZL = Z0 * (0.5 - 0.5j)
    Gamma = calc_gamma(Z0, ZL)
    plt.plot(Gamma, "g", domain=REFLECTANCE_DOMAIN, marker="o", label="$0.5-0.5j$")

    ZL = Z0 * (1 + 1j)
    Gamma = calc_gamma(Z0, ZL)
    plt.plot(Gamma, "r", domain=REFLECTANCE_DOMAIN, marker="o", label="$1+1j$")

    plt.title("Plotting using S-Parameters")
    plt.legend()
    output_path = os.path.join(chart_dir, "vswr_circle_s.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()


def test_vswr_circle_y(chart_dir):
    """Test plotting VSWR circle using normalized admittances."""
    Z0 = 1

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")

    Gamma = 0.5 * np.exp(2j * np.pi * np.linspace(0, 1, 40))
    ZL = calc_load(Z0, Gamma)  # normalized impedance
    YL = 1 / ZL
    plt.plot(YL[:-1], "k", domain=ADMITTANCE_DOMAIN, label="VSWR Circle")

    ZL = 1 + 0j
    YL = 1 / ZL
    plt.plot(YL, "b", domain=ADMITTANCE_DOMAIN, marker="o", label="$1+0j$")

    ZL = 0.5 - 0.5j
    YL = 1 / ZL
    plt.plot(YL, "g", domain=ADMITTANCE_DOMAIN, marker="o", label="$0.5-0.5j$")

    ZL = 1 + 1j
    YL = 1 / ZL
    plt.plot(YL, "r", domain=ADMITTANCE_DOMAIN, marker="o", label="$1+1j$")

    plt.title("Plotting using Y-Parameters")
    plt.legend()
    output_path = os.path.join(chart_dir, "vswr_circle_y.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()


def test_vswr_circle_mixed(chart_dir):
    """Test plotting a VSWR circle on the Smith chart."""
    Gamma = 0.5 * np.exp(2j * np.pi * np.linspace(0, 1, 40))

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith", Z0=1)

    plt.plot(Gamma[:-1], "k", domain=REFLECTANCE_DOMAIN, label="VSWR Circle")
    plt.plot(1 + 0j, "b", marker="o", label="$1+0j$")
    plt.plot(1 + 1j, "r", marker="o", label="$1+1j$")
    plt.plot(1 / (0.5 - 0.5j), "g", domain=ADMITTANCE_DOMAIN, marker="o", label="$0.5-0.5j$")

    plt.title("Plotting using Mixed-Parameters")
    plt.legend()
    output_path = os.path.join(chart_dir, "vswr_circle_mixed.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()
