# pylint: disable=redefined-outer-name
"""
Tests for Smith chart functionality using `pysmithchart`.

This file contains unit tests to validate the behavior of the `pysmithchart` library.
Specifically, it includes tests for plotting elements like VSWR (Voltage Standing
Wave Ratio) circles on the Smith chart and saving them as PDF files.

This is adapted from https://github.com/soerenbnoergaard/pySmithPlot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pytest

from pysmithchart import Z_PARAMETER


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


def test_vswr_circle(chart_dir):
    """Test plotting a VSWR circle on the Smith chart."""
    # Create VSWR circle
    G = 0.5 * np.exp(2j * np.pi * np.linspace(0, 1, 100))
    ZL = -(G + 1) / (G - 1)

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith", axes_impedance=1)

    plt.plot(ZL, "k", datatype=Z_PARAMETER, label="VSWR Circle")
    plt.plot(1 + 0j, "b", datatype=Z_PARAMETER, marker="o", label="$1+0j$")
    plt.plot(1 + 1j, "r", datatype=Z_PARAMETER, marker="o", label="$1+1j$")
    plt.plot(0.5 - 0.5j, "g", datatype=Z_PARAMETER, marker="o", label="$0.5-0.5j$")

    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(chart_dir, "vswr_circle.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()
