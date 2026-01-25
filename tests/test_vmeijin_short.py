# pylint: disable=redefined-outer-name
"""
Tests for advanced Smith chart plotting functionality using `pysmithchart`.

This test file validates various plotting configurations on the Smith chart,
such as interpolation, equipoints, and custom markers.
"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pysmithchart import R_DOMAIN

# have matplotlib legend include three markers instead of one
rcParams.update({"legend.numpoints": 3})


@pytest.fixture
def setup_environment(tmpdir):
    """
    Fixture to provide the directory for saving charts.

    - Locally: Saves charts in the `charts` folder within the `tests` directory.
    - On GitHub Actions: Uses the provided `tmpdir`.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    s11_data_path = os.path.join(script_dir, os.path.join("data", "s11.csv"))
    s11 = load_complex_data(s11_data_path)
    s22_data_path = os.path.join(script_dir, "data/s22.csv")
    s22 = load_complex_data(s22_data_path)

    if os.getenv("GITHUB_ACTIONS") == "true":
        chart_dir = tmpdir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        chart_dir = os.path.join(script_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)

    return s11, s22, chart_dir


def load_complex_data(file_path, step=100):
    """Load and return S data from a CSV file."""
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)[::step]
    return data[:, 1] + 1j * data[:, 2]


def test_smith_chart_plot1(setup_environment):
    """Test for plotting data on a Smith chart using various configurations."""
    _, _, chart_dir = setup_environment

    # Plot data
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot([10, 50, 100], markevery=1, label="[10, 50, 100]")
    plt.plot(200 + 100j, "r", label="200+100j")
    plt.legend(loc="lower right", fontsize=10)
    plt.title("Matplotlib Smith Chart Projection")

    export_path = os.path.join(chart_dir, "short_vmeijin_1.pdf")
    plt.savefig(export_path, format="pdf")
    plt.close()


def test_smith_chart_plot2(setup_environment):
    """Test for plotting data on a Smith chart using various configurations."""
    s11, s22, chart_dir = setup_environment

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    s = "equipoints=22"
    plt.plot(s11, markevery=1, label=s, equipoints=22, domain=R_DOMAIN)
    s += "\nmarkevery=3"
    plt.plot(s22, markevery=3, label=s, equipoints=22, domain=R_DOMAIN)
    plt.legend(loc="lower right", fontsize=10)
    plt.title("Matplotlib Smith Chart Projection")

    export_path = os.path.join(chart_dir, "short_vmeijin_2.pdf")
    plt.savefig(export_path, format="pdf")
    plt.close()
