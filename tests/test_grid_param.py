# pylint:disable=unused-import
"""
Pytest tests for the Smith-chart `grid` parameter.

- Closes all figures to avoid resource leakage across tests.
- Does not write files by default
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pysmithchart  # noqa: F401  (ensures projection is registered)
from pysmithchart import Y_DOMAIN, Z_DOMAIN


@pytest.fixture(autouse=True)
def _close_figures():
    """Automatically close all Matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def admittances():
    """Provide a small set of complex admittances for plotting tests."""
    return np.array([0.5 + 0.5j, 1.0 + 0.0j, 0.5 - 0.5j, 2.0 + 1.0j])


def _assert_grid(ax, *, z_major: bool, z_minor: bool, y_major: bool, y_minor: bool):
    """Assert that Smith-chart grid enable flags match expectations."""
    assert ax.scParams["grid.Z.major.enable"] is z_major
    assert ax.scParams["grid.Z.minor.enable"] is z_minor
    assert ax.scParams["grid.Y.major.enable"] is y_major
    assert ax.scParams["grid.Y.minor.enable"] is y_minor


@pytest.mark.parametrize(
    "grid, expected",
    [
        ("impedance", (True, True, False, False)),
        ("admittance", (False, False, True, True)),
        ("both", (True, True, True, True)),
    ],
)
def test_grid_parameter_enables_expected_grids(grid, expected):
    """Grid keyword enables the correct impedance/admittance grid combinations."""
    ax = plt.subplot(111, projection="smith", grid=grid)
    zmaj, zmin, ymaj, ymin = expected
    _assert_grid(ax, z_major=zmaj, z_minor=zmin, y_major=ymaj, y_minor=ymin)


def test_grid_admittance_allows_plotting(admittances):
    """Admittance grid supports plotting with Y-domain data."""
    ax = plt.subplot(111, projection="smith", grid="admittance")
    _assert_grid(ax, z_major=False, z_minor=False, y_major=True, y_minor=True)

    # Smoke test: should not raise
    ax.plot(admittances, "o-", domain=Y_DOMAIN)


def test_grid_invalid_value_raises():
    """Invalid grid values raise a ValueError."""
    with pytest.raises(ValueError):
        plt.subplots(subplot_kw={"projection": "smith", "grid": "invalid"})


def test_grid_default_is_impedance():
    """Default Smith chart uses the impedance grid only."""
    _, ax = plt.subplots(subplot_kw={"projection": "smith"})
    assert ax.scParams["grid.Z.major.enable"] is True
    assert ax.scParams["grid.Y.major.enable"] is False


def test_grid_combined_with_smith_params():
    """Grid selection works correctly when combined with smith_style overrides."""
    ss = {"grid.Y.major.color": "blue"}
    _, ax = plt.subplots(
        subplot_kw={
            "projection": "smith",
            "grid": "admittance",
            "smith_style": ss,
        }
    )
    assert ax.scParams["grid.Y.major.enable"] is True
    assert ax.scParams["grid.Y.major.color"] == "blue"


@pytest.mark.parametrize(
    "grid, domain",
    [("impedance", Z_DOMAIN), ("admittance", Y_DOMAIN), ("both", Y_DOMAIN)],
)
def test_can_create_axes_and_plot_for_each_grid_mode(grid, domain, admittances):
    """Each grid mode can create an axis and accept a basic plot."""
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(111, projection="smith", grid=grid)

    if domain is Z_DOMAIN:
        data = np.array(1 / admittances) * 50  # unnormalized impedances
    else:
        data = admittances

    ax.plot(data, "o-", domain=domain)
    ax.set_title(f"grid={grid!r}")
