"""Test script for the 'which' parameter."""

import matplotlib.pyplot as plt
import numpy as np

# Add the local modules to path
import sys

sys.path.insert(0, "/home/claude")

import pysmithchart
from pysmithchart import ADMITTANCE_DOMAIN, IMPEDANCE_DOMAIN

print("Testing 'which' parameter functionality...\n")

# Test data
admittances = [0.5 + 0.5j, 1.0 + 0.0j, 0.5 - 0.5j, 2.0 + 1.0j]

# Test 1: which='impedance' (default behavior)
print("Test 1: which='impedance'")
try:
    fig, ax = plt.subplots(subplot_kw={"projection": "smith", "which": "impedance"})
    # Check that impedance grid is enabled
    assert ax.scParams["grid.Z.major.enable"] == True
    assert ax.scParams["grid.Z.minor.enable"] == True
    assert ax.scParams["grid.Y.major.enable"] == False
    assert ax.scParams["grid.Y.minor.enable"] == False
    print("  ✓ Impedance grid enabled correctly")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 2: which='admittance'
print("\nTest 2: which='admittance'")
try:
    fig, ax = plt.subplots(subplot_kw={"projection": "smith", "which": "admittance"})
    # Check that admittance grid is enabled
    assert ax.scParams["grid.Z.major.enable"] == False
    assert ax.scParams["grid.Z.minor.enable"] == False
    assert ax.scParams["grid.Y.major.enable"] == True
    assert ax.scParams["grid.Y.minor.enable"] == True
    print("  ✓ Admittance grid enabled correctly")

    # Test plotting
    ax.plot(admittances, "o-", domain=ADMITTANCE_DOMAIN)
    print("  ✓ Plotting works")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 3: which='both'
print("\nTest 3: which='both'")
try:
    fig, ax = plt.subplots(subplot_kw={"projection": "smith", "which": "both"})
    # Check that both grids are enabled
    assert ax.scParams["grid.Z.major.enable"] == True
    assert ax.scParams["grid.Z.minor.enable"] == True
    assert ax.scParams["grid.Y.major.enable"] == True
    assert ax.scParams["grid.Y.minor.enable"] == True
    print("  ✓ Both grids enabled correctly")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 4: Invalid 'which' value
print("\nTest 4: Invalid 'which' value")
try:
    fig, ax = plt.subplots(subplot_kw={"projection": "smith", "which": "invalid"})
    print("  ✗ Should have raised ValueError")
    plt.close(fig)
except ValueError as e:
    print(f"  ✓ Correctly raised ValueError: {e}")
except Exception as e:
    print(f"  ✗ Unexpected error: {e}")

# Test 5: Default behavior (no 'which' parameter)
print("\nTest 5: Default behavior (no 'which' parameter)")
try:
    fig, ax = plt.subplots(subplot_kw={"projection": "smith"})
    # Default should have impedance grid enabled, admittance disabled
    # (based on SC_DEFAULT_PARAMS)
    assert ax.scParams["grid.Z.major.enable"] == True
    assert ax.scParams["grid.Y.major.enable"] == False
    print("  ✓ Default configuration correct")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 6: Combining 'which' with other parameters
print("\nTest 6: Combining 'which' with smith_params")
try:
    config = {"grid.Y.major.color": "blue"}
    fig, ax = plt.subplots(subplot_kw={"projection": "smith", "which": "admittance", "smith_params": config})
    assert ax.scParams["grid.Y.major.enable"] == True
    assert ax.scParams["grid.Y.major.color"] == "blue"
    print("  ✓ Combined parameters work correctly")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 7: Create actual figures
print("\nTest 7: Create test figures")
try:
    # Create properly with 'which' in constructor
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection="smith", which="impedance")
    ax2 = fig.add_subplot(132, projection="smith", which="admittance")
    ax3 = fig.add_subplot(133, projection="smith", which="both")

    # Plot on each
    impedances = np.array(admittances) * 50  # Convert to unnormalized
    ax1.plot(impedances, "o-", domain=IMPEDANCE_DOMAIN)
    ax1.set_title("Impedance Chart")

    ax2.plot(admittances, "o-", domain=ADMITTANCE_DOMAIN)
    ax2.set_title("Admittance Chart")

    ax3.plot(admittances, "o-", domain=ADMITTANCE_DOMAIN)
    ax3.set_title("Both Grids")

    plt.tight_layout()
    plt.savefig("/home/claude/which_parameter_test.png", dpi=100, bbox_inches="tight")
    print("  ✓ Test figure saved to which_parameter_test.png")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback

    traceback.print_exc()

print("\nAll tests completed!")
