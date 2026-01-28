"""
Test suite for rotation functions in pysmithchart.

Tests all rotation functions with normalized coordinates:
- rotate_z_by_wavelength / rotate_y_by_wavelength
- rotate_z_toward_resistance / rotate_z_toward_reactance
- rotate_y_toward_conductance / rotate_y_toward_susceptance
"""

import pytest
import numpy as np
from pysmithchart.rotation import (
    rotate_z_by_wavelength,
    rotate_y_by_wavelength,
    rotate_z_toward_resistance,
    rotate_z_toward_reactance,
    rotate_y_toward_conductance,
    rotate_y_toward_susceptance,
)


class TestRotateZByWavelength:
    """Tests for rotate_z_by_wavelength function."""

    def test_zero_rotation(self):
        """Test that zero wavelength returns the same impedance."""
        z = 1.5 + 1.0j  # Normalized impedance
        z_rot = rotate_z_by_wavelength(z, 0)
        assert np.isclose(z_rot, z), f"Expected {z}, got {z_rot}"

    def test_quarter_wave_transform(self):
        """Test quarter-wave transformation: z → 1/z."""
        z = 2.0 + 0j  # Pure normalized resistance
        z_rot = rotate_z_by_wavelength(z, 0.25)
        z_expected = 1 / z  # Should be 0.5
        assert np.isclose(z_rot, z_expected, rtol=1e-5), f"Quarter-wave: Expected {z_expected}, got {z_rot}"

    def test_half_wave_returns_to_start(self):
        """Test that half wavelength returns to starting impedance."""
        z = 1.5 + 1.0j
        z_rot = rotate_z_by_wavelength(z, 0.5)
        assert np.isclose(z_rot, z, rtol=1e-5), f"Half-wave should return to start: {z} → {z_rot}"

    def test_full_wave_returns_to_start(self):
        """Test that full wavelength returns to starting impedance."""
        z = 1.5 + 1.0j
        z_rot = rotate_z_by_wavelength(z, 1.0)
        assert np.isclose(z_rot, z, rtol=1e-5), f"Full wave should return to start: {z} → {z_rot}"

    def test_constant_vswr(self):
        """Test that rotation preserves VSWR (constant |Γ|)."""
        z = 1.5 + 1.0j

        # Calculate initial VSWR
        gamma_start = (z - 1) / (z + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate by various amounts
        for wavelength in [0.1, 0.25, 0.333, 0.5]:
            z_rot = rotate_z_by_wavelength(z, wavelength)
            gamma_rot = (z_rot - 1) / (z_rot + 1)
            vswr_rot = (1 + abs(gamma_rot)) / (1 - abs(gamma_rot))

            assert np.isclose(
                vswr_start, vswr_rot, rtol=1e-5
            ), f"VSWR should be constant: {vswr_start} → {vswr_rot} at λ={wavelength}"

    def test_direction_toward_generator(self):
        """Test rotation toward generator (opposite direction)."""
        z = 1.5 + 1.0j

        z_toward_load = rotate_z_by_wavelength(z, 0.1, direction="toward_load")
        z_toward_gen = rotate_z_by_wavelength(z, 0.1, direction="toward_generator")

        # They should be different
        assert not np.isclose(z_toward_load, z_toward_gen), "Different directions should give different results"

        # Going 0.1λ toward load then 0.1λ toward generator should return to start
        z_back = rotate_z_by_wavelength(z_toward_load, 0.1, direction="toward_generator")
        assert np.isclose(z_back, z, rtol=1e-5), f"Should return to start: {z} → {z_toward_load} → {z_back}"

    def test_matched_load_stays_matched(self):
        """Test that matched load (z=1) stays matched after rotation."""
        z = 1.0 + 0j

        for wavelength in [0.1, 0.25, 0.5, 1.0]:
            z_rot = rotate_z_by_wavelength(z, wavelength)
            assert np.isclose(z_rot, z, rtol=1e-5), f"Matched load should stay matched: {z} → {z_rot} at λ={wavelength}"


class TestRotateYByWavelength:
    """Tests for rotate_y_by_wavelength function."""

    def test_zero_rotation(self):
        """Test that zero wavelength returns the same admittance."""
        y = 0.5 + 0.25j  # Normalized admittance
        y_rot = rotate_y_by_wavelength(y, 0)
        assert np.isclose(y_rot, y), f"Expected {y}, got {y_rot}"

    def test_quarter_wave_transform(self):
        """Test quarter-wave transformation: y → 1/y."""
        y = 2.0 + 0j  # Pure normalized conductance
        y_rot = rotate_y_by_wavelength(y, 0.25)
        y_expected = 1 / y  # Should be 0.5
        assert np.isclose(y_rot, y_expected, rtol=1e-5), f"Quarter-wave: Expected {y_expected}, got {y_rot}"

    def test_half_wave_returns_to_start(self):
        """Test that half wavelength returns to starting admittance."""
        y = 0.5 + 0.25j
        y_rot = rotate_y_by_wavelength(y, 0.5)
        assert np.isclose(y_rot, y, rtol=1e-5), f"Half-wave should return to start: {y} → {y_rot}"

    def test_full_wave_returns_to_start(self):
        """Test that full wavelength returns to starting admittance."""
        y = 0.5 + 0.25j
        y_rot = rotate_y_by_wavelength(y, 1.0)
        assert np.isclose(y_rot, y, rtol=1e-5), f"Full wave should return to start: {y} → {y_rot}"

    def test_constant_vswr(self):
        """Test that rotation preserves VSWR (constant |Γ|)."""
        y = 0.5 + 0.25j

        # Convert to impedance to calculate Γ
        z = 1 / y
        gamma_start = (z - 1) / (z + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate by various amounts
        for wavelength in [0.1, 0.25, 0.333, 0.5]:
            y_rot = rotate_y_by_wavelength(y, wavelength)
            z_rot = 1 / y_rot
            gamma_rot = (z_rot - 1) / (z_rot + 1)
            vswr_rot = (1 + abs(gamma_rot)) / (1 - abs(gamma_rot))

            assert np.isclose(
                vswr_start, vswr_rot, rtol=1e-5
            ), f"VSWR should be constant: {vswr_start} → {vswr_rot} at λ={wavelength}"

    def test_matched_load_stays_matched(self):
        """Test that matched load (y=1) stays matched after rotation."""
        y = 1.0 + 0j

        for wavelength in [0.1, 0.25, 0.5, 1.0]:
            y_rot = rotate_y_by_wavelength(y, wavelength)
            assert np.isclose(y_rot, y, rtol=1e-5), f"Matched load should stay matched: {y} → {y_rot} at λ={wavelength}"

    def test_equivalence_with_impedance_rotation(self):
        """Test that y rotation is equivalent to z rotation via reciprocal."""
        y = 0.5 + 0.25j
        wavelength = 0.125

        # Rotate admittance directly
        y_rot_direct = rotate_y_by_wavelength(y, wavelength)

        # Rotate via impedance: y → z → rotate → y
        z = 1 / y
        z_rot = rotate_z_by_wavelength(z, wavelength)
        y_rot_indirect = 1 / z_rot

        assert np.isclose(
            y_rot_direct, y_rot_indirect, rtol=1e-5
        ), f"Direct and indirect rotation should match: {y_rot_direct} vs {y_rot_indirect}"


class TestRotateZTowardResistance:
    """Tests for rotate_z_toward_resistance function."""

    def test_finds_target_resistance(self):
        """Test that the result has the target real part."""
        z_start = 1.5 + 1.0j
        r_target = 1.0

        z_result = rotate_z_toward_resistance(z_start, r_target)

        # Check that real part matches target
        assert np.isclose(z_result.real, r_target, rtol=1e-5), f"Real part should be {r_target}, got {z_result.real}"

    def test_two_solutions_exist(self):
        """Test that both solutions (positive and negative imaginary) exist."""
        z_start = 1.5 + 1.0j
        r_target = 1.0

        z_pos = rotate_z_toward_resistance(z_start, r_target, solution="positive_imag")
        z_neg = rotate_z_toward_resistance(z_start, r_target, solution="negative_imag")

        # Both should have target real part
        assert np.isclose(z_pos.real, r_target, rtol=1e-5)
        assert np.isclose(z_neg.real, r_target, rtol=1e-5)

        # Imaginary parts should have opposite signs
        assert z_pos.imag > 0, "positive_imag solution should have Im(z) > 0"
        assert z_neg.imag < 0, "negative_imag solution should have Im(z) < 0"

    def test_constant_vswr_maintained(self):
        """Test that rotation maintains constant VSWR."""
        z_start = 1.5 + 1.0j
        r_target = 1.2

        # Calculate initial VSWR
        gamma_start = (z_start - 1) / (z_start + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate to target resistance
        z_result = rotate_z_toward_resistance(z_start, r_target)
        gamma_result = (z_result - 1) / (z_result + 1)
        vswr_result = (1 + abs(gamma_result)) / (1 - abs(gamma_result))

        assert np.isclose(vswr_start, vswr_result, rtol=1e-5), f"VSWR should be preserved: {vswr_start} → {vswr_result}"

    def test_already_at_target(self):
        """Test when starting impedance already has target real part."""
        z_start = 1.0 + 1.5j
        r_target = 1.0

        z_result = rotate_z_toward_resistance(z_start, r_target)

        # Should stay at the same point (or very close)
        assert np.isclose(z_result.real, r_target, rtol=1e-5)

    def test_unreachable_target_raises_error(self):
        """Test that unreachable target resistance raises ValueError."""
        z_start = 1.5 + 1.0j  # Has a certain VSWR
        r_target = 0.2  # Too low to reach on this VSWR circle

        with pytest.raises(ValueError, match="not reachable"):
            rotate_z_toward_resistance(z_start, r_target)

    def test_closer_solution(self):
        """Test that 'closer' solution gives shorter rotation."""
        z_start = 1.5 + 1.0j
        r_target = 1.0

        z_closer = rotate_z_toward_resistance(z_start, r_target, solution="closer")
        z_pos = rotate_z_toward_resistance(z_start, r_target, solution="positive_imag")
        z_neg = rotate_z_toward_resistance(z_start, r_target, solution="negative_imag")

        # Closer should be one of the two solutions
        assert np.isclose(z_closer, z_pos, rtol=1e-5) or np.isclose(z_closer, z_neg, rtol=1e-5)


class TestRotateZTowardReactance:
    """Tests for rotate_z_toward_reactance function."""

    def test_finds_target_reactance(self):
        """Test that the result has the target imaginary part."""
        z_start = 1.5 + 1.0j
        x_target = 0  # Real axis

        z_result = rotate_z_toward_reactance(z_start, x_target)

        # Check that imaginary part matches target
        assert np.isclose(
            z_result.imag, x_target, rtol=1e-4, atol=1e-6
        ), f"Imaginary part should be {x_target}, got {z_result.imag}"

    def test_rotate_to_real_axis(self):
        """Test rotation to real axis (x=0)."""
        z_start = 1.5 + 1.0j
        x_target = 0

        z_result = rotate_z_toward_reactance(z_start, x_target)

        # Should be on real axis
        assert np.isclose(z_result.imag, 0, atol=1e-6), f"Should be on real axis, got Im(z) = {z_result.imag}"

        # Real part should be positive
        assert z_result.real > 0, "Real part should be positive"

    def test_two_solutions_exist(self):
        """Test that both solutions (higher and lower real) exist."""
        z_start = 1.5 + 1.0j
        x_target = 0.5

        z_high = rotate_z_toward_reactance(z_start, x_target, solution="higher_real")
        z_low = rotate_z_toward_reactance(z_start, x_target, solution="lower_real")

        # Both should have target imaginary part
        assert np.isclose(z_high.imag, x_target, atol=1e-2)
        assert np.isclose(z_low.imag, x_target, atol=1e-2)

        # Real parts should be different
        assert z_high.real > z_low.real, f"higher_real ({z_high.real}) should be > lower_real ({z_low.real})"

    def test_constant_vswr_maintained(self):
        """Test that rotation maintains constant VSWR."""
        z_start = 1.5 + 1.0j
        x_target = 0

        # Calculate initial VSWR
        gamma_start = (z_start - 1) / (z_start + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate to target reactance
        z_result = rotate_z_toward_reactance(z_start, x_target)
        gamma_result = (z_result - 1) / (z_result + 1)
        vswr_result = (1 + abs(gamma_result)) / (1 - abs(gamma_result))

        assert np.isclose(vswr_start, vswr_result, rtol=1e-5), f"VSWR should be preserved: {vswr_start} → {vswr_result}"

    def test_negative_reactance(self):
        """Test rotation to negative reactance (capacitive)."""
        z_start = 1.5 + 1.0j
        x_target = -0.6

        z_result = rotate_z_toward_reactance(z_start, x_target)

        assert np.isclose(
            z_result.imag, x_target, rtol=1e-4, atol=1e-6
        ), f"Imaginary part should be {x_target}, got {z_result.imag}"

    def test_closer_solution(self):
        """Test that 'closer' solution gives shorter rotation."""
        z_start = 1.5 + 1.0j
        x_target = 0

        z_closer = rotate_z_toward_reactance(z_start, x_target, solution="closer")
        z_high = rotate_z_toward_reactance(z_start, x_target, solution="higher_real")
        z_low = rotate_z_toward_reactance(z_start, x_target, solution="lower_real")

        # Closer should be one of the two solutions
        assert np.isclose(z_closer, z_high, rtol=1e-4) or np.isclose(z_closer, z_low, rtol=1e-4)


class TestRotateYTowardConductance:
    """Tests for rotate_y_toward_conductance function."""

    def test_finds_target_conductance(self):
        """Test that the result has the target real part."""
        y_start = 0.5 + 0.25j
        g_target = 1.0

        y_result = rotate_y_toward_conductance(y_start, g_target)

        # Check that real part matches target
        assert np.isclose(y_result.real, g_target, rtol=1e-5), f"Real part should be {g_target}, got {y_result.real}"

    def test_two_solutions_exist(self):
        """Test that both solutions (positive and negative imaginary) exist."""
        y_start = 0.5 + 0.25j
        g_target = 1.0

        y_pos = rotate_y_toward_conductance(y_start, g_target, solution="positive_imag")
        y_neg = rotate_y_toward_conductance(y_start, g_target, solution="negative_imag")

        # Both should have target real part
        assert np.isclose(y_pos.real, g_target, rtol=1e-5)
        assert np.isclose(y_neg.real, g_target, rtol=1e-5)

        # Imaginary parts should have opposite signs
        assert y_pos.imag > 0, "positive_imag solution should have Im(y) > 0"
        assert y_neg.imag < 0, "negative_imag solution should have Im(y) < 0"

    def test_constant_vswr_maintained(self):
        """Test that rotation maintains constant VSWR."""
        y_start = 0.5 + 0.25j
        g_target = 0.8

        # Convert to impedance to calculate VSWR
        z_start = 1 / y_start
        gamma_start = (z_start - 1) / (z_start + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate to target conductance
        y_result = rotate_y_toward_conductance(y_start, g_target)
        z_result = 1 / y_result
        gamma_result = (z_result - 1) / (z_result + 1)
        vswr_result = (1 + abs(gamma_result)) / (1 - abs(gamma_result))

        assert np.isclose(vswr_start, vswr_result, rtol=1e-5), f"VSWR should be preserved: {vswr_start} → {vswr_result}"

    def test_shunt_stub_matching_setup(self):
        """Test typical shunt stub matching: rotate to g=1.0."""
        y_start = 0.6 + 0.3j
        g_target = 1.0  # Match to Y₀

        y_result = rotate_y_toward_conductance(y_start, g_target)

        # Should have g=1, then add stub to cancel b
        assert np.isclose(y_result.real, g_target, rtol=1e-5)


class TestRotateYTowardSusceptance:
    """Tests for rotate_y_toward_susceptance function."""

    def test_finds_target_susceptance(self):
        """Test that the result has the target imaginary part."""
        y_start = 0.5 + 0.25j
        b_target = 0  # Real axis (purely conductive)

        y_result = rotate_y_toward_susceptance(y_start, b_target)

        # Check that imaginary part matches target
        assert np.isclose(
            y_result.imag, b_target, rtol=1e-4, atol=1e-6
        ), f"Imaginary part should be {b_target}, got {y_result.imag}"

    def test_rotate_to_real_axis(self):
        """Test rotation to real axis (b=0)."""
        y_start = 0.5 + 0.25j
        b_target = 0

        y_result = rotate_y_toward_susceptance(y_start, b_target)

        # Should be on real axis (purely conductive)
        assert np.isclose(y_result.imag, 0, atol=1e-6), f"Should be on real axis, got Im(y) = {y_result.imag}"

        # Real part should be positive
        assert y_result.real > 0, "Real part should be positive"

    def test_two_solutions_exist(self):
        """Test that both solutions (higher and lower real) exist."""
        y_start = 0.5 + 0.25j
        b_target = 0.5

        y_high = rotate_y_toward_susceptance(y_start, b_target, solution="higher_real")
        y_low = rotate_y_toward_susceptance(y_start, b_target, solution="lower_real")

        # Both should have target imaginary part
        assert np.isclose(y_high.imag, b_target, atol=1e-2)
        assert np.isclose(y_low.imag, b_target, atol=1e-2)

        # Real parts should be different
        assert y_high.real > y_low.real, f"higher_real ({y_high.real}) should be > lower_real ({y_low.real})"

    def test_constant_vswr_maintained(self):
        """Test that rotation maintains constant VSWR."""
        y_start = 0.5 + 0.25j
        b_target = 0

        # Convert to impedance to calculate VSWR
        z_start = 1 / y_start
        gamma_start = (z_start - 1) / (z_start + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate to target susceptance
        y_result = rotate_y_toward_susceptance(y_start, b_target)
        z_result = 1 / y_result
        gamma_result = (z_result - 1) / (z_result + 1)
        vswr_result = (1 + abs(gamma_result)) / (1 - abs(gamma_result))

        assert np.isclose(vswr_start, vswr_result, rtol=1e-5), f"VSWR should be preserved: {vswr_start} → {vswr_result}"

    def test_negative_susceptance(self):
        """Test rotation to negative susceptance (inductive)."""
        y_start = 0.5 + 0.25j
        b_target = -0.3

        y_result = rotate_y_toward_susceptance(y_start, b_target)

        assert np.isclose(
            y_result.imag, b_target, rtol=1e-4, atol=1e-6
        ), f"Imaginary part should be {b_target}, got {y_result.imag}"


class TestRotationFunctionsIntegration:
    """Integration tests combining multiple rotation functions."""

    def test_rotate_then_match_resistance(self):
        """Test rotating by wavelength then matching resistance."""
        z_start = 1.5 + 1.0j

        # First rotate by some wavelength
        z_rotated = rotate_z_by_wavelength(z_start, 0.125)

        # Then match to r=1.0 (normalized)
        z_matched = rotate_z_toward_resistance(z_rotated, 1.0)

        assert np.isclose(z_matched.real, 1.0, rtol=1e-5), f"Should have r=1.0, got {z_matched.real}"

    def test_impedance_admittance_symmetry(self):
        """Test that impedance and admittance rotations are consistent."""
        z_start = 2.0 + 1.0j

        # Rotate impedance by λ/8
        z_rotated = rotate_z_by_wavelength(z_start, 0.125)

        # Convert to admittance, rotate, convert back
        y_start = 1 / z_start
        y_rotated = rotate_y_by_wavelength(y_start, 0.125)
        z_from_y = 1 / y_rotated

        assert np.isclose(
            z_rotated, z_from_y, rtol=1e-5
        ), f"Z and Y rotations should be consistent: {z_rotated} vs {z_from_y}"

    def test_match_resistance_then_reactance(self):
        """Test that matching reactance after resistance changes the resistance.

        This demonstrates that you cannot match both R and X by rotation alone.
        Rotation preserves VSWR, so after matching R, rotating to match X will
        change R back (unless already perfectly matched).
        """
        z_start = 1.5 + 1.0j

        # First match resistance to 1.0
        z_r_matched = rotate_z_toward_resistance(z_start, 1.0)
        assert np.isclose(
            z_r_matched.real, 1.0, rtol=1e-5
        ), f"Should have r=1.0 after first rotation, got {z_r_matched.real}"

        # Then match reactance to 0 (real axis)
        z_x_matched = rotate_z_toward_reactance(z_r_matched, 0)
        assert np.isclose(
            z_x_matched.imag, 0, atol=1e-5
        ), f"Should have x=0 after second rotation, got {z_x_matched.imag}"

        # The resistance should have changed (unless we started at VSWR=1)
        # This proves you can't match both by rotation alone
        assert not np.isclose(
            z_x_matched.real, 1.0, rtol=1e-3
        ), f"Resistance should have changed from 1.0, but got {z_x_matched.real}"

        # Verify VSWR is preserved throughout
        gamma_start = (z_start - 1) / (z_start + 1)
        gamma_final = (z_x_matched - 1) / (z_x_matched + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))
        vswr_final = (1 + abs(gamma_final)) / (1 - abs(gamma_final))

        assert np.isclose(vswr_start, vswr_final, rtol=1e-5), f"VSWR should be constant: {vswr_start} → {vswr_final}"

    def test_shunt_stub_matching_workflow(self):
        """Test typical single shunt stub matching workflow."""
        y_start = 0.6 + 0.3j

        # Step 1: Rotate to g=1.0 (conductance = Y₀)
        y_rotated = rotate_y_toward_conductance(y_start, 1.0)

        assert np.isclose(y_rotated.real, 1.0, rtol=1e-5), f"Should have g=1.0, got {y_rotated.real}"

        # Step 2: Add stub to cancel susceptance (just verify b exists)
        stub_susceptance = -y_rotated.imag
        y_matched = y_rotated + 1j * stub_susceptance

        # Should now be matched (y = 1+0j)
        assert np.isclose(y_matched, 1.0 + 0j, rtol=1e-5), f"After stub, should be matched: {y_matched}"

    def test_rotation_equivalence(self):
        """Test that rotation by wavelength preserves VSWR."""
        z_start = 1.5 + 1.0j

        # Rotate by λ/4
        z_rot_lambda = rotate_z_by_wavelength(z_start, 0.25)

        # Check VSWR is maintained
        gamma_start = (z_start - 1) / (z_start + 1)
        gamma_rot = (z_rot_lambda - 1) / (z_rot_lambda + 1)

        assert np.isclose(abs(gamma_start), abs(gamma_rot), rtol=1e-5), "VSWR should be preserved in rotation"

    def test_combined_z_and_y_matching(self):
        """Test using both impedance and admittance rotations together."""
        z_start = 2.0 + 1.5j

        # Rotate impedance to real axis
        z_on_real = rotate_z_toward_reactance(z_start, 0)

        # Convert to admittance and match conductance
        y = 1 / z_on_real
        y_matched = rotate_y_toward_conductance(y, 1.0)

        # Verify conductance is 1.0
        assert np.isclose(y_matched.real, 1.0, rtol=1e-5), f"Should have g=1.0, got {y_matched.real}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
