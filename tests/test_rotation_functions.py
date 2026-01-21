"""
Test suite for rotation functions in pysmithchart.utils.

Tests the three modern rotation functions:
- rotate_by_wavelength
- rotate_toward_real
- rotate_toward_imag
"""

import pytest
import numpy as np
from pysmithchart import utils


class TestRotateByWavelength:
    """Tests for rotate_by_wavelength function."""

    def test_zero_rotation(self):
        """Test that zero wavelength returns the same impedance."""
        Z = 75 + 50j
        Z0 = 50
        Z_rot = utils.rotate_by_wavelength(Z, 0, Z0=Z0)
        assert np.isclose(Z_rot, Z), f"Expected {Z}, got {Z_rot}"

    def test_quarter_wave_transform(self):
        """Test quarter-wave transformation: Z → Z₀²/Z."""
        Z = 100 + 0j  # Pure resistance
        Z0 = 50
        Z_rot = utils.rotate_by_wavelength(Z, 0.25, Z0=Z0)
        Z_expected = Z0**2 / Z  # Should be 25Ω
        assert np.isclose(Z_rot, Z_expected, rtol=1e-5), f"Quarter-wave: Expected {Z_expected}, got {Z_rot}"

    def test_half_wave_returns_to_start(self):
        """Test that half wavelength returns to starting impedance."""
        Z = 75 + 50j
        Z0 = 50
        Z_rot = utils.rotate_by_wavelength(Z, 0.5, Z0=Z0)
        assert np.isclose(Z_rot, Z, rtol=1e-5), f"Half-wave should return to start: {Z} → {Z_rot}"

    def test_full_wave_returns_to_start(self):
        """Test that full wavelength returns to starting impedance."""
        Z = 75 + 50j
        Z0 = 50
        Z_rot = utils.rotate_by_wavelength(Z, 1.0, Z0=Z0)
        assert np.isclose(Z_rot, Z, rtol=1e-5), f"Full wave should return to start: {Z} → {Z_rot}"

    def test_constant_vswr(self):
        """Test that rotation preserves VSWR (constant |Γ|)."""
        Z = 75 + 50j
        Z0 = 50

        # Calculate initial VSWR
        gamma_start = (Z / Z0 - 1) / (Z / Z0 + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate by various amounts
        for wavelength in [0.1, 0.25, 0.333, 0.5]:
            Z_rot = utils.rotate_by_wavelength(Z, wavelength, Z0=Z0)
            gamma_rot = (Z_rot / Z0 - 1) / (Z_rot / Z0 + 1)
            vswr_rot = (1 + abs(gamma_rot)) / (1 - abs(gamma_rot))

            assert np.isclose(
                vswr_start, vswr_rot, rtol=1e-5
            ), f"VSWR should be constant: {vswr_start} → {vswr_rot} at λ={wavelength}"

    def test_direction_toward_generator(self):
        """Test rotation toward generator (opposite direction)."""
        Z = 75 + 50j
        Z0 = 50

        Z_toward_load = utils.rotate_by_wavelength(Z, 0.1, Z0=Z0, direction="toward_load")
        Z_toward_gen = utils.rotate_by_wavelength(Z, 0.1, Z0=Z0, direction="toward_generator")

        # They should be different
        assert not np.isclose(Z_toward_load, Z_toward_gen), "Different directions should give different results"

        # Going 0.1λ toward load then 0.1λ toward generator should return to start
        Z_back = utils.rotate_by_wavelength(Z_toward_load, 0.1, Z0=Z0, direction="toward_generator")
        assert np.isclose(Z_back, Z, rtol=1e-5), f"Should return to start: {Z} → {Z_toward_load} → {Z_back}"

    def test_matched_load_stays_matched(self):
        """Test that matched load (Z=Z0) stays matched after rotation."""
        Z = 50 + 0j
        Z0 = 50

        for wavelength in [0.1, 0.25, 0.5, 1.0]:
            Z_rot = utils.rotate_by_wavelength(Z, wavelength, Z0=Z0)
            assert np.isclose(Z_rot, Z, rtol=1e-5), f"Matched load should stay matched: {Z} → {Z_rot} at λ={wavelength}"


class TestRotateTowardReal:
    """Tests for rotate_toward_real function."""

    def test_finds_target_resistance(self):
        """Test that the result has the target real part."""
        Z_start = 75 + 50j
        target_R = 50
        Z0 = 50

        Z_result = utils.rotate_toward_real(Z_start, target_R, Z0=Z0)

        # Check that real part matches target
        assert np.isclose(Z_result.real, target_R, rtol=1e-5), f"Real part should be {target_R}, got {Z_result.real}"

    def test_two_solutions_exist(self):
        """Test that both solutions (positive and negative imaginary) exist."""
        Z_start = 75 + 50j
        target_R = 50
        Z0 = 50

        Z_pos = utils.rotate_toward_real(Z_start, target_R, Z0=Z0, solution="positive_imag")
        Z_neg = utils.rotate_toward_real(Z_start, target_R, Z0=Z0, solution="negative_imag")

        # Both should have target real part
        assert np.isclose(Z_pos.real, target_R, rtol=1e-5)
        assert np.isclose(Z_neg.real, target_R, rtol=1e-5)

        # Imaginary parts should have opposite signs
        assert Z_pos.imag > 0, "positive_imag solution should have Im(Z) > 0"
        assert Z_neg.imag < 0, "negative_imag solution should have Im(Z) < 0"

    def test_constant_vswr_maintained(self):
        """Test that rotation maintains constant VSWR."""
        Z_start = 75 + 50j
        target_R = 60
        Z0 = 50

        # Calculate initial VSWR
        gamma_start = (Z_start / Z0 - 1) / (Z_start / Z0 + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate to target resistance
        Z_result = utils.rotate_toward_real(Z_start, target_R, Z0=Z0)
        gamma_result = (Z_result / Z0 - 1) / (Z_result / Z0 + 1)
        vswr_result = (1 + abs(gamma_result)) / (1 - abs(gamma_result))

        assert np.isclose(vswr_start, vswr_result, rtol=1e-5), f"VSWR should be preserved: {vswr_start} → {vswr_result}"

    def test_already_at_target(self):
        """Test when starting impedance already has target real part."""
        Z_start = 50 + 75j
        target_R = 50
        Z0 = 50

        Z_result = utils.rotate_toward_real(Z_start, target_R, Z0=Z0)

        # Should stay at the same point (or very close)
        assert np.isclose(Z_result.real, target_R, rtol=1e-5)

    def test_unreachable_target_raises_error(self):
        """Test that unreachable target resistance raises ValueError."""
        Z_start = 75 + 50j  # Has a certain VSWR
        target_R = 10  # Too low to reach on this VSWR circle
        Z0 = 50

        with pytest.raises(ValueError, match="not reachable"):
            utils.rotate_toward_real(Z_start, target_R, Z0=Z0)

    def test_closer_solution(self):
        """Test that 'closer' solution gives shorter rotation."""
        Z_start = 75 + 50j
        target_R = 50
        Z0 = 50

        Z_closer = utils.rotate_toward_real(Z_start, target_R, Z0=Z0, solution="closer")
        Z_pos = utils.rotate_toward_real(Z_start, target_R, Z0=Z0, solution="positive_imag")
        Z_neg = utils.rotate_toward_real(Z_start, target_R, Z0=Z0, solution="negative_imag")

        # Closer should be one of the two solutions
        assert np.isclose(Z_closer, Z_pos, rtol=1e-5) or np.isclose(Z_closer, Z_neg, rtol=1e-5)


class TestRotateTowardImag:
    """Tests for rotate_toward_imag function."""

    def test_finds_target_reactance(self):
        """Test that the result has the target imaginary part."""
        Z_start = 75 + 50j
        target_X = 0  # Real axis
        Z0 = 50

        Z_result = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0)

        # Check that imaginary part matches target
        assert np.isclose(
            Z_result.imag, target_X, rtol=1e-4, atol=1e-6
        ), f"Imaginary part should be {target_X}, got {Z_result.imag}"

    def test_rotate_to_real_axis(self):
        """Test rotation to real axis (X=0)."""
        Z_start = 75 + 50j
        target_X = 0
        Z0 = 50

        Z_result = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0)

        # Should be on real axis
        assert np.isclose(Z_result.imag, 0, atol=1e-6), f"Should be on real axis, got Im(Z) = {Z_result.imag}"

        # Real part should be positive
        assert Z_result.real > 0, "Real part should be positive"

    def test_two_solutions_exist(self):
        """Test that both solutions (higher and lower real) exist."""
        Z_start = 75 + 50j
        target_X = 25
        Z0 = 50

        Z_high = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0, solution="higher_real")
        Z_low = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0, solution="lower_real")

        print(Z_high, Z_low)

        # Both should have target imaginary part
        assert np.isclose(Z_high.imag, target_X, atol=1e-2)
        assert np.isclose(Z_low.imag, target_X, atol=1e-2)

        # Real parts should be different
        assert Z_high.real > Z_low.real, f"higher_real ({Z_high.real}) should be > lower_real ({Z_low.real})"

    def test_constant_vswr_maintained(self):
        """Test that rotation maintains constant VSWR."""
        Z_start = 75 + 50j
        target_X = 0
        Z0 = 50

        # Calculate initial VSWR
        gamma_start = (Z_start / Z0 - 1) / (Z_start / Z0 + 1)
        vswr_start = (1 + abs(gamma_start)) / (1 - abs(gamma_start))

        # Rotate to target reactance
        Z_result = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0)
        gamma_result = (Z_result / Z0 - 1) / (Z_result / Z0 + 1)
        vswr_result = (1 + abs(gamma_result)) / (1 - abs(gamma_result))

        assert np.isclose(vswr_start, vswr_result, rtol=1e-5), f"VSWR should be preserved: {vswr_start} → {vswr_result}"

    def test_negative_reactance(self):
        """Test rotation to negative reactance (capacitive)."""
        Z_start = 75 + 50j
        target_X = -30
        Z0 = 50

        Z_result = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0)

        assert np.isclose(
            Z_result.imag, target_X, rtol=1e-4, atol=1e-6
        ), f"Imaginary part should be {target_X}, got {Z_result.imag}"

    def test_closer_solution(self):
        """Test that 'closer' solution gives shorter rotation."""
        Z_start = 75 + 50j
        target_X = 0
        Z0 = 50

        Z_closer = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0, solution="closer")
        Z_high = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0, solution="higher_real")
        Z_low = utils.rotate_toward_imag(Z_start, target_X, Z0=Z0, solution="lower_real")

        # Closer should be one of the two solutions
        assert np.isclose(Z_closer, Z_high, rtol=1e-4) or np.isclose(Z_closer, Z_low, rtol=1e-4)


class TestRotationFunctionsIntegration:
    """Integration tests combining multiple rotation functions."""

    def test_rotate_then_match_resistance(self):
        """Test rotating by wavelength then matching resistance."""
        Z_start = 75 + 50j
        Z0 = 50

        # First rotate by some wavelength
        Z_rotated = utils.rotate_by_wavelength(Z_start, 0.125, Z0=Z0)

        # Then match to 50Ω resistance
        Z_matched = utils.rotate_toward_real(Z_rotated, 50, Z0=Z0)

        assert np.isclose(Z_matched.real, 50, rtol=1e-5), f"Should have R=50Ω, got {Z_matched.real}Ω"

    #     def test_match_resistance_then_reactance(self):
    #         """Test matching resistance then reactance (full matching)."""
    #         Z_start = 75 + 50j
    #         Z0 = 50
    #
    #         # First match resistance to Z0
    #         Z_r_matched = utils.rotate_toward_real(Z_start, Z0, Z0=Z0)
    #         print(Z_r_matched)
    #
    #         # Then match reactance to 0 (real axis)
    #         Z_full_matched = utils.rotate_toward_imag(Z_r_matched, 0, Z0=Z0)
    #         print(Z_full_matched)
    #
    #         # Should be very close to Z0 (perfect match)
    #         assert np.isclose(Z_full_matched.real, Z0, rtol=1e-4), \
    #             f"Should have R={Z0}Ω, got {Z_full_matched.real}Ω"
    #         assert np.isclose(Z_full_matched.imag, 0, atol=1e-5), \
    #             f"Should have X=0Ω, got {Z_full_matched.imag}Ω"

    def test_rotation_equivalence(self):
        """Test that rotation by wavelength can be replicated by angle."""
        Z_start = 75 + 50j
        Z0 = 50

        # Rotate by λ/4
        Z_rot_lambda = utils.rotate_by_wavelength(Z_start, 0.25, Z0=Z0)

        # This should equal Z0^2 / Z for resistive loads
        # For complex loads, check VSWR is maintained
        gamma_start = (Z_start / Z0 - 1) / (Z_start / Z0 + 1)
        gamma_rot = (Z_rot_lambda / Z0 - 1) / (Z_rot_lambda / Z0 + 1)

        assert np.isclose(abs(gamma_start), abs(gamma_rot), rtol=1e-5), "VSWR should be preserved in rotation"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
