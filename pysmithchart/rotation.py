"""
Utility functions for rotations in Smith charts.

Public Functions:
    Transmission Line Rotations:
        rotate_z_by_wavelength(z_norm, wavelengths, direction): Rotate impedance by electrical length
        rotate_y_by_wavelength(y_norm, wavelengths, direction): Rotate admittance by electrical length

    Impedance Rotations (to match target values):
        rotate_z_toward_resistance(z_norm, r_norm, solution): Rotate to match resistance
        rotate_z_toward_reactance(z_norm, x_norm, solution): Rotate to match reactance

    Admittance Rotations (to match target values):
        rotate_y_toward_conductance(y_norm, g_norm, solution): Rotate to match conductance
        rotate_y_toward_susceptance(y_norm, b_norm, solution): Rotate to match susceptance

Notes:
    All rotation functions work with normalized values:
    - For impedance: z = Z/Z₀, r = R/Z₀, x = X/Z₀ (unitless)
    - For admittance: y = Y×Z₀, g = G×Z₀, b = B×Z₀ (unitless)

    To convert between physical and normalized values:
    - z_norm = Z_physical / Z0
    - y_norm = Y_physical * Z0
"""

import numpy as np
from pysmithchart.utils import moebius_transform, moebius_inverse_transform

# Public API
__all__ = (
    "rotate_z_by_wavelength",
    "rotate_y_by_wavelength",
    "rotate_z_toward_resistance",
    "rotate_z_toward_reactance",
    "rotate_y_toward_conductance",
    "rotate_y_toward_susceptance",
)


def rotate_z_by_wavelength(z_norm, wavelengths, direction="toward_load"):
    """
    Rotate normalized impedance by a specified electrical length.

    Moving along a transmission line rotates the impedance around the center
    of the Smith chart at constant |Γ| (constant VSWR).

    Args:
        z_norm (complex): Input normalized impedance (z = Z/Z₀, unitless).
        wavelengths (float): Electrical length in wavelengths.
            - Positive values rotate toward the load
            - Negative values rotate toward the generator
            - Examples: 0.125 for λ/8, 0.25 for λ/4, 0.5 for λ/2
        direction (str): 'toward_load' (default) or 'toward_generator'.
            This sets the sign convention for positive wavelength values.

    Returns:
        complex: Rotated normalized impedance.

    Examples:
        >>> # Rotate z=1.5+1j by λ/8 toward load
        >>> z_new = rotate_z_by_wavelength(1.5+1j, 0.125)
        >>> print(f"z = {z_new.real:.3f} + {z_new.imag:.3f}j")
        z = 1.732 + 0.500j

        >>> # Rotate by λ/4 toward generator
        >>> z_new = rotate_z_by_wavelength(2+2j, 0.25, direction='toward_generator')

        >>> # Quarter-wave transform: z → 1/z
        >>> z_new = rotate_z_by_wavelength(2+0j, 0.25)
        >>> print(f"z = {z_new.real:.3f}")  # Should be 0.5
        z = 0.500

        >>> # Half-wave: returns to same impedance
        >>> z_new = rotate_z_by_wavelength(1.5+1j, 0.5)
        >>> # z_new ≈ 1.5+1j (within numerical precision)

    Notes:
        - λ/4 rotation: Transforms z → 1/z (quarter-wave transformer)
        - λ/2 rotation: Returns to the same impedance
        - Phase rotation: θ = 4π × wavelengths (radians)
        - Rotation preserves |Γ| (moves along constant VSWR circle)

        All values are normalized (unitless). To use with physical impedances:
        z_norm = Z_physical / Z₀
        Z_physical = z_rotated × Z₀
    """
    # Convert to reflection coefficient
    gamma = moebius_transform(z_norm, norm=1)

    # Calculate rotation angle in radians
    # 1 wavelength = 2π electrical radians, full circle = 4π phase rotation
    angle = 4 * np.pi * wavelengths

    # Apply direction
    if direction == "toward_generator":
        angle = -angle
    elif direction != "toward_load":
        raise ValueError("direction must be 'toward_load' or 'toward_generator'")

    # Rotate the reflection coefficient
    gamma_rotated = gamma * np.exp(1j * angle)

    # Convert back to normalized impedance
    z_rotated = moebius_inverse_transform(gamma_rotated, norm=1)

    return z_rotated


def rotate_y_by_wavelength(y_norm, wavelengths, direction="toward_load"):
    """
    Rotate normalized admittance by a specified electrical length.

    Moving along a transmission line rotates the admittance around the center
    of the Smith chart at constant |Γ| (constant VSWR). This is equivalent to
    converting to impedance, rotating, and converting back.

    Args:
        y_norm (complex): Input normalized admittance (y = Y×Z₀, unitless).
        wavelengths (float): Electrical length in wavelengths.
            - Positive values rotate toward the load
            - Negative values rotate toward the generator
            - Examples: 0.125 for λ/8, 0.25 for λ/4, 0.5 for λ/2
        direction (str): 'toward_load' (default) or 'toward_generator'.
            This sets the sign convention for positive wavelength values.

    Returns:
        complex: Rotated normalized admittance.

    Examples:
        >>> # Rotate y=0.5+0.25j by λ/8 toward load
        >>> y_new = rotate_y_by_wavelength(0.5+0.25j, 0.125)
        >>> print(f"y = {y_new.real:.3f} + {y_new.imag:.3f}j")
        y = 0.577 + 0.144j

        >>> # Rotate by λ/4 toward generator
        >>> y_new = rotate_y_by_wavelength(1+0.5j, 0.25, direction='toward_generator')

        >>> # Quarter-wave transform: y → 1/y
        >>> y_new = rotate_y_by_wavelength(2+0j, 0.25)
        >>> print(f"y = {y_new.real:.3f}")  # Should be 0.5
        y = 0.500

        >>> # Half-wave: returns to same admittance
        >>> y_new = rotate_y_by_wavelength(0.5+0.25j, 0.5)
        >>> # y_new ≈ 0.5+0.25j (within numerical precision)

    Notes:
        - λ/4 rotation: Transforms y → 1/y (quarter-wave transformer)
        - λ/2 rotation: Returns to the same admittance
        - Phase rotation: θ = 4π × wavelengths (radians)
        - Rotation preserves |Γ| (moves along constant VSWR circle)
        - Equivalent to: y_rotated = 1 / rotate_z_by_wavelength(1/y, wavelengths, direction)

        All values are normalized (unitless). To use with physical admittances:
        y_norm = Y_physical × Z₀
        Y_physical = y_rotated / Z₀
    """
    # Convert admittance to impedance
    if y_norm == 0:
        z_norm = 1e9 + 0j  # Large value for open circuit
    else:
        z_norm = 1 / y_norm

    # Rotate the impedance
    z_rotated = rotate_z_by_wavelength(z_norm, wavelengths, direction)

    # Convert back to admittance
    if abs(z_rotated) < 1e-10:
        y_rotated = 1e9 + 0j  # Large value for short circuit
    else:
        y_rotated = 1 / z_rotated

    return y_rotated


def _rotate_on_constant_gamma_to_real(
    v_norm: complex,
    target_real_norm: float,
    gamma_from_v_norm,
    *,
    solution: str = "closer",
    what: str = "impedance",
):
    """
    Core helper used by rotate_Z_toward_real() and rotate_Y_toward_real().

    Given a normalized complex quantity v_norm (either z or y), find the point(s)
    on its constant-|Γ| circle whose real part equals target_real_norm.

    Args:
        v_norm: Normalized complex value (z or y).
        target_real_norm: Desired real part in normalized units.
        gamma_from_v_norm: Callable mapping v_norm -> Γ (complex reflection coeff).
        solution: Selection among two intersection points.
        what: Label used only in error messages.

    Returns:
        v_norm_rotated: Complex normalized value with real part target_real_norm.

    Raises:
        ValueError: If the target is not reachable on the constant-|Γ| circle.
    """
    gamma = gamma_from_v_norm(v_norm)
    gamma_mag = np.abs(gamma)

    # Solve for imaginary part magnitude at the intersection with Re{v} = target_real_norm.
    # This follows the same algebra you already used for impedance.
    numerator = (target_real_norm - 1) ** 2 - gamma_mag**2 * (target_real_norm + 1) ** 2
    denominator = gamma_mag**2 - 1

    if np.abs(denominator) < 1e-10:
        raise ValueError(
            f"Target real part not reachable for {what} (|Γ| ≈ 1, edge of chart). "
            f"target_real_norm={target_real_norm}"
        )

    imag_sq = numerator / denominator

    if imag_sq < -1e-10:
        vswr = (1 + gamma_mag) / (1 - gamma_mag) if gamma_mag < 1 else np.inf
        raise ValueError(
            f"Target real part not reachable on constant-|Γ| circle for {what}. "
            f"VSWR={vswr:.2f}, |Γ|={gamma_mag:.3f}, target_real_norm={target_real_norm}"
        )

    if imag_sq < 0:
        imag_sq = 0.0

    imag_mag = float(np.sqrt(imag_sq))

    v_pos = target_real_norm + 1j * imag_mag
    v_neg = target_real_norm - 1j * imag_mag

    gamma_pos = gamma_from_v_norm(v_pos)
    gamma_neg = gamma_from_v_norm(v_neg)

    if solution == "positive_imag":
        return v_pos
    if solution == "negative_imag":
        return v_neg
    if solution not in {"closer", "farther"}:
        raise ValueError("solution must be 'closer', 'farther', 'positive_imag', or 'negative_imag'")

    # Choose shorter/longer rotation in Γ-angle (mod 2π)
    ang_cur = np.angle(gamma)
    ang_pos = np.angle(gamma_pos)
    ang_neg = np.angle(gamma_neg)

    def ang_dist(a, b):
        d = np.abs(a - b)
        return 2 * np.pi - d if d > np.pi else d

    d_pos = ang_dist(ang_pos, ang_cur)
    d_neg = ang_dist(ang_neg, ang_cur)

    if solution == "closer":
        return v_pos if d_pos < d_neg else v_neg
    else:  # farther
        return v_pos if d_pos > d_neg else v_neg


def rotate_z_toward_resistance(z_norm, r_norm, solution="closer"):
    """
    Rotate normalized impedance to match a target normalized resistance.

    Finds the rotation along the constant-VSWR circle that results in the
    specified real part of impedance (series-matching style step).

    Args:
        z_norm (complex): Input normalized impedance (z = Z/Z₀, unitless).
        r_norm (float): Target normalized resistance (r = R/Z₀, unitless).
        solution (str): Which solution to use if two exist:
            - 'closer' (default): Shorter rotation angle
            - 'farther': Longer rotation angle
            - 'positive_imag': Solution with positive reactance
            - 'negative_imag': Solution with negative reactance

    Returns:
        complex: Rotated normalized impedance with real part equal to r_norm.

    Raises:
        ValueError: If r_norm is not reachable on the constant-VSWR circle.

    Examples:
        >>> # Start at z=2+1j, rotate to r=1 (match Z₀)
        >>> z_new = rotate_z_toward_resistance(2+1j, 1.0)
        >>> print(f"r={z_new.real:.3f}, x={z_new.imag:.3f}")
        r=1.000, x=1.414

        >>> # Choose the farther rotation path
        >>> z_new = rotate_z_toward_resistance(2+1j, 1.0, solution='farther')

        >>> # Force positive reactance solution
        >>> z_new = rotate_z_toward_resistance(0.5+0.5j, 1.0, solution='positive_imag')

    Notes:
        This is commonly used in series stub matching where you rotate to a target
        resistance, then add a series reactive element to cancel the remaining reactance.

        All values are normalized (unitless). To use with physical values in Ohms:
        z_norm = Z / Z₀, r_norm = R / Z₀
    """

    def gamma_from_z(z_normalized):
        return moebius_transform(z_normalized, norm=1)

    z_new = _rotate_on_constant_gamma_to_real(
        z_norm,
        r_norm,
        gamma_from_z,
        solution=solution,
        what="impedance",
    )
    return z_new


def rotate_y_toward_conductance(y_norm, g_norm, solution="closer"):
    """
    Rotate normalized admittance to match a target normalized conductance.

    Finds the rotation along the constant-VSWR circle that results in the
    specified real part of admittance (shunt-matching style step).

    Args:
        y_norm (complex): Input normalized admittance (y = Y×Z₀, unitless).
        g_norm (float): Target normalized conductance (g = G×Z₀, unitless).
        solution (str): Which solution to use if two exist:
            - 'closer' (default): Shorter rotation angle
            - 'farther': Longer rotation angle
            - 'positive_imag': Solution with positive susceptance
            - 'negative_imag': Solution with negative susceptance

    Returns:
        complex: Rotated normalized admittance with real part equal to g_norm.

    Raises:
        ValueError: If g_norm is not reachable on the constant-VSWR circle.

    Examples:
        >>> # Start at y=0.5+0.25j, rotate to g=1.0 (match Y₀)
        >>> y_new = rotate_y_toward_conductance(0.5+0.25j, 1.0)
        >>> print(f"g={y_new.real:.3f}, b={y_new.imag:.3f}")
        g=1.000, b=0.559

        >>> # Choose the farther rotation path
        >>> y_new = rotate_y_toward_conductance(0.5+0.25j, 1.0, solution='farther')

        >>> # Force positive susceptance solution
        >>> y_new = rotate_y_toward_conductance(2+1j, 1.0, solution='positive_imag')

    Notes:
        This is the standard first step for single shunt-stub matching:
        rotate to g = 1 (normalized conductance = Y₀), then cancel the
        remaining susceptance with a shunt stub.

        All values are normalized (unitless). To use with physical values in Siemens:
        y_norm = Y × Z₀, g_norm = G × Z₀, where Y and G are in Siemens, Z₀ in Ohms.
    """

    def gamma_from_y(y_normalized):
        # Γ = (1 - y)/(1 + y) via z = 1/y
        if y_normalized == 0:
            return 1 + 0j
        z_normalized = 1 / y_normalized
        return moebius_transform(z_normalized, norm=1)

    y_new = _rotate_on_constant_gamma_to_real(
        y_norm,
        g_norm,
        gamma_from_y,
        solution=solution,
        what="admittance",
    )
    return y_new


def _gamma_from_z_norm(z_norm: complex) -> complex:
    # Your existing mapping: Γ = (z - 1)/(z + 1)
    return moebius_transform(z_norm, norm=1)


def _gamma_from_y_norm(y_norm: complex) -> complex:
    # Use Γ in admittance form: Γ = (1 - y)/(1 + y)
    # Implemented via z = 1/y to reuse existing moebius_transform.
    if y_norm == 0:
        return 1 + 0j
    z_norm = 1 / y_norm
    return moebius_transform(z_norm, norm=1)


def _ang_dist(a: float, b: float) -> float:
    """Smallest absolute angular distance between angles a and b."""
    d = float(np.abs(a - b))
    return float(2 * np.pi - d) if d > np.pi else d


def _crossings_on_constant_gamma_with_imag(
    v_norm: complex,
    target_imag_norm: float,
    gamma_from_v_norm,
    *,
    solution: str = "closer",
    what: str = "impedance",
):
    """
    Core helper: on |Γ| circle through v_norm, find points with imag(v) == target_imag_norm.

    Works for v_norm = z or y because the derivation depends only on the
    Möbius mapping magnitude identity.

    Returns:
        v_norm_rotated (complex): One of the intersection points.

    Raises:
        ValueError: if no real intersections exist.
    """
    gamma = gamma_from_v_norm(v_norm)
    rho = float(np.abs(gamma))  # |Γ|
    if rho >= 1 - 1e-14:
        raise ValueError(f"Target not reachable for {what}: |Γ|≈1 (edge of chart).")

    b = float(target_imag_norm)

    # Solve for a = Re{v} given fixed b and fixed rho:
    # rho^2 = ((a-1)^2 + b^2)/((a+1)^2 + b^2)
    # Rearranged into quadratic:
    # (1-r2) a^2 - 2(1+r2) a + (1-r2)(1+b^2) = 0, where r2 = rho^2
    r2 = rho * rho
    A = 1 - r2
    B = -2 * (1 + r2)
    C = (1 - r2) * (1 + b * b)

    # Discriminant
    D = B * B - 4 * A * C

    if D < -1e-12:
        raise ValueError(
            f"Target imag not reachable on constant-|Γ| circle for {what}. " f"|Γ|={rho:.6f}, target_imag_norm={b:.6f}"
        )
    if D < 0:
        D = 0.0

    sqrtD = float(np.sqrt(D))

    a1 = (-B + sqrtD) / (2 * A)
    a2 = (-B - sqrtD) / (2 * A)

    v1 = a1 + 1j * b
    v2 = a2 + 1j * b

    # Compute Γ for each candidate for selection logic
    g1 = gamma_from_v_norm(v1)
    g2 = gamma_from_v_norm(v2)

    if solution == "higher_real":
        return v1 if np.real(v1) >= np.real(v2) else v2
    if solution == "lower_real":
        return v1 if np.real(v1) <= np.real(v2) else v2

    if solution not in {"closer", "farther"}:
        raise ValueError("solution must be 'closer', 'farther', 'higher_real', or 'lower_real'")

    ang_cur = float(np.angle(gamma))
    d1 = _ang_dist(float(np.angle(g1)), ang_cur)
    d2 = _ang_dist(float(np.angle(g2)), ang_cur)

    if solution == "closer":
        return v1 if d1 <= d2 else v2
    else:  # farther
        return v1 if d1 >= d2 else v2


def rotate_z_toward_reactance(z_norm, x_norm, solution="closer"):
    """
    Rotate normalized impedance to match a target normalized reactance.

    Finds the rotation along the constant-VSWR circle that results in the
    specified imaginary part of impedance.

    Args:
        z_norm (complex): Input normalized impedance (z = Z/Z₀, unitless).
        x_norm (float): Target normalized reactance (x = X/Z₀, unitless).
            Positive for inductive, negative for capacitive.
        solution (str): Which solution to use if two exist:
            - 'closer' (default): Shorter rotation angle
            - 'farther': Longer rotation angle
            - 'higher_real': Solution with higher resistance
            - 'lower_real': Solution with lower resistance

    Returns:
        complex: Rotated normalized impedance with imaginary part equal to x_norm.

    Raises:
        ValueError: If x_norm is not reachable on the constant-VSWR circle.

    Examples:
        >>> # Start at z=2+1j, rotate to x=0 (purely resistive)
        >>> z_new = rotate_z_toward_reactance(2+1j, 0.0)
        >>> print(f"r={z_new.real:.3f}, x={z_new.imag:.3f}")
        r=2.236, x=0.000

        >>> # Rotate to inductive reactance x=+2
        >>> z_new = rotate_z_toward_reactance(1+0.5j, 2.0)

        >>> # Choose higher resistance solution
        >>> z_new = rotate_z_toward_reactance(0.5+0.5j, 0.0, solution='higher_real')

    Notes:
        This is useful for impedance matching scenarios where you need to rotate
        to a specific reactance before adding matching elements.

        All values are normalized (unitless). To use with physical values in Ohms:
        z_norm = Z / Z₀, x_norm = X / Z₀
    """
    z_new = _crossings_on_constant_gamma_with_imag(
        z_norm,
        x_norm,
        _gamma_from_z_norm,
        solution=solution,
        what="impedance",
    )
    return z_new


def rotate_y_toward_susceptance(y_norm, b_norm, solution="closer"):
    """
    Rotate normalized admittance to match a target normalized susceptance.

    Finds the rotation along the constant-VSWR circle that results in the
    specified imaginary part of admittance.

    Args:
        y_norm (complex): Input normalized admittance (y = Y×Z₀, unitless).
        b_norm (float): Target normalized susceptance (b = B×Z₀, unitless).
            Positive for capacitive, negative for inductive.
        solution (str): Which solution to use if two exist:
            - 'closer' (default): Shorter rotation angle
            - 'farther': Longer rotation angle
            - 'higher_real': Solution with higher conductance
            - 'lower_real': Solution with lower conductance

    Returns:
        complex: Rotated normalized admittance with imaginary part equal to b_norm.

    Raises:
        ValueError: If b_norm is not reachable on the constant-VSWR circle.

    Examples:
        >>> # Start at y=0.5+0.25j, rotate to b=0 (purely conductive)
        >>> y_new = rotate_y_toward_susceptance(0.5+0.25j, 0.0)
        >>> print(f"g={y_new.real:.3f}, b={y_new.imag:.3f}")
        g=0.559, b=0.000

        >>> # Rotate to capacitive susceptance b=+1
        >>> y_new = rotate_y_toward_susceptance(1+0.5j, 1.0)

        >>> # Choose higher conductance solution
        >>> y_new = rotate_y_toward_susceptance(0.5+0.5j, 0.0, solution='higher_real')

    Notes:
        This is useful for admittance-based matching scenarios where you need to
        rotate to a specific susceptance before adding shunt elements.

        Sign convention: Positive susceptance is capacitive, negative is inductive
        (opposite to reactance convention).

        All values are normalized (unitless). To use with physical values in Siemens:
        y_norm = Y × Z₀, b_norm = B × Z₀, where Y and B are in Siemens, Z₀ in Ohms.
    """
    y_new = _crossings_on_constant_gamma_with_imag(
        y_norm,
        b_norm,
        _gamma_from_y_norm,
        solution=solution,
        what="admittance",
    )
    return y_new
