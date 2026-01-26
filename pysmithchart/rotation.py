"""
Utility functions for rotations in Smith charts.

Public Functions:
    Rotation Functions:
        rotate_by_wavelength(Z, wavelength, Z0, direction): Rotate by electrical length
        rotate_Z_toward_real(Z, target_resistance, Z0, solution): Rotate to match resistance
        rotate_Z_toward_imag(Z, target_reactance, Z0, solution): Rotate to match reactance
        rotate_Y_toward_real(Z, target_conductance, Z0, solution): Rotate to match condunctance
        rotate_Y_toward_imag(Z, target_susceptance, Z0, solution): Rotate to match susceptance

"""

import numpy as np
from pysmithchart.utils import moebius_transform, moebius_inverse_transform

# Public API
__all__ = (
    "rotate_by_wavelength",
    "rotate_Z_toward_real",
    "rotate_Z_toward_imag",
    "rotate_Y_toward_real",
    "rotate_Y_toward_imag",
)


def rotate_by_wavelength(Z, wavelength, Z0=50, direction="toward_load"):
    """
    Rotate an impedance by a specified electrical length (in wavelengths).

    Moving along a transmission line rotates the impedance around the center
    of the Smith chart at constant |Γ|.

    Args:
        Z (complex): Input impedance in Ohms.
        wavelength (float): Electrical length in wavelengths (e.g., 0.25 for λ/4).
            Positive rotates toward load, negative toward generator.
        Z0 (float): Characteristic impedance (default: 50Ω).
        direction (str): 'toward_load' (default) or 'toward_generator'.

    Returns:
        complex: Rotated impedance in Ohms.

    Examples:
        >>> # Rotate 75+50j by λ/8 toward load
        >>> Z_new = rotate_by_wavelength(75+50j, 0.125, Z0=50)

        >>> # Rotate by λ/4 toward generator
        >>> Z_new = rotate_by_wavelength(75+50j, 0.25, direction='toward_generator')

    Notes:
        - λ/4 rotation transforms Z → Z0²/Z
        - λ/2 rotation returns to the same impedance
        - Phase rotation: θ = 4π * wavelength (radians)
    """
    # Convert to normalized impedance
    z = Z / Z0

    # Convert to reflection coefficient
    gamma = moebius_transform(z, norm=1)

    # Calculate rotation angle in radians
    # 1 wavelength = 2π electrical radians, full circle = 4π phase rotation
    angle = 4 * np.pi * wavelength

    # Apply direction
    if direction == "toward_generator":
        angle = -angle
    elif direction != "toward_load":
        raise ValueError("direction must be 'toward_load' or 'toward_generator'")

    # Rotate the reflection coefficient
    gamma_rotated = gamma * np.exp(1j * angle)

    # Convert back to normalized impedance
    z_rotated = moebius_inverse_transform(gamma_rotated, norm=1)

    # Convert back to Ohms
    return z_rotated * Z0


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


def rotate_Z_toward_real(Z, target_resistance, Z0=50, solution="closer"):
    """
    Rotate an impedance to match a target real part (series-matching style step).

    Finds the rotation along the constant-VSWR circle that results in the
    specified real part of impedance.

    Args:
        Z (complex): Input impedance in Ohms.
        target_resistance (float): Desired real part in Ohms.
        Z0 (float): Characteristic impedance (default: 50Ω).
        solution (str): Which solution to use if two exist:
            'closer' (default), 'farther', 'positive_imag', 'negative_imag'.

    Returns:
        complex: Rotated impedance with the target real part.

    Raises:
        ValueError: If target_resistance is not reachable on the constant-VSWR circle.
    """
    z = Z / Z0
    r_target = target_resistance / Z0

    def gamma_from_z(z_norm):
        # Your existing mapping: Γ = (z - 1)/(z + 1)
        return moebius_transform(z_norm, norm=1)

    z_new = _rotate_on_constant_gamma_to_real(
        z,
        r_target,
        gamma_from_z,
        solution=solution,
        what="impedance",
    )
    return z_new * Z0


def rotate_Y_toward_real(Y, target_conductance, Y0=1 / 50, solution="closer"):
    """
    Rotate an admittance to match a target conductance (shunt-matching style step).

    Finds the rotation along the constant-VSWR circle that results in the
    specified real part of admittance (conductance).

    Args:
        Y (complex): Input admittance in Siemens.
        target_conductance (float): Desired real part in Siemens.
        Y0 (float): Characteristic admittance (default: 1/50 S).
        solution (str): Which solution to use if two exist:
            'closer' (default), 'farther', 'positive_imag', 'negative_imag'.

    Returns:
        complex: Rotated admittance with the target conductance.

    Raises:
        ValueError: If target_conductance is not reachable on the constant-VSWR circle.

    Notes:
        This is the standard first step for single shunt-stub matching:
        choose the stub location so that Re{Y(d)} = Y0 (i.e., normalized g = 1),
        then cancel the remaining susceptance with the stub.
    """
    y = Y / Y0
    g_target = target_conductance / Y0

    def gamma_from_y(y_norm):
        # Use the identity: Γ = (z - 1)/(z + 1) with z = 1/y
        # This yields Γ = (1 - y)/(1 + y), i.e., the admittance-form reflection coefficient.
        if y_norm == 0:
            # y=0 -> z=∞ -> Γ≈+1, but avoid division error.
            return 1 + 0j
        z_norm = 1 / y_norm
        return moebius_transform(z_norm, norm=1)

    y_new = _rotate_on_constant_gamma_to_real(
        y,
        g_target,
        gamma_from_y,
        solution=solution,
        what="admittance",
    )
    return y_new * Y0


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


def rotate_Z_toward_imag(Z, target_reactance, Z0=50, solution="closer"):
    """
    Rotate an impedance along its constant-VSWR circle to hit a target reactance.

    Args:
        Z (complex): Impedance in ohms.
        target_reactance (float): Desired imaginary part in ohms.
        Z0 (float): Characteristic impedance.
        solution (str): 'closer', 'farther', 'higher_real', 'lower_real'.

    Returns:
        complex: Rotated impedance (ohms) with imag(Z) == target_reactance.

    Raises:
        ValueError: If target is not reachable on the constant-|Γ| circle.
    """
    z = Z / Z0
    x_target = target_reactance / Z0

    z_new = _crossings_on_constant_gamma_with_imag(
        z,
        x_target,
        _gamma_from_z_norm,
        solution=solution,
        what="impedance",
    )
    return z_new * Z0


def rotate_Y_toward_imag(Y, target_susceptance, Y0=1 / 50, solution="closer"):
    """
    Rotate an admittance along its constant-VSWR circle to hit a target susceptance.

    Args:
        Y (complex): Admittance in siemens.
        target_susceptance (float): Desired imaginary part in siemens.
        Y0 (float): Characteristic admittance.
        solution (str): 'closer', 'farther', 'higher_real', 'lower_real'.

    Returns:
        complex: Rotated admittance (siemens) with imag(Y) == target_susceptance.

    Raises:
        ValueError: If target is not reachable on the constant-|Γ| circle.
    """
    y = Y / Y0
    b_target = target_susceptance / Y0

    y_new = _crossings_on_constant_gamma_with_imag(
        y,
        b_target,
        _gamma_from_y_norm,
        solution=solution,
        what="admittance",
    )
    return y_new * Y0
