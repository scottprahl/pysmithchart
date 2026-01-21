"""
Utility functions for Smith chart computations.

This module provides mathematical utilities, transformations, and helper functions
for working with Smith charts and RF impedance calculations.

Public Functions:
    Domain Validation:
        validate_domain(domain, allow_none): Validate domain parameter
        get_domain_name(domain): Get human-readable domain name

    Complex Number Utilities:
        xy_to_z(xy): Convert x,y components to complex
        z_to_xy(z): Split complex into x,y components
        split_complex(z): Alias for z_to_xy
        cs(z, N): Format complex number as string

    Möbius Transformations:
        moebius_transform(z, norm): Forward Möbius transform (Z → Γ)
        moebius_inverse_transform(s, norm): Inverse Möbius transform (Γ → Z)
        moebius_z(z, norm, normalize): Legacy forward transform
        moebius_inv_z(s, norm, normalize): Legacy inverse transform

    RF Calculations:
        calc_gamma(Z_0, Z_L): Calculate reflection coefficient
        calc_vswr(Z_0, Z_L): Calculate VSWR from impedances
        calc_load(Z_0, gamma): Calculate load impedance from Γ

    Rotation Functions:
        rotate_by_wavelength(Z, wavelength, Z0, direction): Rotate by electrical length
        rotate_toward_real(Z, target_real, Z0, solution): Rotate to match resistance
        rotate_toward_imag(Z, target_imag, Z0, solution): Rotate to match reactance

    Angle/Wavelength Conversions:
        ang_to_c(ang, radius): Angle to complex on circle
        lambda_to_rad(lmb): Wavelength fraction to radians
        rad_to_lambda(rad): Radians to wavelength fraction

For detailed documentation, see individual function docstrings.
"""

from collections.abc import Iterable
import numpy as np
from .constants import SC_EPSILON, SC_INFINITY
from .constants import REFLECTANCE_DOMAIN, IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, ABSOLUTE_DOMAIN

# Public API
__all__ = [
    # Domain validation
    "validate_domain",
    "get_domain_name",
    # Complex utilities
    "xy_to_z",
    "z_to_xy",
    "split_complex",
    "cs",
    # Möbius transforms
    "moebius_transform",
    "moebius_inverse_transform",
    # RF calculations
    "calc_gamma",
    "calc_vswr",
    "calc_load",
    # Rotation functions
    "rotate_by_wavelength",
    "rotate_toward_real",
    "rotate_toward_imag",
    # Angle/wavelength conversions
    "ang_to_c",
    "lambda_to_rad",
    "rad_to_lambda",
    "choose_minor_divider",
]


# ============================================================================
# Parameter Type Validation and Transformation Utilities
# ============================================================================


def validate_domain(domain, allow_none=False):
    """
    Validate that a domain parameter is valid.

    Args:
        domain: The domain to validate (REFLECTANCE_DOMAIN, IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, or ABSOLUTE_DOMAIN)
        allow_none (bool): If True, None is considered valid

    Returns:
        The validated domain

    Raises:
        ValueError: If domain is invalid

    Example:
        >>> from pysmithchart.constants import IMPEDANCE_DOMAIN
        >>> validate_domain(IMPEDANCE_DOMAIN)
        'Z'
    """
    valid_types = [REFLECTANCE_DOMAIN, IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, ABSOLUTE_DOMAIN]

    if domain is None and allow_none:
        return None

    if domain not in valid_types:
        raise ValueError(
            f"Invalid domain: {domain}. "
            f"Must be one of: REFLECTANCE_DOMAIN, IMPEDANCE_DOMAIN, ADMITTANCE_DOMAIN, or ABSOLUTE_DOMAIN"
        )

    return domain


def get_domain_name(domain):
    """
    Get a human-readable name for a domain constant.

    Args:
        domain: The domain constant

    Returns:
        str: Human-readable name

    Example:
        >>> from pysmithchart.constants import IMPEDANCE_DOMAIN
        >>> get_domain_name(IMPEDANCE_DOMAIN)
        'IMPEDANCE_DOMAIN (Impedance)'
    """
    names = {
        REFLECTANCE_DOMAIN: "REFLECTANCE_DOMAIN (Scattering/Reflection coefficient)",
        IMPEDANCE_DOMAIN: "IMPEDANCE_DOMAIN (Impedance)",
        ADMITTANCE_DOMAIN: "ADMITTANCE_DOMAIN (Admittance)",
        ABSOLUTE_DOMAIN: "ABSOLUTE_DOMAIN (Arbitrary/Direct)",
    }
    return names.get(domain, f"Unknown domain: {domain}")


def calc_rho(vswr):
    """Converts VSWR to reflection-coefficient magnitude |Gamma|.

    VSWR and the reflection coefficient magnitude are related (for a lossless line) by:

        VSWR = (1 + |Gamma|) / (1 - |Gamma|)

    Solving for |Gamma| gives:

        |Gamma| = (VSWR - 1) / (VSWR + 1)

    This function accepts either a scalar or an array-like VSWR and returns the
    corresponding scalar or NumPy array of |Gamma| values.

    Args:
        vswr: Voltage standing wave ratio(s). Must satisfy VSWR >= 1. May be a
            float or any array-like object accepted by `numpy.asarray`.

    Returns:
        The reflection-coefficient magnitude(s) |Gamma|. The return type matches
        the input shape:
        - If `vswr` is a scalar, returns a scalar-like NumPy value.
        - If `vswr` is array-like, returns an `ndarray` of the same shape.

        Values satisfy 0 <= |Gamma| < 1 for finite VSWR. (As VSWR -> infinity,
        |Gamma| -> 1.)

    Raises:
        ValueError: If any VSWR value is < 1.

    Examples:
        Convert a single VSWR value:

            rho = calc_rho(2.0)     # 0.333...

        Convert an array of values:

            vswrs = np.array([1.0, 1.5, 2.0, 3.0])
            rho = calc_rho(vswrs)

    Notes:
        - VSWR = 1 corresponds to a perfect match (|Gamma| = 0).
        - This mapping assumes the standard lossless relationship between VSWR
          and |Gamma|.
    """
    v = np.asarray(vswr)
    if np.any(v < 1):
        raise ValueError("VSWR must be >= 1.")
    return (v - 1) / (v + 1)


def calc_vswr_from_rho(rho):
    """Converts reflection-coefficient magnitude |Gamma| to VSWR.

    For a lossless line, VSWR and reflection coefficient magnitude are related by:

        VSWR = (1 + |Gamma|) / (1 - |Gamma|)

    This function accepts either a scalar or an array-like |Gamma| magnitude and
    returns the corresponding scalar or NumPy array of VSWR values.

    Args:
        rho: Reflection-coefficient magnitude(s) |Gamma|. Must satisfy
            0 <= rho < 1. May be a float or any array-like object accepted by
            `numpy.asarray`.

    Returns:
        VSWR value(s). The return type matches the input shape:
        - If `rho` is a scalar, returns a scalar-like NumPy value.
        - If `rho` is array-like, returns an `ndarray` of the same shape.

        Values satisfy 1 <= VSWR < infinity for valid rho. (As rho -> 1,
        VSWR -> infinity.)

    Raises:
        ValueError: If any rho value is < 0 or >= 1.

    Examples:
        Convert a single magnitude:

            vswr = calc_vswr_from_rho(0.2)   # 1.5

        Convert an array of magnitudes:

            rho = np.array([0.0, 0.2, 0.5])
            vswr = calc_vswr_from_rho(rho)

    Notes:
        - rho = 0 corresponds to a perfect match (VSWR = 1).
        - rho values at or above 1 are non-physical for passive terminations
          and cause a divide-by-zero or negative denominator in the VSWR formula,
          so they are rejected.
    """
    r = np.asarray(rho)
    if np.any(r < 0) or np.any(r >= 1):
        raise ValueError("|Gamma| must satisfy 0 <= rho < 1.")
    return (1 + r) / (1 - r)


def calc_vswr(Z_0, Z_L):
    """Computes VSWR for a load impedance referenced to Z0.

    This is a convenience wrapper that computes the reflection coefficient:

        Gamma = (Z_L - Z_0) / (Z_L + Z_0)

    then converts its magnitude to VSWR:

        VSWR = (1 + |Gamma|) / (1 - |Gamma|)

    Args:
        Z_0: Reference (characteristic) impedance Z0 in ohms. Typically a real,
            positive number (e.g., 50 or 75). May be complex, but many
            transmission-line interpretations assume real Z0.
        Z_L: Load impedance in ohms. May be complex.

    Returns:
        The VSWR (a real, non-negative float). For a perfect match, VSWR = 1.
        As the mismatch approaches |Gamma| -> 1, VSWR grows without bound.

    Raises:
        ZeroDivisionError: If `Z_L + Z_0` evaluates to zero (reflection
            coefficient is singular).
        ValueError: If `Z_0` is zero (not a valid reference impedance).

    Examples:
        Matched load:

            calc_vswr(50, 50 + 0j)     # 1.0

        Reactive mismatch:

            calc_vswr(50, 50 + 1j*25)

    Notes:
        - This function assumes the standard lossless relationship between VSWR
          and |Gamma|.
        - If you already have Gamma, prefer `calc_vswr_from_rho(abs(Gamma))` or,
          if you add it, a direct `calc_vswr_from_gamma(Gamma)` helper.
    """
    if Z_0 == 0:
        raise ValueError("Z_0 must be non-zero.")
    gamma = calc_gamma(Z_0, Z_L)
    return (1 + abs(gamma)) / (1 - abs(gamma))


def calc_gamma(Z_0, Z_L):
    """Computes the reflection coefficient Gamma for a load impedance.

    The reflection coefficient is defined as:

        Gamma = (Z_L - Z_0) / (Z_L + Z_0)

    where Z0 is the reference (characteristic) impedance and ZL is the load
    impedance.

    Args:
        Z_0: Reference (characteristic) impedance Z0 in ohms. Typically real and
            positive (e.g., 50 or 75).
        Z_L: Load impedance ZL in ohms. May be complex.

    Returns:
        The complex reflection coefficient Gamma.

    Raises:
        ZeroDivisionError: If `Z_L + Z_0` evaluates to zero.
        ValueError: If `Z_0` is zero (not a valid reference impedance).

    Examples:
        A matched load has Gamma = 0:

            calc_gamma(50, 50)     # 0j

        A short circuit (ZL = 0) yields Gamma = -1:

            calc_gamma(50, 0)      # -1+0j

        An open circuit (ZL -> infinity) approaches Gamma -> +1.

    Notes:
        - |Gamma| <= 1 for passive loads when Z0 is real and positive.
        - Gamma is the natural quantity to plot in `REFLECTANCE_DOMAIN`.
    """
    if Z_0 == 0:
        raise ValueError("Z_0 must be non-zero.")
    return (Z_L - Z_0) / (Z_L + Z_0)


def calc_load(Z_0, gamma):
    """Computes the load impedance ZL from reflection coefficient Gamma.

    This is the inverse relationship of `calc_gamma`:

        Z_L = Z_0 * (1 + Gamma) / (1 - Gamma)

    Args:
        Z_0: Reference (characteristic) impedance Z0 in ohms. Typically real and
            positive (e.g., 50 or 75).
        gamma: Complex reflection coefficient Gamma.

    Returns:
        The load impedance ZL in ohms (complex in general).

    Raises:
        ZeroDivisionError: If `1 - gamma` evaluates to zero (Gamma = 1).
        ValueError: If `Z_0` is zero (not a valid reference impedance).

    Examples:
        Recover ZL from Gamma:

            g = calc_gamma(50, 100 + 1j*25)
            ZL = calc_load(50, g)

        Gamma = 1 corresponds to an open circuit (ZL -> infinity) and is
        singular in this formula.

    Notes:
        - If `gamma` is exactly 1, the returned impedance is unbounded; this
          function will raise a division error.
        - This function is useful when converting measured/simulated S11 (Gamma)
          back into impedance for matching calculations.
    """
    if Z_0 == 0:
        raise ValueError("Z_0 must be non-zero.")
    return Z_0 * (gamma + 1) / (1 - gamma)


def cs_scalar(z, N=5):
    """Convert complex number to string for printing."""
    if z.imag < 0:
        form = "(%% .%df - %%.%dfj)" % (N, N)
    else:
        form = "(%% .%df + %%.%dfj)" % (N, N)
    return form % (z.real, abs(z.imag))


def cs(z, N=5):
    """Convert complex number to string for printing."""
    if np.isscalar(z):
        return cs_scalar(z, N)
    s = ""
    for zz in z:
        s += cs_scalar(zz, N) + " "
    return s


def xy_to_z(*xy):
    """
    Converts input arguments to a complex scalar or an array of complex numbers.

    Args:
        *xy (tuple):

            - If a single argument is passed:

                - If the argument is a complex number or an array-like of complex numbers,
                  it is returned as-is.
                - If the argument is an iterable with two rows (e.g., shape `(2, N)`), it
                  is interpreted as real and imaginary parts, and a complex array is returned.
                - If the argument has more than two dimensions, a `ValueError` is raised.

            - If two arguments are passed:

                - The first argument represents the real part (`x`), and the second
                  represents the imaginary part (`y`).
                - Both arguments must be scalars or iterable objects of the same size.
                  If they are iterable, they are combined to form a complex array.
                - If the sizes of `x` and `y` do not match, a `ValueError` is raised.

    Returns:
        complex or numpy.ndarray: The complex scalar or array of complex numbers.
    """
    if len(xy) == 1:
        z = xy[0]
        if isinstance(z, Iterable):
            z = np.array(z)
            if len(z.shape) == 2:
                if z.shape[0] == 2:  # Ensure the first dimension has size 2
                    z0 = z[0]  # handle case when line.get_data() returns [['0.0'],['']]
                    z0 = np.where(z0 == "", "0.0", z0.astype(object)).astype(float)
                    z1 = z[1]
                    z1 = np.where(z1 == "", "0.0", z1.astype(object)).astype(float)
                    z = z0 + 1j * z1
                else:
                    raise ValueError("Input array must have shape (2, N) for 2D arrays.")
            elif len(z.shape) > 2:
                raise ValueError("Input array has too many dimensions.")
    elif len(xy) == 2:
        x, y = xy
        if isinstance(x, Iterable):
            x = np.array(x)
            y = np.array(y)
            if len(x) == len(y):
                z = x + 1j * y
            else:
                raise ValueError("x and y vectors don't match in type and/or size.")
        else:
            z = float(x) + 1j * float(y)  # Cast scalars to float
    else:
        raise ValueError("Arguments are not a valid complex scalar or array.")

    return z


def z_to_xy(z):
    """
    Converts input data to separate x (real) and y (imaginary) arrays.

    Args:
        z (array-like or scalar):

            - If z is a real or complex number, returns its real and imaginary parts.
            - If z is an array-like object of real or complex numbers, splits it into
              two arrays: real (x) and imaginary (y).
            - If z is already in a 2D array with shape (2, N), it assumes it is [x, y].

    Returns:
        tuple: Two arrays (x, y) representing the real and imaginary parts.
    """
    if isinstance(z, Iterable):
        z = np.array(z)

        # single 1D array
        if len(z.shape) == 1:
            if np.iscomplexobj(z):  # Complex numbers
                x = np.real(z)
                y = np.imag(z)
            else:  # Real numbers
                x = z
                y = np.zeros_like(z)

        # 2D array assume in the form [real, imag]
        elif len(z.shape) == 2:  # 2D array
            if z.shape[0] == 2:  # each row has two elements
                x = z[0]
                y = z[1]
            else:
                raise ValueError("2D input array must have shape (2, N) for [real, imag].")
        else:
            raise ValueError("Input array must be 1D or 2D.")
    else:  # Scalar input
        if np.iscomplex(z):  # Complex scalar
            x = np.real(z)
            y = np.imag(z)
        else:  # Real scalar
            x = np.real(z)
            y = 0.0

    return x, y


def moebius_transform(z, norm=1):
    """
    Apply Möbius transformation to impedance values (CANONICAL IMPLEMENTATION).

    This is the single source of truth for the Möbius transformation formula.
    Maps impedance space to reflection coefficient space.

    Formula: S = 1 - 2*norm / (z + norm)

    Args:
        z (complex or array): Complex impedance value(s)
        norm (float): Normalization constant
            - Use 1 for normalized impedance
            - Use Z0 (e.g., 50) for absolute impedance

    Returns:
        complex or array: Complex reflection coefficient value(s)

    Examples:
        >>> # Normalized impedance to S-parameter
        >>> z_norm = 1 + 0.5j
        >>> s = moebius_transform(z_norm, norm=1)

        >>> # Absolute impedance to S-parameter
        >>> Z_abs = 50 + 25j
        >>> s = moebius_transform(Z_abs, norm=50)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1 - 2 * norm / (z + norm)

    # Replace any inf/nan with inf
    if np.isscalar(result):
        return np.inf if not np.isfinite(result) else result
    else:
        return np.where(np.isfinite(result), result, np.inf)


def moebius_inverse_transform(s, norm=1):
    """
    Apply inverse Möbius transformation to reflection coefficients (CANONICAL IMPLEMENTATION).

    This is the single source of truth for the inverse Möbius transformation formula.
    Maps reflection coefficient space back to impedance space.

    Formula: Z = norm * (1 + S) / (1 - S)

    Args:
        s (complex or array): Complex reflection coefficient value(s)
        norm (float): Normalization constant
            - Use 1 to get normalized impedance
            - Use Z0 (e.g., 50) to get absolute impedance

    Returns:
        complex or array: Complex impedance value(s)

    Examples:
        >>> # S-parameter to normalized impedance
        >>> s = 0.5 + 0.3j
        >>> z_norm = moebius_inverse_transform(s, norm=1)

        >>> # S-parameter to absolute impedance
        >>> s = 0.5 + 0.3j
        >>> Z_abs = moebius_inverse_transform(s, norm=50)
    """
    # Avoid division by zero when s == 1
    s_safe = np.where(s == 1, 1 - SC_EPSILON, s) if hasattr(s, "__iter__") else (1 - SC_EPSILON if s == 1 else s)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = norm * (1 + s_safe) / (1 - s_safe)

    # Replace any inf/nan with SC_INFINITY
    if np.isscalar(result):
        return SC_INFINITY if not np.isfinite(result) else result
    else:
        return np.where(np.isfinite(result), result, SC_INFINITY)


def ang_to_c(ang, radius=1):
    """Converts an angle to a complex number on a circle with the given radius."""
    return radius * (np.cos(ang) + np.sin(ang) * 1j)


def lambda_to_rad(lmb):
    """Converts a wavelength fraction to radians."""
    return lmb * 4 * np.pi


def rad_to_lambda(rad):
    """Converts an angle in radians to a wavelength fraction."""
    return rad * 0.25 / np.pi


def split_complex(z):
    """Splits a complex number into its real and imaginary components."""
    return [np.real(z), np.imag(z)]


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


def rotate_toward_real(Z, target_real, Z0=50, solution="closer"):
    """
    Rotate an impedance to match a target real part.

    Finds the rotation along the constant-VSWR circle that results in the
    specified real part of impedance.

    Args:
        Z (complex): Input impedance in Ohms.
        target_real (float): Desired real part in Ohms.
        Z0 (float): Characteristic impedance (default: 50Ω).
        solution (str): Which solution to use if two exist:
            'closer' - shorter rotation (default)
            'farther' - longer rotation
            'positive_imag' - solution with positive imaginary part
            'negative_imag' - solution with negative imaginary part

    Returns:
        complex: Rotated impedance with the target real part.

    Raises:
        ValueError: If target_real is not reachable on the constant-VSWR circle.

    Examples:
        >>> # Rotate 75+50j to have real part = 50Ω
        >>> Z_new = rotate_toward_real(75+50j, 50, Z0=50)

        >>> # Get the solution with positive reactance
        >>> Z_new = rotate_toward_real(75+50j, 50, solution='positive_imag')

    Notes:
        Not all real parts are reachable - the target must lie on the constant-VSWR circle.
    """
    # Normalize
    z = Z / Z0
    r_target = target_real / Z0

    # Convert to reflection coefficient to get |Γ|
    gamma = moebius_transform(z, norm=1)
    gamma_mag = np.abs(gamma)

    # On a constant-|Γ| circle, we need to find z = r_target + jx such that |Γ(z)| = gamma_mag
    # Γ(z) = (z - 1) / (z + 1)
    # |Γ|² = |z - 1|² / |z + 1|²
    # Let z = r + jx, then:
    # |Γ|² = [(r-1)² + x²] / [(r+1)² + x²]
    #
    # Solving for x given r and |Γ|:
    # |Γ|² [(r+1)² + x²] = (r-1)² + x²
    # |Γ|² (r+1)² + |Γ|² x² = (r-1)² + x²
    # x² (|Γ|² - 1) = (r-1)² - |Γ|² (r+1)²
    # x² = [(r-1)² - |Γ|² (r+1)²] / (|Γ|² - 1)

    numerator = (r_target - 1) ** 2 - gamma_mag**2 * (r_target + 1) ** 2
    denominator = gamma_mag**2 - 1

    # Check if denominator is too small (|Γ| ≈ 1, edge of chart)
    if np.abs(denominator) < 1e-10:
        raise ValueError(f"Target real={target_real}Ω not reachable (|Γ| ≈ 1, edge of chart)")

    x_squared = numerator / denominator

    # Check if solution exists (x² must be non-negative)
    if x_squared < -1e-10:  # Small negative tolerance for numerical errors
        raise ValueError(
            f"Target real={target_real}Ω not reachable on constant-VSWR circle. "
            f"VSWR={(1+gamma_mag)/(1-gamma_mag):.2f}, |Γ|={gamma_mag:.3f}"
        )

    # Handle near-zero case
    if x_squared < 0:
        x_squared = 0

    x_mag = np.sqrt(x_squared)

    # Two solutions: +x and -x
    z_pos = r_target + 1j * x_mag
    z_neg = r_target - 1j * x_mag

    # Select solution based on criterion
    gamma_current = gamma
    gamma_pos = moebius_transform(z_pos, norm=1)
    gamma_neg = moebius_transform(z_neg, norm=1)

    if solution == "positive_imag":
        result = z_pos
    elif solution == "negative_imag":
        result = z_neg
    elif solution == "closer":
        # Choose shorter rotation (smaller phase change)
        angle_current = np.angle(gamma_current)
        angle_pos = np.angle(gamma_pos)
        angle_neg = np.angle(gamma_neg)

        # Calculate angular distances
        diff_pos = np.abs(angle_pos - angle_current)
        diff_neg = np.abs(angle_neg - angle_current)

        # Normalize to [-π, π]
        if diff_pos > np.pi:
            diff_pos = 2 * np.pi - diff_pos
        if diff_neg > np.pi:
            diff_neg = 2 * np.pi - diff_neg

        result = z_pos if diff_pos < diff_neg else z_neg
    elif solution == "farther":
        # Choose longer rotation
        angle_current = np.angle(gamma_current)
        angle_pos = np.angle(gamma_pos)
        angle_neg = np.angle(gamma_neg)

        diff_pos = np.abs(angle_pos - angle_current)
        diff_neg = np.abs(angle_neg - angle_current)

        if diff_pos > np.pi:
            diff_pos = 2 * np.pi - diff_pos
        if diff_neg > np.pi:
            diff_neg = 2 * np.pi - diff_neg

        result = z_pos if diff_pos > diff_neg else z_neg
    else:
        raise ValueError("solution must be 'closer', 'farther', 'positive_imag', or 'negative_imag'")

    return result * Z0


def rotate_toward_imag(Z, target_imag, Z0=50, solution="closer"):
    """
    Rotate an impedance to match a target imaginary part (reactance).

    Finds the rotation along the constant-VSWR circle that results in the
    specified imaginary part of impedance.

    Args:
        Z (complex): Input impedance in Ohms.
        target_imag (float): Desired imaginary part in Ohms (reactance).
            Positive for inductive, negative for capacitive.
        Z0 (float): Characteristic impedance (default: 50Ω).
        solution (str): Which solution to use if two exist:
            'closer' - shorter rotation (default)
            'farther' - longer rotation
            'higher_real' - solution with higher real part
            'lower_real' - solution with lower real part

    Returns:
        complex: Rotated impedance with the target imaginary part.

    Raises:
        ValueError: If target_imag is not reachable on the constant-VSWR circle.

    Examples:
        >>> # Rotate 75+50j to have reactance = 25Ω
        >>> Z_new = rotate_toward_imag(75+50j, 25, Z0=50)

        >>> # Get the solution with higher resistance
        >>> Z_new = rotate_toward_imag(75+50j, 25, solution='higher_real')
    """
    # Normalize
    z = Z / Z0
    x_target = target_imag / Z0

    # Convert to reflection coefficient
    gamma = moebius_transform(z, norm=1)
    gamma_mag = np.abs(gamma)

    # Target point on the constant reactance circle
    # For constant X, we need to find where our constant-VSWR circle intersects
    # This is more complex as constant X forms arcs, not circles in Γ-space

    # Parametrize: try different real parts and find which gives target imaginary
    # Use binary search or Newton's method

    # Simpler approach: sweep angles and find closest match FIXME
    angles = np.linspace(0, 2 * np.pi, 100000)
    gammas = gamma_mag * np.exp(1j * angles)
    zs = moebius_inverse_transform(gammas, norm=1)

    imag_parts = np.imag(zs)
    idx = np.argmin(np.abs(imag_parts - x_target))

    # Check if reachable
    if np.abs(imag_parts[idx] - x_target) > 0.1:  # tolerance
        raise ValueError(
            f"Target imag={target_imag}Ω not reachable on constant-VSWR circle. "
            f"Closest achievable: {imag_parts[idx]*Z0:.1f}Ω"
        )

    # Find the two closest solutions (on opposite sides)
    sorted_idx = np.argsort(np.abs(imag_parts - x_target))

    candidates = []
    for i in sorted_idx[:10]:  # Check top 10 matches
        if np.abs(imag_parts[i] - x_target) < 0.1:
            candidates.append((i, zs[i]))

    if len(candidates) < 2:
        # Only one solution found
        return zs[idx] * Z0

    # We have multiple solutions, pick based on criterion
#    angle_current = np.angle(gamma)

    if solution == "higher_real":
        result = max(candidates, key=lambda x: np.real(x[1]))[1]
    elif solution == "lower_real":
        result = min(candidates, key=lambda x: np.real(x[1]))[1]
    elif solution in ["closer", "farther"]:
        # Calculate angular distances
        diffs = [(np.abs(np.angle(gamma_mag * np.exp(1j * angles[i]) / gamma)), z) for i, z in candidates]
        if solution == "closer":
            result = min(diffs, key=lambda x: x[0])[1]
        else:  # farther
            result = max(diffs, key=lambda x: x[0])[1]
    else:
        raise ValueError("solution must be 'closer', 'farther', 'higher_real', or 'lower_real'")

    return result * Z0


def choose_minor_divider(
    p0,
    p1,
    candidates,
    threshold,
    map_func,
    *,
    max_divisions=None,
    prefer_aligned=True,
    prefer_nice=True,
    tol=1e-9,
):
    """Choose a minor-grid divider that yields visually acceptable spacing.

    This is the shared algorithm used by both fancy and non-fancy minor grids.

    The divider is selected from `candidates` by testing each candidate divider `d`
    via the *minimum adjacent distance* in mapped (Moebius) space over the entire
    interval [p0, p1]. Candidates that do not meet `threshold` are rejected.

    Among acceptable candidates, the selection prefers:
      1) aligned endpoints (both endpoints are integer multiples of the implied step),
      2) "nice" decimal steps (mantissas in {1, 2, 2.5, 5} × 10^n),
      3) larger divider counts.

    Args:
        p0:
            Interval endpoint
        p1:
            Interval endpoint, with p1 > p0.
        candidates:
            Iterable of integer candidate division counts (e.g. [1, 2, 3, 4, 5, 10]).
        threshold:
            Minimum acceptable adjacent spacing in mapped space.
        map_func:
            Callable ``map_func(p) -> complex`` mapping a parameter value to a
            complex coordinate (e.g., Moebius-space point).
        max_divisions:
            If provided, ignore candidates greater than this value.
        prefer_aligned:
            If True, prefer candidates where both endpoints are close to integer
            multiples of the implied step.
        prefer_nice:
            If True, prefer "nice" decimal steps (mantissas in {1, 2, 2.5, 5}×10^n).
        tol:
            Tolerance used for alignment checks.

    Returns:
        int: Chosen division count. Falls back to the smallest candidate if none
        satisfy the threshold.
    """
    p0 = float(p0)
    p1 = float(p1)
    if p1 <= p0:
        raise ValueError("p1 must be greater than p0")

    cand = [int(c) for c in candidates]
    cand = sorted({c for c in cand if c > 0})
    if not cand:
        raise ValueError("candidates must contain at least one positive integer")

    if max_divisions is not None:
        max_divisions = int(max_divisions)
        cand2 = [c for c in cand if c <= max_divisions]
        cand = cand2 or [max_divisions]

    def _is_close_to_int(x: float) -> bool:
        return abs(x - round(x)) <= tol

    def _is_nice_step(step: float) -> bool:
        step = abs(float(step))
        if step == 0:
            return False
        exp = np.floor(np.log10(step))
        mant = step / (10.0**exp)
        for m in (1.0, 2.0, 2.5, 5.0, 10.0):
            if abs(mant - m) <= 1e-12:
                return True
        # allow small floating drift
        for m in (1.0, 2.0, 2.5, 5.0, 10.0):
            if abs(mant - m) / m <= 1e-6:
                return True
        return False

    def _min_mapped_spacing(div: int) -> float:
        pts = np.linspace(p0, p1, div + 1)
        zs = np.array([map_func(p) for p in pts])
        if zs.size < 2:
            return 0.0
        return float(np.min(np.abs(np.diff(zs))))

    ok = [d for d in cand if _min_mapped_spacing(d) > threshold]
    if not ok:
        return cand[0]

    def _score(div: int):
        step = (p1 - p0) / div
        aligned = _is_close_to_int(p0 / step) and _is_close_to_int(p1 / step)
        nice = _is_nice_step(step)
        return (
            aligned if prefer_aligned else True,
            nice if prefer_nice else True,
            div,
        )

    return max(ok, key=_score)
