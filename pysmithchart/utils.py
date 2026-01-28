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

    RF Calculations:
        calc_gamma(Z_0, Z_L): Calculate reflection coefficient
        calc_vswr(Z_0, Z_L): Calculate VSWR from impedances
        calc_load(Z_0, gamma): Calculate load impedance from Γ
        reactance_to_component(X, freq): describe reactance as C or L

    Angle/Wavelength Conversions:
        ang_to_c(ang, radius): Angle to complex on circle
        lambda_to_rad(lmb): Wavelength fraction to radians
        rad_to_lambda(rad): Radians to wavelength fraction

"""

from collections.abc import Iterable
import numpy as np
from .constants import SC_EPSILON, SC_INFINITY
from .constants import R_DOMAIN, Z_DOMAIN, Y_DOMAIN, NORM_Z_DOMAIN, NORM_Y_DOMAIN

# Public API
__all__ = (
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
    # Angle/wavelength conversions
    "ang_to_c",
    "lambda_to_rad",
    "rad_to_lambda",
    "choose_minor_divider",
    "reactance_to_component",
)


def validate_domain(domain):
    """
    Validate that a domain parameter is valid.

    Args:
        domain: The domain to validate

    Returns:
        The validated domain
    """
    valid_types = [R_DOMAIN, Z_DOMAIN, Y_DOMAIN, NORM_Z_DOMAIN, NORM_Y_DOMAIN]

    if domain not in valid_types:
        raise ValueError(
            f"Invalid domain: {domain}. " f"Must be one of: R_DOMAIN, Z_DOMAIN, Y_DOMAIN, NORM_Z_DOMAIN, NORM_Y_DOMAIN"
        )

    return domain


def get_domain_name(domain):
    """
    Get a human-readable name for a domain constant.

    Args:
        domain: The domain constant

    Returns:
        str: Human-readable name
    """
    names = {
        R_DOMAIN: "R_DOMAIN Scattering/Reflection coefficient",
        Z_DOMAIN: "Z_DOMAIN Impedance in Ohms",
        Y_DOMAIN: "Y_DOMAIN Admittance in Siemens",
        NORM_Z_DOMAIN: "NORM_Z_DOMAIN normalized impedance",
        NORM_Y_DOMAIN: "NORM_Z_DOMAIN normalized admittance",
    }
    return names.get(domain, f"Unknown domain: {domain}")


def vswr_to_gamma_mag(vswr):
    """Converts VSWR to reflection-coefficient magnitude |Gamma|.

    VSWR and the reflection coefficient magnitude are related (for a lossless line) by:

        VSWR = (1 + |Gamma|) / (1 - |Gamma|)

    Solving for |Gamma| gives:

        |Gamma| = (VSWR - 1) / (VSWR + 1)

    This function accepts either a scalar or an array-like VSWR and returns the
    corresponding scalar or NumPy array of |Gamma| values.

    This mapping assumes the standard lossless relationship between VSWR
    and |Gamma|.

    Returned values satisfy 0 <= |Gamma| < 1 for finite VSWR. (As VSWR -> infinity,
    |Gamma| -> 1.)

    Args:
        vswr: Voltage standing wave ratio(s). Must satisfy VSWR >= 1. May be a
            float or any array-like object accepted by `numpy.asarray`.

    Returns:
        The reflection-coefficient magnitude(s) |Gamma|. The return type matches
        the input shape.
    """
    v = np.asarray(vswr)
    if np.any(v < 1):
        raise ValueError("VSWR must be >= 1.")
    return (v - 1) / (v + 1)


def calc_vswr_from_gamma(gamma):
    """Converts reflection-coefficient magnitude |Gamma| to VSWR.

    For a lossless line, VSWR and reflection coefficient magnitude are related by:

        VSWR = (1 + |Gamma|) / (1 - |Gamma|)

    This function accepts either a scalar or an array-like |Gamma| magnitude and
    returns the corresponding scalar or NumPy array of VSWR values.

    Args:
        gamma: Reflection-coefficient magnitude(s) |Gamma|. Must satisfy
            0 <= gamma < 1. May be a float or any array-like object accepted by
            `numpy.asarray`.

    Returns:
        VSWR value(s). The return type matches the input shape:
    """
    r = np.asarray(gamma)
    if np.any(r < 0) or np.any(r >= 1):
        raise ValueError("|Gamma| must satisfy 0 <= gamma < 1.")
    return (1 + r) / (1 - r)


def calc_vswr(Z_0, Z_L):
    """Computes VSWR for a load impedance referenced to Z0.

    This is a convenience wrapper that computes the reflection coefficient:

        Gamma = (Z_L - Z_0) / (Z_L + Z_0)

    then converts its magnitude to VSWR:

        VSWR = (1 + |Gamma|) / (1 - |Gamma|)

    This function assumes the standard lossless relationship between VSWR
    and |Gamma|.

    Args:
        Z_0: Reference (characteristic) impedance Z0 in ohms. Typically a real,
            positive number (e.g., 50 or 75). May be complex, but many
            transmission-line interpretations assume real Z0.
        Z_L: Load impedance in ohms. May be complex.

    Returns:
        The VSWR (a real, non-negative float). For a perfect match, VSWR = 1.
        As the mismatch approaches |Gamma| -> 1, VSWR grows without bound.
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
    """
    if Z_0 == 0:
        raise ValueError("Z_0 must be non-zero.")
    return Z_0 * (gamma + 1) / (1 - gamma)


def cs_scalar(z, N=3, parens=False, trim_zeros=True):
    """Convert complex number to string for printing."""
    form = "%% .%df" % N
    s_real = form % z.real
    s_imag = form % abs(z.imag)
    if trim_zeros:
        if "." in s_real:
            s_real = s_real.rstrip("0").rstrip(".")
        if "." in s_imag:
            s_imag = s_imag.rstrip("0").rstrip(".")

    if z.imag < 0:
        s_imag = "- %sj" % s_imag
    else:
        s_imag = "+ %sj" % s_imag

    if parens:
        return "(%s%s)" % (s_real, s_imag)

    if abs(z.imag) < 10 ** (-N):
        return s_real
    if abs(z.real) < 10 ** (-N):
        return s_imag

    return "%s %s" % (s_real, s_imag)


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

            - If two arguments are passed:
                - The first argument represents the real part (`x`), and the second
                  represents the imaginary part (`y`).
                - Both arguments must be scalars or iterable objects of the same size.
                  If they are iterable, they are combined to form a complex array.

    Returns:
        complex or numpy.ndarray: A complex scalar or array of complex numbers.
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
    Apply Möbius transformation to impedance values.

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

    return np.where(np.isfinite(result), result, np.inf)


def moebius_inverse_transform(s, norm=1):
    """
    Apply inverse Möbius transformation to reflection coefficients.

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


def reactance_to_component(X, freq):
    """
    Convert reactance to component value (L or C).

    Args:
        X: Reactance in Ohms
        freq: Frequency in Hz

    Returns:
        (component_type, value, unit)
    """
    omega = 2 * np.pi * freq

    vals = ("None", 0, "")

    if X > 0:  # Inductor
        L = X / omega
        if L >= 1e-3:
            vals = ("Capacitor", L * 1e6, "mH")
        elif L >= 1e-6:
            vals = ("Inductor", L * 1e6, "µH")
        else:
            vals = ("Inductor", L * 1e9, "nH")

    if X < 0:  # Capacitor
        C = -1 / (omega * X)
        if C >= 1e-3:
            vals = ("Capacitor", C * 1e6, "mF")
        elif C >= 1e-6:
            vals = ("Capacitor", C * 1e6, "µF")
        elif C >= 1e-9:
            vals = ("Capacitor", C * 1e9, "nF")
        else:
            vals = ("Capacitor", C * 1e12, "pF")

    return vals
