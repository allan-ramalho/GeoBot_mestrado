"""
Magnetic Processing Functions
Core magnetic data processing implementations
Each function is decorated with @register for auto-discovery
"""

import numpy as np
from scipy import signal, fft
from typing import Dict, Any, Tuple
import logging

from app.services.geophysics.function_registry import register

logger = logging.getLogger(__name__)


@register(
    name="reduction_to_pole",
    description="""
    Apply Reduction to Pole (RTP) transformation to magnetic data.
    
    Reduces magnetic anomalies as if measured at the magnetic pole, removing the
    asymmetry caused by inclination and declination of the Earth's magnetic field.
    This simplifies interpretation by centering anomalies over their sources.
    
    Keywords: RTP, reduction, pole, magnetic field, transformation, inclination, declination
    
    Best practices:
    - Use for low-latitude data to improve interpretability
    - Requires accurate magnetic field inclination and declination
    - May amplify noise at low inclinations (<15°)
    - Consider using pseudo-gravity for very low latitudes
    
    References:
    - Baranov (1957) - Original RTP formulation
    - Li & Oldenburg (1998) - Improved RTP methods
    """,
    keywords=[
        "RTP", "reduction to pole", "magnetic", "transformation",
        "inclination", "declination", "field reduction", "pole reduction"
    ],
    parameters={
        "inclination": {
            "type": "number",
            "description": "Magnetic field inclination in degrees (-90 to 90)",
            "required": True
        },
        "declination": {
            "type": "number",
            "description": "Magnetic field declination in degrees (-180 to 180)",
            "required": True
        }
    },
    examples=[
        "Apply reduction to pole with inclination -30 and declination -20",
        "Reduce to pole using I=-30, D=-20",
        "RTP with inclination -30 degrees"
    ]
)
def reduction_to_pole(
    data: Dict[str, Any],
    inclination: float,
    declination: float
) -> Dict[str, Any]:
    """
    Reduction to Pole transformation
    
    Args:
        data: Dictionary with 'z' (magnetic field), 'x', 'y' coordinates
        inclination: Magnetic field inclination (degrees)
        declination: Magnetic field declination (degrees)
    
    Returns:
        Processed data with RTP applied
    """
    logger.info(f"🧲 RTP: I={inclination}°, D={declination}°")
    
    z = data['z']
    ny, nx = z.shape
    
    # FFT
    Z = fft.fft2(z)
    
    # Frequency domain
    kx = fft.fftfreq(nx, d=1.0)
    ky = fft.fftfreq(ny, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Convert angles to radians
    inc_rad = np.deg2rad(inclination)
    dec_rad = np.deg2rad(declination)
    
    # RTP operator
    theta = np.arctan2(KY, KX)
    L = (K * np.sin(inc_rad) + 
         1j * (KX * np.cos(inc_rad) * np.cos(dec_rad) + 
               KY * np.cos(inc_rad) * np.sin(dec_rad)))
    
    # Avoid division by zero
    L[K == 0] = 1.0
    
    # Apply RTP operator
    Z_rtp = Z / (L + 1e-10)
    
    # Inverse FFT
    z_rtp = fft.ifft2(Z_rtp).real
    
    result = data.copy()
    result['z'] = z_rtp
    result['processing_history'] = data.get('processing_history', []) + [
        f"RTP: I={inclination}°, D={declination}°"
    ]
    
    logger.info("✅ RTP completed")
    return result


@register(
    name="upward_continuation",
    description="""
    Upward continuation of potential field data.
    
    Simulates measuring data at a higher elevation, effectively attenuating
    high-frequency components and emphasizing regional trends. Useful for
    separating regional from residual anomalies.
    
    Keywords: upward, continuation, filtering, regional, smooth, elevation
    
    Best practices:
    - Use to separate regional from residual anomalies
    - Larger heights = more smoothing
    - Stable operation (low-pass filter)
    - Height should be fraction of grid spacing
    
    References:
    - Blakely (1995) - Potential Theory in Gravity and Magnetic Applications
    """,
    keywords=[
        "upward continuation", "continuation", "regional", "filtering",
        "elevation", "smooth", "low-pass", "separation"
    ],
    parameters={
        "height": {
            "type": "number",
            "description": "Continuation height in meters (positive = upward)",
            "required": True
        }
    },
    examples=[
        "Apply upward continuation to 500 meters",
        "Continue upward by 1000m",
        "Upward continue to 500m elevation"
    ]
)
def upward_continuation(
    data: Dict[str, Any],
    height: float
) -> Dict[str, Any]:
    """
    Upward continuation
    
    Args:
        data: Dictionary with field data
        height: Continuation height (meters, positive = upward)
    
    Returns:
        Continued field data
    """
    logger.info(f"⬆️ Upward continuation: h={height}m")
    
    z = data['z']
    ny, nx = z.shape
    
    # FFT
    Z = fft.fft2(z)
    
    # Wavenumber
    dx = np.mean(np.diff(data['x']))
    dy = np.mean(np.diff(data['y']))
    
    kx = fft.fftfreq(nx, d=dx)
    ky = fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Upward continuation operator
    operator = np.exp(-2 * np.pi * K * height)
    
    # Apply
    Z_cont = Z * operator
    
    # Inverse FFT
    z_cont = fft.ifft2(Z_cont).real
    
    result = data.copy()
    result['z'] = z_cont
    result['processing_history'] = data.get('processing_history', []) + [
        f"Upward Continuation: {height}m"
    ]
    
    logger.info("✅ Upward continuation completed")
    return result


@register(
    name="horizontal_gradient",
    description="""
    Calculate Total Horizontal Gradient (THG) of potential field data.
    
    The THG highlights edges and boundaries of anomalous sources, making it
    excellent for mapping geological contacts, faults, and intrusion boundaries.
    
    Keywords: gradient, horizontal, THG, edge, boundary, contact, derivative
    
    Best practices:
    - Excellent for edge detection
    - Maxima locate source edges
    - Relatively noise-insensitive
    - Combine with vertical derivative for better results
    
    References:
    - Cordell & Grauch (1985) - Mapping basement magnetization zones
    """,
    keywords=[
        "horizontal gradient", "THG", "total horizontal gradient",
        "edge detection", "boundary", "contact", "gradient", "derivative"
    ],
    parameters={},
    examples=[
        "Calculate horizontal gradient",
        "Compute THG",
        "Apply total horizontal gradient",
        "Calculate horizontal derivative"
    ]
)
def horizontal_gradient(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Total Horizontal Gradient
    
    Args:
        data: Dictionary with field data
    
    Returns:
        Total horizontal gradient
    """
    logger.info("📐 Calculating Total Horizontal Gradient")
    
    z = data['z']
    
    # Calculate gradients
    dy, dx = np.gradient(z)
    
    # Total horizontal gradient
    thg = np.sqrt(dx**2 + dy**2)
    
    result = data.copy()
    result['z'] = thg
    result['processing_history'] = data.get('processing_history', []) + [
        "Total Horizontal Gradient"
    ]
    
    logger.info("✅ THG completed")
    return result


@register(
    name="vertical_derivative",
    description="""
    Calculate first vertical derivative of potential field data.
    
    Enhances shallow sources and high-frequency anomalies. The vertical derivative
    sharpens anomalies and helps in detailed interpretation of near-surface features.
    
    Keywords: vertical, derivative, first derivative, enhancement, shallow
    
    Best practices:
    - Enhances shallow sources
    - Amplifies noise - consider filtering first
    - Useful for detailed mapping
    - Combine with horizontal gradient
    
    References:
    - Blakely (1995) - Potential Theory
    """,
    keywords=[
        "vertical derivative", "first derivative", "VDR",
        "enhancement", "shallow", "derivative", "differentiation"
    ],
    parameters={
        "order": {
            "type": "integer",
            "description": "Derivative order (1, 2, or 3)",
            "required": False,
            "default": 1
        }
    },
    examples=[
        "Calculate first vertical derivative",
        "Compute vertical derivative",
        "Apply second vertical derivative"
    ]
)
def vertical_derivative(
    data: Dict[str, Any],
    order: int = 1
) -> Dict[str, Any]:
    """
    Vertical Derivative
    
    Args:
        data: Dictionary with field data
        order: Derivative order (1, 2, or 3)
    
    Returns:
        Vertical derivative
    """
    logger.info(f"📏 Calculating {order}th vertical derivative")
    
    z = data['z']
    ny, nx = z.shape
    
    # FFT
    Z = fft.fft2(z)
    
    # Wavenumber
    dx = np.mean(np.diff(data['x']))
    dy = np.mean(np.diff(data['y']))
    
    kx = fft.fftfreq(nx, d=dx)
    ky = fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Vertical derivative operator
    operator = (2 * np.pi * K) ** order
    
    # Apply
    Z_deriv = Z * operator
    
    # Inverse FFT
    z_deriv = fft.ifft2(Z_deriv).real
    
    result = data.copy()
    result['z'] = z_deriv
    result['processing_history'] = data.get('processing_history', []) + [
        f"Vertical Derivative (order {order})"
    ]
    
    logger.info("✅ Vertical derivative completed")
    return result


@register(
    name="tilt_derivative",
    description="""
    Calculate Tilt Angle (Tilt Derivative) of magnetic data.
    
    The tilt angle normalizes gradients, making it excellent for edge detection
    independent of source depth. Values range from -π/2 to π/2, with zero
    crossing at source edges.
    
    Keywords: tilt, angle, TDR, edge detection, normalization
    
    Best practices:
    - Zero crossings locate edges
    - Depth-independent
    - Good for varying source depths
    - Relatively noise-resistant
    
    References:
    - Miller & Singh (1994) - Potential field tilt
    - Verduzco et al. (2004) - Tilt angle applications
    """,
    keywords=[
        "tilt angle", "tilt derivative", "TDR", "THDR",
        "edge detection", "normalization", "angle"
    ],
    parameters={},
    examples=[
        "Calculate tilt angle",
        "Compute tilt derivative",
        "Apply tilt angle transformation"
    ]
)
def tilt_derivative(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Tilt Angle / Tilt Derivative
    
    Args:
        data: Dictionary with field data
    
    Returns:
        Tilt angle in radians
    """
    logger.info("📊 Calculating Tilt Angle")
    
    z = data['z']
    
    # Horizontal gradient
    dy, dx = np.gradient(z)
    thg = np.sqrt(dx**2 + dy**2)
    
    # Vertical derivative (approximate)
    vdr = np.gradient(z, axis=0)
    
    # Tilt angle
    tilt = np.arctan2(vdr, thg + 1e-10)
    
    result = data.copy()
    result['z'] = tilt
    result['processing_history'] = data.get('processing_history', []) + [
        "Tilt Angle"
    ]
    
    logger.info("✅ Tilt angle completed")
    return result


@register(
    name="analytic_signal",
    description="Calculate analytic signal amplitude for source edge detection",
    keywords=["analytic signal", "total gradient", "amplitude", "edge detection", "envelope"],
    best_practices=[
        "Independent of magnetization direction",
        "Peaks directly over source edges",
        "Good for low magnetic latitudes",
        "Combines horizontal and vertical derivatives"
    ],
    references=[
        "Nabighian (1972). The analytic signal of two-dimensional magnetic bodies",
        "Roest et al. (1992). Magnetic interpretation using the 3-D analytic signal",
        "MacLeod et al. (1993). Analytic signal and reduction-to-pole"
    ]
)
def analytic_signal(
    data: Dict[str, Any],
    dx: float = 1.0,
    dy: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate 3D analytic signal amplitude
    
    The analytic signal amplitude is independent of magnetization direction
    and peaks directly over source edges, making it ideal for interpretation
    at low magnetic latitudes where RTP is problematic.
    
    Formula: |A(x,y)| = sqrt((∂T/∂x)² + (∂T/∂y)² + (∂T/∂z)²)
    
    Args:
        data: Dictionary with magnetic field data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        
    Returns:
        Dictionary with analytic signal amplitude
    """
    logger.info("📊 Calculating Analytic Signal")
    
    z = data['z']
    
    # Calculate horizontal derivatives
    dy_deriv, dx_deriv = np.gradient(z, dy, dx)
    
    # Calculate vertical derivative (using FFT for accuracy)
    from scipy.fft import fft2, ifft2, fftfreq
    
    ny, nx = z.shape
    f_data = fft2(z)
    
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Vertical derivative in frequency domain
    f_vd = f_data * K
    vd = np.real(ifft2(f_vd))
    
    # Calculate analytic signal amplitude
    analytic_amp = np.sqrt(dx_deriv**2 + dy_deriv**2 + vd**2)
    
    result = data.copy()
    result['z'] = analytic_amp
    result['processing_history'] = data.get('processing_history', []) + [
        f"Analytic Signal (dx={dx}m, dy={dy}m)"
    ]
    
    logger.info("✅ Analytic signal completed")
    return result


@register(
    name="total_horizontal_derivative",
    description="Calculate total horizontal derivative for edge detection",
    keywords=["horizontal derivative", "THD", "THDR", "edge", "gradient", "boundary"],
    best_practices=[
        "Peaks over source edges",
        "Less sensitive to source depth than analytic signal",
        "Good first-order edge detector",
        "Combine with tilt derivative for robust interpretation"
    ],
    references=[
        "Cordell & Grauch (1985). Mapping basement magnetization zones",
        "Phillips (2000). Processing and interpretation of aeromagnetic data"
    ]
)
def total_horizontal_derivative(
    data: Dict[str, Any],
    dx: float = 1.0,
    dy: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate total horizontal derivative (THD)
    
    THD is the magnitude of the horizontal gradient, highlighting edges
    and boundaries of magnetic sources.
    
    Formula: THD = sqrt((∂T/∂x)² + (∂T/∂y)²)
    
    Args:
        data: Dictionary with magnetic field data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        
    Returns:
        Dictionary with total horizontal derivative
    """
    logger.info("📊 Calculating Total Horizontal Derivative")
    
    z = data['z']
    
    # Calculate horizontal derivatives
    dy_deriv, dx_deriv = np.gradient(z, dy, dx)
    
    # Total horizontal derivative
    thd = np.sqrt(dx_deriv**2 + dy_deriv**2)
    
    result = data.copy()
    result['z'] = thd
    result['processing_history'] = data.get('processing_history', []) + [
        f"Total Horizontal Derivative (dx={dx}m, dy={dy}m)"
    ]
    
    logger.info("✅ Total horizontal derivative completed")
    return result


@register(
    name="pseudogravity",
    category="magnetic",
    description="Transform magnetic data to equivalent gravity field (Poisson relation)",
    keywords=["pseudogravity", "pseudo-gravity", "poisson", "transformation", "gravity equivalent"],
    best_practices=[
        "Useful for joint gravity-magnetic interpretation",
        "Requires magnetization and density contrast ratio",
        "Good for comparing magnetic and gravity anomalies",
        "Assumes same source geometry"
    ],
    references=[
        "Baranov (1957). A new method for interpretation of aeromagnetic maps",
        "Blakely (1995). Potential Theory in Gravity and Magnetic Applications"
    ]
)
def pseudogravity(
    data: Dict[str, Any],
    dx: float = 1.0,
    dy: float = 1.0,
    inclination: float = -30.0,
    declination: float = 0.0,
    mag_to_dens_ratio: float = 0.03,
    **kwargs
) -> Dict[str, Any]:
    """
    Transform magnetic field to pseudogravity
    
    Converts magnetic anomalies to equivalent gravity anomalies using
    Poisson's relation, allowing direct comparison with gravity data.
    
    Args:
        data: Dictionary with magnetic field data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        inclination: Magnetic field inclination (degrees)
        declination: Magnetic field declination (degrees)
        mag_to_dens_ratio: Magnetization/density contrast ratio
        
    Returns:
        Dictionary with pseudogravity field
    """
    logger.info(f"📊 Calculating Pseudogravity (inc={inclination}°, dec={declination}°)")
    
    from scipy.fft import fft2, ifft2, fftfreq
    
    z = data['z']
    ny, nx = z.shape
    
    # Convert angles to radians
    inc_rad = np.deg2rad(inclination)
    dec_rad = np.deg2rad(declination)
    
    # FFT of magnetic data
    f_mag = fft2(z)
    
    # Wavenumber grids
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Avoid division by zero
    K = np.where(K == 0, 1e-10, K)
    
    # Direction cosines of magnetization
    mx = np.cos(inc_rad) * np.sin(dec_rad)
    my = np.cos(inc_rad) * np.cos(dec_rad)
    mz = np.sin(inc_rad)
    
    # Pseudogravity transformation
    # Remove magnetization direction effect
    theta = (KX * mx + KY * my + K * mz) / K
    
    # Apply Poisson relation
    f_pseudograv = (f_mag / (theta + 1e-10)) * mag_to_dens_ratio
    
    # Inverse FFT
    pseudograv = np.real(ifft2(f_pseudograv))
    
    result = data.copy()
    result['z'] = pseudograv
    result['processing_history'] = data.get('processing_history', []) + [
        f"Pseudogravity (inc={inclination}°, dec={declination}°, ratio={mag_to_dens_ratio})"
    ]
    
    logger.info("✅ Pseudogravity transformation completed")
    return result


@register(
    name="matched_filter",
    category="magnetic",
    description="Apply matched filter to enhance anomalies from specific depth range",
    keywords=["matched filter", "depth", "wavelength", "enhancement", "spectral"],
    best_practices=[
        "Enhances sources at target depth",
        "Requires approximate source depth",
        "Use multiple filters for different depth ranges",
        "Good for isolating shallow vs deep sources"
    ],
    references=[
        "Syberg (1972). A Fourier method for the regional-residual problem",
        "Phillips (2001). Two-step processing for 3D magnetic data"
    ]
)
def matched_filter(
    data: Dict[str, Any],
    dx: float = 1.0,
    dy: float = 1.0,
    target_depth: float = 1000.0,
    depth_range: float = 500.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply matched filter for depth-selective enhancement
    
    Enhances anomalies originating from sources within a specified
    depth range by designing a filter in the frequency domain.
    
    Args:
        data: Dictionary with magnetic/gravity field data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        target_depth: Target source depth (meters)
        depth_range: Depth window width (meters)
        
    Returns:
        Dictionary with filtered data highlighting target depth
    """
    logger.info(f"📊 Applying Matched Filter (depth={target_depth}±{depth_range}m)")
    
    from scipy.fft import fft2, ifft2, fftfreq
    
    z = data['z']
    ny, nx = z.shape
    
    # FFT
    f_data = fft2(z)
    
    # Wavenumber grids
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Matched filter for depth range
    # Based on exponential decay e^(-k*z)
    z1 = target_depth - depth_range / 2
    z2 = target_depth + depth_range / 2
    
    filter_matched = (np.exp(-K * z1) - np.exp(-K * z2)) / (K * (z2 - z1) + 1e-10)
    
    # Normalize filter
    filter_matched = filter_matched / np.max(np.abs(filter_matched))
    
    # Apply filter
    f_filtered = f_data * filter_matched
    result_data = np.real(ifft2(f_filtered))
    
    result = data.copy()
    result['z'] = result_data
    result['processing_history'] = data.get('processing_history', []) + [
        f"Matched Filter (depth={target_depth}m, range=±{depth_range}m)"
    ]
    
    logger.info("✅ Matched filter applied")
    return result


# Initialize registry on module import
logger.info("🔧 Magnetic processing functions registered")
