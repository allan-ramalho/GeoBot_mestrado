"""
Signal Processing Filters for Geophysical Data
Comprehensive filtering suite for noise reduction and signal enhancement
"""

import numpy as np
from scipy import signal, ndimage
from scipy.fft import fft2, ifft2, fftfreq
from typing import Dict, Any, Optional, Tuple
import logging

from app.services.geophysics.function_registry import register

logger = logging.getLogger(__name__)


@register(
    name="butterworth_filter",
    category="filters",
    description="Apply Butterworth filter (low-pass, high-pass, or band-pass) in frequency domain",
    keywords=["butterworth", "filter", "lowpass", "highpass", "bandpass", "frequency"],
    best_practices=[
        "Low-pass: remove high-frequency noise",
        "High-pass: enhance short-wavelength features",
        "Band-pass: isolate specific wavelength range",
        "Higher order = sharper cutoff but potential ringing"
    ],
    references=[
        "Butterworth, S. (1930). On the Theory of Filter Amplifiers",
        "Blakely (1995). Potential Theory in Gravity and Magnetic Applications",
        "Oppenheim & Schafer (2009). Discrete-Time Signal Processing"
    ]
)
def butterworth_filter(
    data: np.ndarray,
    dx: float,
    dy: float,
    cutoff_wavelength: float,
    filter_type: str = "low-pass",
    order: int = 4,
    high_cutoff: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Butterworth filter to geophysical grid data
    
    Butterworth filter provides smooth frequency response with no ripples
    in passband or stopband. Ideal for general-purpose filtering.
    
    Args:
        data: Input grid data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        cutoff_wavelength: Cutoff wavelength (meters)
        filter_type: 'low-pass', 'high-pass', or 'band-pass'
        order: Filter order (default: 4, higher = sharper cutoff)
        high_cutoff: High cutoff wavelength for band-pass (meters)
        
    Returns:
        Dictionary with filtered data and metadata
        
    Example:
        >>> # Remove noise with wavelengths < 500m
        >>> filtered = butterworth_filter(
        ...     data, dx=50, dy=50,
        ...     cutoff_wavelength=500,
        ...     filter_type='high-pass',
        ...     order=4
        ... )
    """
    logger.info(f"Applying Butterworth {filter_type} filter, cutoff={cutoff_wavelength}m, order={order}")
    
    # FFT of data
    f_data = fft2(data)
    
    # Frequency grids
    ny, nx = data.shape
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Cutoff wavenumber
    k_cutoff = 2 * np.pi / cutoff_wavelength
    
    # Build Butterworth filter
    if filter_type == "low-pass":
        # H(k) = 1 / (1 + (k/k_c)^(2n))
        butter_filter = 1.0 / (1.0 + (K / k_cutoff)**(2 * order))
        
    elif filter_type == "high-pass":
        # H(k) = 1 - 1 / (1 + (k/k_c)^(2n))
        butter_filter = 1.0 - 1.0 / (1.0 + (K / k_cutoff)**(2 * order))
        
    elif filter_type == "band-pass":
        if high_cutoff is None:
            raise ValueError("high_cutoff required for band-pass filter")
        
        k_high = 2 * np.pi / high_cutoff
        
        # Band-pass: low-pass at k_high AND high-pass at k_cutoff
        low_pass = 1.0 / (1.0 + (K / k_high)**(2 * order))
        high_pass = 1.0 - 1.0 / (1.0 + (K / k_cutoff)**(2 * order))
        butter_filter = low_pass * high_pass
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    
    # Apply filter in frequency domain
    f_filtered = f_data * butter_filter
    
    # Inverse FFT
    result = np.real(ifft2(f_filtered))
    
    metadata = {
        "function": "butterworth_filter",
        "filter_type": filter_type,
        "cutoff_wavelength_m": cutoff_wavelength,
        "high_cutoff_wavelength_m": high_cutoff,
        "order": order,
        "grid_spacing_m": (dx, dy),
        "original_range": (float(np.nanmin(data)), float(np.nanmax(data))),
        "filtered_range": (float(np.nanmin(result)), float(np.nanmax(result))),
        "energy_removed": float(np.sum(np.abs(data - result)**2) / np.sum(np.abs(data)**2) * 100)
    }
    
    logger.info(f"Butterworth filter applied. Energy removed: {metadata['energy_removed']:.1f}%")
    
    return {
        "result": result,
        "filter": butter_filter,
        "metadata": metadata
    }


@register(
    name="gaussian_filter",
    category="filters",
    description="Apply Gaussian smoothing filter for noise reduction while preserving edges",
    keywords=["gaussian", "smooth", "filter", "noise", "blur"],
    best_practices=[
        "Good for noise reduction with minimal edge distortion",
        "Sigma controls smoothing strength",
        "Larger sigma = more smoothing",
        "Use for pre-processing before derivative calculations"
    ],
    references=[
        "Gonzalez & Woods (2008). Digital Image Processing",
        "Blakely (1995). Potential Theory in Gravity and Magnetic Applications"
    ]
)
def gaussian_filter(
    data: np.ndarray,
    sigma: float = 1.0,
    truncate: float = 4.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Gaussian smoothing filter
    
    Gaussian filter is optimal for noise reduction while preserving
    signal characteristics. Widely used in image processing and
    geophysical data enhancement.
    
    Args:
        data: Input grid data
        sigma: Standard deviation of Gaussian kernel (default: 1.0)
        truncate: Truncate filter at this many standard deviations (default: 4.0)
        
    Returns:
        Dictionary with smoothed data and metadata
        
    Example:
        >>> smoothed = gaussian_filter(data, sigma=2.0)
    """
    logger.info(f"Applying Gaussian filter with sigma={sigma}")
    
    # Apply Gaussian filter
    result = ndimage.gaussian_filter(data, sigma=sigma, truncate=truncate)
    
    # Calculate difference (noise removed)
    noise = data - result
    
    metadata = {
        "function": "gaussian_filter",
        "sigma": sigma,
        "truncate": truncate,
        "original_std": float(np.nanstd(data)),
        "filtered_std": float(np.nanstd(result)),
        "noise_std": float(np.nanstd(noise)),
        "smoothing_factor": float(np.nanstd(result) / np.nanstd(data))
    }
    
    logger.info(f"Gaussian smoothing applied. Smoothing factor: {metadata['smoothing_factor']:.3f}")
    
    return {
        "result": result,
        "noise": noise,
        "metadata": metadata
    }


@register(
    name="median_filter",
    category="filters",
    description="Apply median filter for robust outlier and spike removal",
    keywords=["median", "filter", "outlier", "spike", "robust"],
    best_practices=[
        "Excellent for removing isolated spikes",
        "Preserves edges better than mean filters",
        "Size controls neighborhood (odd number)",
        "Non-linear filter - may affect frequency content"
    ],
    references=[
        "Tukey, J.W. (1977). Exploratory Data Analysis",
        "Huang et al. (1979). A fast two-dimensional median filtering algorithm"
    ]
)
def median_filter(
    data: np.ndarray,
    size: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply median filter for spike and outlier removal
    
    Median filter replaces each pixel with the median of its neighborhood.
    Very effective for removing isolated anomalies while preserving edges.
    
    Args:
        data: Input grid data
        size: Filter window size (odd number, default: 3)
        
    Returns:
        Dictionary with filtered data and metadata
        
    Example:
        >>> despíked = median_filter(noisy_data, size=5)
    """
    logger.info(f"Applying median filter with size={size}")
    
    if size % 2 == 0:
        raise ValueError("Filter size must be odd number")
    
    # Apply median filter
    result = ndimage.median_filter(data, size=size)
    
    # Calculate difference (spikes removed)
    spikes = data - result
    
    # Count significant spikes (> 3 sigma)
    spike_threshold = 3 * np.nanstd(data)
    n_spikes = np.sum(np.abs(spikes) > spike_threshold)
    
    metadata = {
        "function": "median_filter",
        "window_size": size,
        "n_spikes_removed": int(n_spikes),
        "spike_threshold": float(spike_threshold),
        "max_spike": float(np.nanmax(np.abs(spikes))),
        "original_range": (float(np.nanmin(data)), float(np.nanmax(data))),
        "filtered_range": (float(np.nanmin(result)), float(np.nanmax(result)))
    }
    
    logger.info(f"Median filter applied. Spikes removed: {n_spikes}")
    
    return {
        "result": result,
        "spikes": spikes,
        "metadata": metadata
    }


@register(
    name="directional_filter",
    category="filters",
    description="Apply directional filter to enhance features in specific azimuth",
    keywords=["directional", "azimuth", "orientation", "lineament", "structural"],
    best_practices=[
        "Use to enhance linear features (faults, dikes)",
        "Azimuth in degrees from North",
        "Width controls selectivity",
        "Combine multiple azimuths for comprehensive analysis"
    ],
    references=[
        "Cordell & Grauch (1985). Mapping basement magnetization zones",
        "Milligan & Gunn (1997). Enhancement and presentation of airborne geophysical data"
    ]
)
def directional_filter(
    data: np.ndarray,
    dx: float,
    dy: float,
    azimuth: float,
    width: float = 15.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply directional filter to enhance features at specific azimuth
    
    Enhances features oriented in specified direction by creating a
    cone-shaped pass filter in the frequency domain.
    
    Args:
        data: Input grid data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        azimuth: Direction to enhance (degrees from North, 0-360)
        width: Angular width of filter (degrees, default: 15)
        
    Returns:
        Dictionary with directionally filtered data and metadata
        
    Example:
        >>> # Enhance E-W trending features
        >>> ew_enhanced = directional_filter(
        ...     data, dx=50, dy=50,
        ...     azimuth=90, width=15
        ... )
    """
    logger.info(f"Applying directional filter at azimuth={azimuth}°, width={width}°")
    
    # Convert azimuth from degrees to radians (geographic to math convention)
    azimuth_rad = np.deg2rad(90 - azimuth)
    width_rad = np.deg2rad(width)
    
    # FFT of data
    f_data = fft2(data)
    
    # Frequency grids
    ny, nx = data.shape
    kx = fftfreq(nx, dx)
    ky = fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    
    # Calculate angle of each frequency component
    angle = np.arctan2(KY, KX)
    
    # Angular difference from target azimuth
    angle_diff = np.abs(angle - azimuth_rad)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # Wrap to [0, π]
    
    # Gaussian cone filter
    dir_filter = np.exp(-(angle_diff**2) / (2 * width_rad**2))
    
    # Apply filter
    f_filtered = f_data * dir_filter
    result = np.real(ifft2(f_filtered))
    
    metadata = {
        "function": "directional_filter",
        "azimuth_deg": azimuth,
        "width_deg": width,
        "grid_spacing_m": (dx, dy),
        "original_range": (float(np.nanmin(data)), float(np.nanmax(data))),
        "filtered_range": (float(np.nanmin(result)), float(np.nanmax(result)))
    }
    
    logger.info("Directional filter applied")
    
    return {
        "result": result,
        "filter": dir_filter,
        "metadata": metadata
    }


@register(
    name="cosine_directional_filter",
    category="filters",
    description="Apply cosine directional filter (1st vertical derivative directional)",
    keywords=["cosine", "directional", "derivative", "gradient", "structural"],
    best_practices=[
        "Enhances edges perpendicular to azimuth",
        "Simulates illumination from specified direction",
        "Good for structural interpretation",
        "Azimuth is direction of illumination"
    ],
    references=[
        "Cooper & Cowan (2006). Enhancing potential field data",
        "Cordell & Grauch (1985). Mapping basement magnetization zones"
    ]
)
def cosine_directional_filter(
    data: np.ndarray,
    dx: float,
    dy: float,
    azimuth: float,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply cosine directional filter (directional derivative)
    
    Calculates derivative in specified direction, equivalent to illuminating
    the field from that direction.
    
    Formula: ∂f/∂θ = cos(θ)∂f/∂x + sin(θ)∂f/∂y
    
    Args:
        data: Input grid data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        azimuth: Direction of derivative (degrees from North)
        
    Returns:
        Dictionary with directional derivative and metadata
    """
    logger.info(f"Applying cosine directional filter at azimuth={azimuth}°")
    
    # Convert azimuth to radians (geographic to math convention)
    azimuth_rad = np.deg2rad(90 - azimuth)
    
    # Calculate gradients
    gy, gx = np.gradient(data, dy, dx)
    
    # Directional derivative
    cos_theta = np.cos(azimuth_rad)
    sin_theta = np.sin(azimuth_rad)
    result = cos_theta * gx + sin_theta * gy
    
    metadata = {
        "function": "cosine_directional_filter",
        "azimuth_deg": azimuth,
        "grid_spacing_m": (dx, dy),
        "result_range": (float(np.nanmin(result)), float(np.nanmax(result)))
    }
    
    logger.info("Cosine directional filter applied")
    
    return {
        "result": result,
        "metadata": metadata
    }


@register(
    name="wiener_filter",
    category="filters",
    description="Apply Wiener filter for optimal noise reduction with known noise spectrum",
    keywords=["wiener", "optimal", "noise", "snr", "restoration"],
    best_practices=[
        "Requires estimate of noise power spectrum",
        "Balances noise reduction and signal preservation",
        "SNR parameter controls filtering strength",
        "Higher SNR = less filtering"
    ],
    references=[
        "Wiener, N. (1949). Extrapolation, Interpolation, and Smoothing of Stationary Time Series",
        "Gonzalez & Woods (2008). Digital Image Processing"
    ]
)
def wiener_filter(
    data: np.ndarray,
    noise_variance: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Wiener filter for optimal noise reduction
    
    Wiener filter minimizes mean square error between original and
    filtered signal, providing optimal filtering when noise characteristics
    are known.
    
    Args:
        data: Input grid data
        noise_variance: Estimated noise variance (default: auto-estimate)
        
    Returns:
        Dictionary with filtered data and metadata
    """
    logger.info("Applying Wiener filter")
    
    from scipy.signal import wiener as scipy_wiener
    
    # Auto-estimate noise if not provided
    if noise_variance is None:
        # Estimate noise from high-frequency components
        noise_variance = np.var(data - ndimage.gaussian_filter(data, 1))
    
    # Apply Wiener filter
    result = scipy_wiener(data, noise=noise_variance)
    
    metadata = {
        "function": "wiener_filter",
        "noise_variance": float(noise_variance),
        "original_variance": float(np.var(data)),
        "filtered_variance": float(np.var(result)),
        "noise_reduction_db": float(10 * np.log10(np.var(data) / np.var(result)))
    }
    
    logger.info(f"Wiener filter applied. Noise reduction: {metadata['noise_reduction_db']:.1f} dB")
    
    return {
        "result": result,
        "metadata": metadata
    }
