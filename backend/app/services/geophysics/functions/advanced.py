"""
Advanced Geophysical Processing Functions
Source parameter imaging, depth estimation, and inversion methods
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter
from ..function_registry import register

logger = logging.getLogger(__name__)


@register(
    name="euler_deconvolution",
    category="advanced",
    description="Estimate source depth and location using Euler deconvolution",
    keywords=["euler", "depth", "source location", "structural index", "automatic"],
    best_practices=[
        "Use structural index: 0 (contact), 1 (sill/dike), 2 (pipe), 3 (sphere)",
        "Apply on gridded derivatives data",
        "Filter solutions by depth uncertainty",
        "Cluster solutions to identify real sources"
    ],
    references=[
        "Thompson (1982). EULDPH - A new technique for making computer-assisted depth estimates",
        "Reid et al. (1990). Magnetic interpretation in three dimensions using Euler deconvolution",
        "Mushayandebvu et al. (2001). Grid Euler deconvolution with constraints"
    ]
)
def euler_deconvolution(
    data: Dict[str, Any],
    dx: float = 1.0,
    dy: float = 1.0,
    window_size: int = 10,
    structural_index: float = 1.0,
    max_depth_uncertainty: float = 0.15,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Euler deconvolution for automated source depth estimation
    
    Euler's homogeneity equation relates field and derivatives to source location:
    (x - x₀)∂T/∂x + (y - y₀)∂T/∂y + (z - z₀)∂T/∂z = N * T
    
    Where N is the structural index describing source geometry.
    
    Args:
        data: Dictionary with magnetic/gravity field data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        window_size: Moving window size for local solutions
        structural_index: N value (0=contact, 1=sill/dike, 2=pipe, 3=sphere)
        max_depth_uncertainty: Maximum relative depth uncertainty (0-1)
        
    Returns:
        Dictionary with Euler solutions (x, y, depth, base level)
    """
    logger.info(f"📊 Euler Deconvolution (N={structural_index}, window={window_size})")
    
    z = data['z']
    ny, nx = z.shape
    
    # Calculate derivatives
    dy_deriv, dx_deriv = np.gradient(z, dy, dx)
    
    # Vertical derivative (approximation using second differences)
    dz_deriv = np.zeros_like(z)
    dz_deriv[1:-1, 1:-1] = (
        z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2] - 4 * z[1:-1, 1:-1]
    ) / 2.0
    
    # Storage for solutions
    solutions = []
    
    # Sliding window
    half_window = window_size // 2
    
    for i in range(half_window, ny - half_window, window_size // 2):
        for j in range(half_window, nx - half_window, window_size // 2):
            # Extract window
            i_start, i_end = i - half_window, i + half_window
            j_start, j_end = j - half_window, j + half_window
            
            # Window coordinates
            y_win = np.arange(i_start, i_end) * dy
            x_win = np.arange(j_start, j_end) * dx
            X_win, Y_win = np.meshgrid(x_win, y_win)
            
            # Extract window data
            T_win = z[i_start:i_end, j_start:j_end].flatten()
            dTdx_win = dx_deriv[i_start:i_end, j_start:j_end].flatten()
            dTdy_win = dy_deriv[i_start:i_end, j_start:j_end].flatten()
            dTdz_win = dz_deriv[i_start:i_end, j_start:j_end].flatten()
            X_flat = X_win.flatten()
            Y_flat = Y_win.flatten()
            
            # Build design matrix: [∂T/∂x, ∂T/∂y, ∂T/∂z, -N]
            A = np.column_stack([dTdx_win, dTdy_win, dTdz_win, -structural_index * np.ones(len(T_win))])
            b = X_flat * dTdx_win + Y_flat * dTdy_win
            
            # Solve least squares: A * [x₀, y₀, z₀, b]ᵀ = b
            try:
                solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                
                if len(residuals) > 0:
                    # Calculate uncertainty
                    n_obs = len(T_win)
                    rms_residual = np.sqrt(residuals[0] / n_obs)
                    
                    # Depth solution
                    x0, y0, z0, base = solution
                    
                    # Quality check: depth should be positive and reasonable
                    if z0 > 0 and z0 < 10000 and rms_residual < max_depth_uncertainty * np.abs(z0):
                        solutions.append({
                            'x': x0,
                            'y': y0,
                            'depth': z0,
                            'base_level': base,
                            'uncertainty': rms_residual
                        })
            except np.linalg.LinAlgError:
                continue
    
    logger.info(f"✅ Euler deconvolution completed - {len(solutions)} solutions found")
    
    result = {
        'solutions': solutions,
        'structural_index': structural_index,
        'window_size': window_size,
        'n_solutions': len(solutions),
        'processing_history': data.get('processing_history', []) + [
            f"Euler Deconvolution (N={structural_index}, {len(solutions)} solutions)"
        ]
    }
    
    return result


@register(
    name="source_parameter_imaging",
    category="advanced",
    description="SPI method for simultaneous depth and structural index estimation",
    keywords=["SPI", "source parameter", "depth", "structural index", "local wavenumber"],
    best_practices=[
        "More robust than Euler for noisy data",
        "Automatically determines structural index",
        "Works well with analytic signal",
        "Good for complex geology"
    ],
    references=[
        "Thurston & Smith (1997). Automatic conversion of magnetic data to depth",
        "Thurston et al. (2002). A multi-model method for depth estimation",
        "Smith et al. (1998). Improvements to local wavenumber methods"
    ]
)
def source_parameter_imaging(
    data: Dict[str, Any],
    dx: float = 1.0,
    dy: float = 1.0,
    min_depth: float = 0.0,
    max_depth: float = 5000.0,
    n_depth_tests: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Source Parameter Imaging for depth and structural index
    
    SPI uses local wavenumber analysis to simultaneously estimate:
    - Source depth
    - Structural index (source geometry)
    
    More robust than Euler deconvolution for noisy data.
    
    Args:
        data: Dictionary with magnetic/gravity field data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        min_depth: Minimum test depth (meters)
        max_depth: Maximum test depth (meters)
        n_depth_tests: Number of depth levels to test
        
    Returns:
        Dictionary with depth and structural index maps
    """
    logger.info(f"📊 Source Parameter Imaging ({n_depth_tests} depth levels)")
    
    from scipy.fft import fft2, ifft2, fftfreq
    
    z = data['z']
    ny, nx = z.shape
    
    # Calculate analytic signal components
    dy_deriv, dx_deriv = np.gradient(z, dy, dx)
    
    # Vertical derivative via FFT
    f_data = fft2(z)
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    f_vd = f_data * K
    vd = np.real(ifft2(f_vd))
    
    # Analytic signal amplitude
    A = np.sqrt(dx_deriv**2 + dy_deriv**2 + vd**2)
    
    # Local wavenumber (gradient of analytic signal phase)
    # k = |∇φ| where φ = atan2(vd, sqrt(dx²+dy²))
    horizontal_amp = np.sqrt(dx_deriv**2 + dy_deriv**2)
    
    # Avoid division by zero
    safe_A = np.where(A < 1e-10, 1e-10, A)
    safe_h = np.where(horizontal_amp < 1e-10, 1e-10, horizontal_amp)
    
    # Local wavenumber components
    k_x = dx_deriv / safe_A
    k_y = dy_deriv / safe_A
    k_z = vd / safe_A
    
    # Local wavenumber magnitude
    k_local = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    
    # Depth estimation: depth ≈ 1 / k_local
    depth_estimate = 1.0 / (k_local + 1e-10)
    
    # Clip to reasonable range
    depth_estimate = np.clip(depth_estimate, min_depth, max_depth)
    
    # Structural index from derivatives
    # N ≈ (horizontal_deriv / vertical_deriv) * depth
    structural_index = (horizontal_amp / (np.abs(vd) + 1e-10)) * k_local
    structural_index = np.clip(structural_index, 0, 3)
    
    # Calculate confidence based on analytic signal amplitude
    confidence = A / (np.max(A) + 1e-10)
    
    result = data.copy()
    result['depth_estimate'] = depth_estimate
    result['structural_index'] = structural_index
    result['confidence'] = confidence
    result['local_wavenumber'] = k_local
    result['processing_history'] = data.get('processing_history', []) + [
        f"Source Parameter Imaging (depth range: {min_depth}-{max_depth}m)"
    ]
    
    logger.info("✅ Source Parameter Imaging completed")
    return result


@register(
    name="werner_deconvolution",
    category="advanced",
    description="Werner deconvolution for contact and thin dike depth estimation",
    keywords=["werner", "depth", "contact", "dike", "profile", "2D"],
    best_practices=[
        "Works on profile data (2D)",
        "Good for contacts and thin dikes",
        "Requires isolated anomalies",
        "Best with high-quality data"
    ],
    references=[
        "Werner (1953). Interpretation of magnetic anomalies at sheet-like bodies",
        "Hartman et al. (1971). computer programs for Werner deconvolution",
        "Ku & Sharp (1983). Werner deconvolution for automated interpretation"
    ]
)
def werner_deconvolution(
    data: Dict[str, Any],
    dx: float = 1.0,
    profile_direction: str = 'x',
    window_size: int = 5,
    min_depth: float = 0.0,
    max_depth: float = 5000.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Werner deconvolution for contact/dike depth estimation
    
    Werner deconvolution assumes 2D sources (infinite strike length)
    and works on profile data to estimate:
    - Depth to top of source
    - Horizontal position
    - Susceptibility contrast (for thin dikes)
    
    Args:
        data: Dictionary with magnetic field data
        dx: Grid spacing along profile (meters)
        profile_direction: 'x' or 'y' for profile orientation
        window_size: Number of points in moving window
        min_depth: Minimum acceptable depth (meters)
        max_depth: Maximum acceptable depth (meters)
        
    Returns:
        Dictionary with Werner solutions for each profile
    """
    logger.info(f"📊 Werner Deconvolution (direction={profile_direction}, window={window_size})")
    
    z = data['z']
    
    # Extract profiles
    if profile_direction == 'x':
        profiles = z  # Each row is a profile
        n_profiles, n_points = z.shape
    else:  # 'y'
        profiles = z.T  # Each column becomes a profile
        n_profiles, n_points = z.T.shape
    
    all_solutions = []
    
    for profile_idx, profile in enumerate(profiles):
        # Calculate first derivative
        deriv1 = np.gradient(profile, dx)
        
        # Moving window analysis
        half_window = window_size // 2
        
        for i in range(half_window, n_points - half_window):
            # Extract window
            i_start, i_end = i - half_window, i + half_window + 1
            x_win = np.arange(i_start, i_end) * dx
            T_win = profile[i_start:i_end]
            dT_win = deriv1[i_start:i_end]
            
            # Werner's equations for thin dike:
            # T(x) = C * (x - x₀) / [(x - x₀)² + z₀²]
            # dT/dx = C * [z₀² - (x - x₀)²] / [(x - x₀)² + z₀²]²
            
            # Build design matrix for least squares
            # Linearize: T = a₁*(x-x₀) + a₂
            # dT/dx = b₁ + b₂*(x-x₀)²
            
            try:
                # Simple 3-point method for quick estimation
                if window_size >= 3:
                    # Use peak and adjacent points
                    max_idx = half_window
                    x0 = x_win[max_idx]
                    T0 = T_win[max_idx]
                    dT0 = dT_win[max_idx]
                    
                    # Estimate depth from peak width
                    # Half-width at half-maximum approximation
                    half_max = T0 / 2
                    left_idx = np.argmin(np.abs(T_win[:max_idx] - half_max))
                    right_idx = max_idx + np.argmin(np.abs(T_win[max_idx:] - half_max))
                    
                    half_width = (x_win[right_idx] - x_win[left_idx]) / 2
                    depth = half_width * 0.866  # Approximation for thin dike
                    
                    # Quality check
                    if min_depth <= depth <= max_depth and not np.isnan(depth):
                        position = (i * dx) if profile_direction == 'x' else None
                        
                        all_solutions.append({
                            'profile': profile_idx,
                            'x_position': position if profile_direction == 'x' else profile_idx * dx,
                            'y_position': profile_idx * dx if profile_direction == 'x' else position,
                            'depth': depth,
                            'amplitude': T0,
                            'quality': np.abs(dT0) / (np.abs(T0) + 1e-10)
                        })
            
            except Exception as e:
                continue
    
    logger.info(f"✅ Werner deconvolution completed - {len(all_solutions)} solutions")
    
    result = {
        'solutions': all_solutions,
        'profile_direction': profile_direction,
        'window_size': window_size,
        'n_solutions': len(all_solutions),
        'processing_history': data.get('processing_history', []) + [
            f"Werner Deconvolution ({profile_direction} profiles, {len(all_solutions)} solutions)"
        ]
    }
    
    return result


@register(
    name="tilt_depth_method",
    category="advanced",
    description="Depth estimation using tilt angle zero-crossing method",
    keywords=["tilt depth", "zero crossing", "depth", "automatic", "simple"],
    best_practices=[
        "Very simple and fast method",
        "Works best for isolated sources",
        "Depth = horizontal distance to zero contour",
        "Independent of source geometry"
    ],
    references=[
        "Salem et al. (2007). Tilt-depth method: A simple depth estimation method",
        "Salem et al. (2010). Depth to magnetic basement estimates"
    ]
)
def tilt_depth_method(
    data: Dict[str, Any],
    dx: float = 1.0,
    dy: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate source depth using tilt angle zero-crossing method
    
    The tilt-depth method uses the property that the zero contour
    of the tilt angle passes over source edges, and the horizontal
    distance from the maximum of the tilt angle to its zero crossing
    equals the source depth.
    
    Extremely simple but effective for isolated sources.
    
    Args:
        data: Dictionary with magnetic field data
        dx: Grid spacing in x-direction (meters)
        dy: Grid spacing in y-direction (meters)
        
    Returns:
        Dictionary with depth estimates
    """
    logger.info("📊 Tilt-Depth Method")
    
    from scipy.fft import fft2, ifft2, fftfreq
    from scipy.ndimage import maximum_filter, label
    
    z = data['z']
    ny, nx = z.shape
    
    # Calculate horizontal and vertical derivatives
    dy_deriv, dx_deriv = np.gradient(z, dy, dx)
    horizontal_grad = np.sqrt(dx_deriv**2 + dy_deriv**2)
    
    # Vertical derivative via FFT
    f_data = fft2(z)
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    f_vd = f_data * K
    vd = np.real(ifft2(f_vd))
    
    # Tilt angle
    tilt = np.arctan2(vd, horizontal_grad + 1e-10)
    
    # Find local maxima of tilt angle (source edges)
    tilt_maxima = maximum_filter(tilt, size=5)
    is_maximum = (tilt == tilt_maxima) & (tilt > 0.5)  # > ~30 degrees
    
    # Label connected regions
    labeled, n_features = label(is_maximum)
    
    depth_estimates = []
    
    for region_id in range(1, n_features + 1):
        # Get region coordinates
        region_mask = (labeled == region_id)
        y_coords, x_coords = np.where(region_mask)
        
        if len(x_coords) == 0:
            continue
        
        # Center of maximum region
        x_max = np.mean(x_coords) * dx
        y_max = np.mean(y_coords) * dy
        
        # Find nearest zero crossing
        # Look in small neighborhood
        window_size = 20
        i_center = int(np.mean(y_coords))
        j_center = int(np.mean(x_coords))
        
        i_start = max(0, i_center - window_size)
        i_end = min(ny, i_center + window_size)
        j_start = max(0, j_center - window_size)
        j_end = min(nx, j_center + window_size)
        
        # Extract local tilt
        local_tilt = tilt[i_start:i_end, j_start:j_end]
        
        # Find zero crossings
        zero_crossing = np.abs(local_tilt) < 0.1  # Close to zero
        
        if np.any(zero_crossing):
            y_zero, x_zero = np.where(zero_crossing)
            # Closest zero crossing
            distances = np.sqrt((x_zero - (j_center - j_start))**2 + 
                              (y_zero - (i_center - i_start))**2)
            closest_idx = np.argmin(distances)
            
            # Depth = horizontal distance to zero crossing
            depth = distances[closest_idx] * dx
            
            if depth > 0 and depth < 5000:  # Reasonable range
                depth_estimates.append({
                    'x': x_max,
                    'y': y_max,
                    'depth': depth,
                    'tilt_max': tilt[i_center, j_center]
                })
    
    logger.info(f"✅ Tilt-depth method completed - {len(depth_estimates)} estimates")
    
    result = {
        'depth_estimates': depth_estimates,
        'tilt_angle': tilt,
        'n_estimates': len(depth_estimates),
        'processing_history': data.get('processing_history', []) + [
            f"Tilt-Depth Method ({len(depth_estimates)} estimates)"
        ]
    }
    
    return result


# Initialize registry
logger.info("🧮 Advanced geophysical processing functions registered")
