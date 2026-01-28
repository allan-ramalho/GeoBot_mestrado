"""
Synthetic Benchmark Data for Scientific Validation

Provides canonical geophysics models with analytical solutions for:
- Testing and validation
- Benchmarking performance
- Educational examples
- Documentation

All models follow literature standards (Blakely 1995, Telford 1990, etc.)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SyntheticModels:
    """
    Factory for creating synthetic geophysical models
    
    All models include:
    - Known parameters (size, depth, properties)
    - Analytical or numerical solutions
    - Metadata for validation
    """
    
    @staticmethod
    def magnetic_sphere(
        radius: float = 50.0,
        depth: float = 200.0,
        magnetization: float = 1000.0,
        inclination: float = -30.0,
        declination: float = 0.0,
        grid_size: int = 100,
        grid_extent: float = 1000.0
    ) -> Dict:
        """
        Magnetic field from uniformly magnetized sphere
        
        Analytical solution available (useful for RTP validation)
        
        Args:
            radius: Sphere radius (meters)
            depth: Burial depth, positive downward (meters)
            magnetization: Total magnetization (A/m)
            inclination: Magnetization inclination (degrees)
            declination: Magnetization declination (degrees)
            grid_size: Number of points per axis
            grid_extent: Grid size (meters)
            
        Returns:
            Dict with 'x', 'y', 'z' (field), and 'model_params'
        """
        # Create observation grid
        x = np.linspace(0, grid_extent, grid_size)
        y = np.linspace(0, grid_extent, grid_size)
        X, Y = np.meshgrid(x, y)
        Z_obs = np.zeros_like(X)  # Observation at ground level
        
        # Sphere center
        x0 = grid_extent / 2
        y0 = grid_extent / 2
        z0 = -depth  # Negative because depth is positive downward
        
        # Distance from observation points to sphere center
        R = np.sqrt((X - x0)**2 + (Y - y0)**2 + (Z_obs - z0)**2)
        
        # Convert angles to radians
        inc_rad = np.deg2rad(inclination)
        dec_rad = np.deg2rad(declination)
        
        # Magnetization direction vector
        mx = magnetization * np.cos(inc_rad) * np.cos(dec_rad)
        my = magnetization * np.cos(inc_rad) * np.sin(dec_rad)
        mz = magnetization * np.sin(inc_rad)
        
        # Position vectors
        rx = X - x0
        ry = Y - y0
        rz = Z_obs - z0
        
        # Magnetic moment of sphere
        moment = (4 * np.pi / 3) * radius**3 * magnetization
        
        # Dipole field (simplified approximation for TMI)
        # For exact solution, would need full vector calculation
        # This gives vertical component (approximation of TMI at low latitudes)
        dot_product = mx * rx + my * ry + mz * rz
        Z_field = (moment / (4 * np.pi)) * (3 * dot_product * rz / R**5 - mz / R**3)
        
        logger.info(f"✅ Created magnetic sphere: r={radius}m, depth={depth}m, M={magnetization}A/m")
        
        return {
            'x': x,
            'y': y,
            'z': Z_field,
            'model_params': {
                'type': 'magnetic_sphere',
                'radius': radius,
                'depth': depth,
                'center': (x0, y0, z0),
                'magnetization': magnetization,
                'inclination': inclination,
                'declination': declination
            },
            'metadata': {
                'units': 'nT',
                'description': 'Magnetic field from uniformly magnetized sphere',
                'reference': 'Blakely (1995), Section 3.2'
            }
        }
    
    @staticmethod
    def gravity_sphere(
        radius: float = 50.0,
        depth: float = 200.0,
        density_contrast: float = 1.0,
        grid_size: int = 100,
        grid_extent: float = 1000.0
    ) -> Dict:
        """
        Gravity field from uniform density sphere
        
        Analytical solution (Blakely 1995, eq. 3.19)
        
        Args:
            radius: Sphere radius (meters)
            depth: Burial depth (meters)
            density_contrast: Density contrast (g/cm³)
            grid_size: Points per axis
            grid_extent: Grid size (meters)
            
        Returns:
            Dict with gravity anomaly
        """
        # Create grid
        x = np.linspace(0, grid_extent, grid_size)
        y = np.linspace(0, grid_extent, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Sphere center
        x0 = grid_extent / 2
        y0 = grid_extent / 2
        z0 = -depth
        
        # Distance to sphere center
        R = np.sqrt((X - x0)**2 + (Y - y0)**2 + (0 - z0)**2)
        
        # Gravitational constant
        G = 6.67430e-11  # m³/(kg·s²)
        G_mgal = G * 1e5  # Convert to mGal units
        
        # Sphere mass
        volume = (4 * np.pi / 3) * radius**3
        density_kg_m3 = density_contrast * 1000  # g/cm³ to kg/m³
        mass = volume * density_kg_m3
        
        # Vertical component of gravity (gz)
        gz = -G_mgal * mass * depth / R**3
        
        logger.info(f"✅ Created gravity sphere: r={radius}m, depth={depth}m, Δρ={density_contrast}g/cm³")
        
        return {
            'x': x,
            'y': y,
            'z': gz,
            'model_params': {
                'type': 'gravity_sphere',
                'radius': radius,
                'depth': depth,
                'center': (x0, y0, z0),
                'density_contrast': density_contrast
            },
            'metadata': {
                'units': 'mGal',
                'description': 'Gravity anomaly from uniform density sphere',
                'reference': 'Blakely (1995), Equation 3.19'
            }
        }
    
    @staticmethod
    def prism_gravity(
        bounds: Tuple[float, float, float, float, float, float],
        density_contrast: float = 1.0,
        grid_size: int = 100,
        grid_extent: float = 1000.0
    ) -> Dict:
        """
        Gravity from rectangular prism
        
        Uses numerical integration (can be validated against Harmonica)
        
        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax) in meters
            density_contrast: Density contrast (g/cm³)
            grid_size: Points per axis
            grid_extent: Grid size (meters)
            
        Returns:
            Dict with gravity field
        """
        try:
            import harmonica as hm
        except ImportError:
            logger.error("Harmonica required for prism gravity calculation")
            raise
        
        # Create observation grid
        x = np.linspace(0, grid_extent, grid_size)
        y = np.linspace(0, grid_extent, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)  # Observation at ground
        
        # Flatten for Harmonica
        coordinates = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Create prism
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        prisms = [{
            'west': xmin, 'east': xmax,
            'south': ymin, 'north': ymax,
            'bottom': zmin, 'top': zmax,
            'density': density_contrast
        }]
        
        # Calculate gravity using Harmonica
        gz = hm.prism_gravity(coordinates, prisms, field='g_z')
        gz = gz.reshape(X.shape)
        
        logger.info(f"✅ Created prism gravity: bounds={bounds}, Δρ={density_contrast}g/cm³")
        
        return {
            'x': x,
            'y': y,
            'z': gz,
            'model_params': {
                'type': 'prism_gravity',
                'bounds': bounds,
                'density_contrast': density_contrast
            },
            'metadata': {
                'units': 'mGal',
                'description': 'Gravity from rectangular prism (Harmonica)',
                'reference': 'Nagy et al. (2000)'
            }
        }
    
    @staticmethod
    def noisy_grid(
        clean_data: Dict,
        noise_level: float = 0.05,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Add Gaussian noise to clean data
        
        Args:
            clean_data: Dictionary with 'z' field
            noise_level: Noise level as fraction of signal std (e.g., 0.05 = 5%)
            seed: Random seed for reproducibility
            
        Returns:
            Noisy data dictionary
        """
        if seed is not None:
            np.random.seed(seed)
        
        z_clean = clean_data['z']
        signal_std = np.std(z_clean)
        noise = np.random.normal(0, noise_level * signal_std, z_clean.shape)
        
        z_noisy = z_clean + noise
        
        result = clean_data.copy()
        result['z'] = z_noisy
        result['metadata'] = result.get('metadata', {}).copy()
        result['metadata']['noise_added'] = {
            'noise_level': noise_level,
            'noise_std': noise_level * signal_std,
            'snr_db': 20 * np.log10(1 / noise_level)
        }
        
        logger.info(f"✅ Added {noise_level*100}% noise (SNR = {20*np.log10(1/noise_level):.1f} dB)")
        
        return result


class BenchmarkDatasets:
    """
    Standard benchmark datasets for testing
    
    These are fixed datasets for reproducible testing
    """
    
    @staticmethod
    def get_rtp_test_case() -> Dict:
        """
        Standard RTP test case
        
        Magnetic sphere at mid-latitude (I=-30°)
        Used for validating RTP implementation
        """
        return SyntheticModels.magnetic_sphere(
            radius=50,
            depth=200,
            magnetization=1000,
            inclination=-30,
            declination=0,
            grid_size=100,
            grid_extent=1000
        )
    
    @staticmethod
    def get_upward_continuation_test_case() -> Dict:
        """
        Standard upward continuation test
        
        High-frequency anomaly for testing smoothing
        """
        # Create high-frequency checkerboard pattern
        x = np.linspace(0, 1000, 100)
        y = np.linspace(0, 1000, 100)
        X, Y = np.meshgrid(x, y)
        
        # Checkerboard with multiple wavelengths
        Z = (100 * np.sin(2*np.pi*X/100) * np.cos(2*np.pi*Y/100) +
             50 * np.sin(2*np.pi*X/200) * np.cos(2*np.pi*Y/200) +
             25 * np.sin(2*np.pi*X/50) * np.cos(2*np.pi*Y/50))
        
        return {
            'x': x,
            'y': y,
            'z': Z,
            'model_params': {
                'type': 'checkerboard',
                'wavelengths': [100, 200, 50]
            },
            'metadata': {
                'description': 'Multi-frequency checkerboard for UC testing',
                'units': 'nT'
            }
        }
    
    @staticmethod
    def get_bouguer_test_case() -> Dict:
        """
        Standard Bouguer correction test
        
        Simple linear topography
        """
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        X, Y = np.meshgrid(x, y)
        
        # Linear slope
        elevation = 0.5 * X  # 0.5m per meter = 50% grade
        
        # Synthetic gravity (should be corrected)
        observed_gravity = -0.3086 * elevation + 0.04193 * 2.67 * elevation
        
        return {
            'observed_gravity': observed_gravity,
            'elevation': elevation,
            'x': x,
            'y': y,
            'model_params': {
                'type': 'linear_topography',
                'gradient': 0.5,
                'density': 2.67
            }
        }


# Convenience functions

def create_test_sphere_magnetic(**kwargs) -> Dict:
    """Quick magnetic sphere creation"""
    return SyntheticModels.magnetic_sphere(**kwargs)


def create_test_sphere_gravity(**kwargs) -> Dict:
    """Quick gravity sphere creation"""
    return SyntheticModels.gravity_sphere(**kwargs)


def create_test_prism(**kwargs) -> Dict:
    """Quick prism creation"""
    bounds = kwargs.pop('bounds', (400, 600, 400, 600, -300, -100))
    return SyntheticModels.prism_gravity(bounds=bounds, **kwargs)


# Export commonly used datasets
BENCHMARK_RTP = BenchmarkDatasets.get_rtp_test_case()
BENCHMARK_UC = BenchmarkDatasets.get_upward_continuation_test_case()
BENCHMARK_BOUGUER = BenchmarkDatasets.get_bouguer_test_case()
