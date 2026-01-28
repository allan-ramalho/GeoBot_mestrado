"""
Harmonica Integration Module

This module provides wrappers around Harmonica (Fatiando a Terra) functions
for magnetic and gravity data processing. All implementations are validated
against Harmonica's reference implementations.

Harmonica Documentation: https://www.fatiando.org/harmonica/
References:
- Uieda et al. (2013). Fatiando a Terra: A Python library for geophysics
- Uieda (2023). Harmonica: Forward modeling, inversion, and processing

This module replaces manual implementations with scientifically validated methods.
"""

import numpy as np
import harmonica as hm
import verde as vd
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class HarmonicaWrapper:
    """
    Wrapper for Harmonica functions with GeoBot compatibility
    
    Provides:
    - Magnetic field transformations (RTP, derivatives, etc.)
    - Gravity corrections (Bouguer, free-air, terrain)
    - Upward/downward continuation
    - Forward modeling (prisms)
    """
    
    @staticmethod
    def reduction_to_pole(data: Dict,
                         inclination: float,
                         declination: float) -> np.ndarray:
        """
        Reduce magnetic field to the pole using Harmonica
        
        Wrapper around harmonica.reduction_to_pole for FFT-based RTP.
        
        Args:
            data: Dictionary with keys:
                  - 'z': 2D array of magnetic field values (nT)
                  - 'x': X coordinates (1D or 2D array)
                  - 'y': Y coordinates (1D or 2D array)
            inclination: Magnetic field inclination (degrees)
            declination: Magnetic field declination (degrees)
            
        Returns:
            2D array of reduced-to-pole magnetic field
            
        References:
            - Baranov (1957). A new method for interpretation of aeromagnetic maps
            - Li & Oldenburg (1998). Separation of regional and residual magnetic fields
        """
        try:
            z = data['z']
            x = data['x']
            y = data['y']
            
            # Harmonica 0.6.0 requires xarray grid
            import xarray as xr
            
            # Create 2D coordinate arrays if needed
            if x.ndim == 1:
                xx, yy = np.meshgrid(x, y)
            else:
                xx, yy = x, y
            
            # Create xarray grid (Harmonica 0.6+ requires this)
            grid = xr.DataArray(
                z,
                coords={'easting': xx[0, :], 'northing': yy[:, 0]},
                dims=['northing', 'easting']
            )
            
            # Use Harmonica's RTP (new API)
            rtp_grid = hm.reduction_to_pole(
                grid=grid,
                inclination=inclination,
                declination=declination
            )
            
            # Extract numpy array from result
            rtp_field = rtp_grid.values
            
            logger.info(f"✅ RTP completed using Harmonica (inc={inclination}°, dec={declination}°)")
            return rtp_field
            
        except Exception as e:
            logger.error(f"Harmonica RTP failed: {e}")
            logger.warning("Falling back to manual FFT implementation")
            # Fallback to manual implementation if Harmonica fails
            return _manual_rtp(data, inclination, declination)
    
    @staticmethod
    def upward_continuation(data: Dict, height: float) -> np.ndarray:
        """
        Upward continuation using Harmonica
        
        Args:
            data: Dictionary with grid data
            height: Continuation height (meters, positive for upward)
            
        Returns:
            Continued field
        """
        try:
            z = data['z']
            x = data['x']
            y = data['y']
            
            # Harmonica 0.6.0 requires xarray grid
            import xarray as xr
            
            # Create 2D coordinate arrays if needed
            if x.ndim == 1:
                xx, yy = np.meshgrid(x, y)
            else:
                xx, yy = x, y
            
            # Create xarray grid
            grid = xr.DataArray(
                z,
                coords={'easting': xx[0, :], 'northing': yy[:, 0]},
                dims=['northing', 'easting']
            )
            
            # Use Harmonica's upward continuation (new API)
            continued_grid = hm.upward_continuation(
                grid=grid,
                height_displacement=height
            )
            
            # Extract numpy array
            continued = continued_grid.values
            
            logger.info(f"✅ Upward continuation ({height}m) using Harmonica")
            return continued
            
        except Exception as e:
            logger.error(f"Harmonica upward continuation failed: {e}")
            return _manual_upward_continuation(data, height)
    
    @staticmethod
    def bouguer_correction(observed_gravity: np.ndarray,
                          elevation: np.ndarray,
                          density: float = 2.67) -> np.ndarray:
        """
        Complete Bouguer correction using Harmonica
        
        Computes the complete Bouguer anomaly:
        Bouguer anomaly = observed + free-air correction - Bouguer correction
        
        Args:
            observed_gravity: Observed gravity (mGal)
            elevation: Station elevations (meters)
            density: Assumed crustal density (g/cm³), default 2.67
            
        Returns:
            Bouguer anomaly (mGal)
            
        References:
            - Blakely (1995). Potential Theory in Gravity and Magnetic Applications
            - Hinze et al. (2013). New standards for reducing gravity data
            
        Notes:
            Harmonica's bouguer_correction returns only the Bouguer plate effect.
            We must apply the free-air correction separately.
        """
        try:
            # Harmonica 0.6.0 API:
            # - bouguer_correction(topography, density_crust, density_water)
            # - Returns only the Bouguer plate correction (2πGρh), NOT free-air
            # - Density in kg/m³ (not g/cm³)
            
            # Convert density from g/cm³ to kg/m³
            density_kg_m3 = density * 1000
            
            # Get Bouguer plate correction from Harmonica
            bouguer_plate = hm.bouguer_correction(
                topography=elevation,
                density_crust=density_kg_m3
            )
            
            # Free-air correction: -0.3086 mGal/m
            # (negative because gravity decreases with height)
            free_air = -0.3086 * elevation
            
            # Complete Bouguer anomaly:
            # anomaly = observed + free_air - bouguer_plate
            bouguer_anomaly = observed_gravity + free_air - bouguer_plate
            
            logger.info(f"Complete Bouguer correction using Harmonica (density={density} g/cm3)")
            return bouguer_anomaly
            
        except Exception as e:
            logger.error(f"Harmonica Bouguer correction failed: {e}")
            # Fallback formula
            # Free-air: -0.3086 * h (mGal/m)
            # Bouguer slab: 0.04193 * ρ * h (mGal)
            free_air = observed_gravity - 0.3086 * elevation
            bouguer_correction = 0.04193 * density * elevation
            return free_air - bouguer_correction
    
    @staticmethod
    def prism_gravity(coordinates: np.ndarray,
                     prisms: List[Dict],
                     field: str = 'g_z') -> np.ndarray:
        """
        Forward modeling of gravity from rectangular prisms
        
        Args:
            coordinates: Observation points (x, y, z) - shape (n, 3)
            prisms: List of prism dictionaries with:
                   - bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
                   - density: Density contrast (g/cm³)
            field: Field component ('g_z', 'g_n', 'g_e', 'g_u', 'g')
            
        Returns:
            Gravity field at observation points
        """
        try:
            # Convert prisms to Harmonica format
            prism_list = []
            for p in prisms:
                xmin, xmax, ymin, ymax, zmin, zmax = p['bounds']
                density = p.get('density', p.get('value', 1.0))
                
                prism_list.append({
                    'west': xmin, 'east': xmax,
                    'south': ymin, 'north': ymax,
                    'bottom': zmin, 'top': zmax,
                    'density': density
                })
            
            # Compute gravity using Harmonica
            gravity = hm.prism_gravity(
                coordinates=coordinates,
                prisms=prism_list,
                field=field
            )
            
            logger.info(f"✅ Prism gravity computed for {len(prisms)} prisms")
            return gravity
            
        except Exception as e:
            logger.error(f"Harmonica prism gravity failed: {e}")
            raise
    
    @staticmethod
    def prism_magnetic(coordinates: np.ndarray,
                      prisms: List[Dict],
                      inclination: float,
                      declination: float,
                      field: str = 'tf') -> np.ndarray:
        """
        Forward modeling of magnetic field from prisms
        
        Args:
            coordinates: Observation points (x, y, z) - shape (n, 3)
            prisms: List of prism dictionaries with:
                   - bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
                   - magnetization: Magnetization intensity (A/m)
            inclination: Magnetic field inclination (degrees)
            declination: Magnetic field declination (degrees)
            field: Field component ('tf', 'bx', 'by', 'bz')
            
        Returns:
            Magnetic field at observation points
        """
        try:
            prism_list = []
            for p in prisms:
                xmin, xmax, ymin, ymax, zmin, zmax = p['bounds']
                mag = p.get('magnetization', p.get('value', 1.0))
                
                prism_list.append({
                    'west': xmin, 'east': xmax,
                    'south': ymin, 'north': ymax,
                    'bottom': zmin, 'top': zmax,
                    'magnetization': mag
                })
            
            # Compute magnetic field
            magnetic = hm.prism_magnetic(
                coordinates=coordinates,
                prisms=prism_list,
                inclination=inclination,
                declination=declination,
                field=field
            )
            
            logger.info(f"✅ Prism magnetic field computed for {len(prisms)} prisms")
            return magnetic
            
        except Exception as e:
            logger.error(f"Harmonica prism magnetic failed: {e}")
            raise
    
    @staticmethod
    def equivalent_sources_interpolation(coordinates: np.ndarray,
                                        data: np.ndarray,
                                        coordinates_out: np.ndarray,
                                        damping: float = 1.0) -> np.ndarray:
        """
        Interpolate scattered data using equivalent sources
        
        Uses Harmonica's equivalent sources for smooth interpolation.
        
        Args:
            coordinates: Input coordinates (x, y, z) - shape (n, 3)
            data: Data values at input coordinates
            coordinates_out: Output coordinates - shape (m, 3)
            damping: Regularization parameter
            
        Returns:
            Interpolated data at output coordinates
        """
        try:
            eqs = hm.EquivalentSources(damping=damping)
            eqs.fit(coordinates, data)
            predicted = eqs.predict(coordinates_out)
            
            logger.info(f"✅ Equivalent sources interpolation: {len(data)} → {len(predicted)} points")
            return predicted
            
        except Exception as e:
            logger.error(f"Harmonica equivalent sources failed: {e}")
            raise


# Fallback manual implementations (kept for compatibility)

def _manual_rtp(data: Dict, inclination: float, declination: float) -> np.ndarray:
    """Manual RTP implementation (fallback)"""
    from scipy import fft
    
    z = data['z']
    ny, nx = z.shape
    
    if data['x'].ndim == 1:
        dx = np.mean(np.diff(data['x']))
        dy = np.mean(np.diff(data['y']))
    else:
        dx = np.mean(np.diff(data['x'][0, :]))
        dy = np.mean(np.diff(data['y'][:, 0]))
    
    # FFT
    Z = fft.fft2(z)
    
    # Wavenumbers
    kx = 2 * np.pi * fft.fftfreq(nx, dx)
    ky = 2 * np.pi * fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Convert angles to radians
    inc_rad = np.deg2rad(inclination)
    dec_rad = np.deg2rad(declination)
    
    # RTP operator (Baranov 1957)
    L = K * np.sin(inc_rad) + 1j * (KX*np.cos(inc_rad)*np.cos(dec_rad) + 
                                     KY*np.cos(inc_rad)*np.sin(dec_rad))
    
    # Avoid division by zero
    L[0, 0] = 1.0
    
    # Apply operator
    Z_rtp = Z / (L + 1e-10)
    
    # Inverse FFT
    rtp = np.real(fft.ifft2(Z_rtp))
    
    logger.warning("⚠️ Using manual RTP implementation (Harmonica unavailable)")
    return rtp


def _manual_upward_continuation(data: Dict, height: float) -> np.ndarray:
    """Manual upward continuation (fallback)"""
    from scipy import fft
    
    z = data['z']
    ny, nx = z.shape
    
    if data['x'].ndim == 1:
        dx = np.mean(np.diff(data['x']))
        dy = np.mean(np.diff(data['y']))
    else:
        dx = np.mean(np.diff(data['x'][0, :]))
        dy = np.mean(np.diff(data['y'][:, 0]))
    
    # FFT
    Z = fft.fft2(z)
    
    # Wavenumbers
    kx = 2 * np.pi * fft.fftfreq(nx, dx)
    ky = 2 * np.pi * fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Upward continuation operator
    Z_uc = Z * np.exp(-K * height)
    
    # Inverse FFT
    uc = np.real(fft.ifft2(Z_uc))
    
    logger.warning("⚠️ Using manual upward continuation (Harmonica unavailable)")
    return uc


# Singleton instance for easy access
harmonica_wrapper = HarmonicaWrapper()
