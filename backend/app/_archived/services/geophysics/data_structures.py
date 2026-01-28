"""
Data Structure Classes for 1D, 2D, and 3D Geophysical Data

Provides comprehensive support for:
- 1D profile data (stations along line)
- 2D grid data (maps/surfaces)
- 3D volume data (voxels/prisms)

Follows best practices for geophysics data handling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Profile1D:
    """
    1D profile data structure for geophysical measurements
    
    Used for:
    - Gravity profiles
    - Magnetic profiles
    - Seismic sections
    - Well logs
    
    Attributes:
        distance: Distance along profile (meters)
        values: Measured values (e.g., mGal, nT)
        stations: Station coordinates (x, y, elevation)
        metadata: Additional information (survey name, date, etc.)
    """
    distance: np.ndarray
    values: np.ndarray
    stations: Optional[np.ndarray] = None  # Shape: (n_stations, 3) - x, y, z
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization"""
        self.distance = np.asarray(self.distance)
        self.values = np.asarray(self.values)
        
        if len(self.distance) != len(self.values):
            raise ValueError("Distance and values must have same length")
        
        if self.stations is not None:
            self.stations = np.asarray(self.stations)
            if self.stations.shape[0] != len(self.distance):
                raise ValueError("Stations must have same length as distance/values")
    
    @property
    def n_stations(self) -> int:
        """Number of stations in profile"""
        return len(self.distance)
    
    @property
    def extent(self) -> Tuple[float, float]:
        """Profile extent (min_distance, max_distance)"""
        return (self.distance.min(), self.distance.max())
    
    def interpolate(self, new_distance: np.ndarray, method: str = 'linear') -> 'Profile1D':
        """
        Interpolate profile to new distance points
        
        Args:
            new_distance: New distance array
            method: Interpolation method ('linear', 'cubic', 'nearest')
            
        Returns:
            New Profile1D instance with interpolated values
        """
        from scipy.interpolate import interp1d
        
        f = interp1d(self.distance, self.values, kind=method, 
                    fill_value='extrapolate')
        new_values = f(new_distance)
        
        return Profile1D(
            distance=new_distance,
            values=new_values,
            metadata=self.metadata.copy()
        )
    
    def apply_filter(self, filter_func, **kwargs) -> 'Profile1D':
        """
        Apply filter to profile data
        
        Args:
            filter_func: Function that takes values array and returns filtered values
            **kwargs: Additional arguments for filter_func
            
        Returns:
            New Profile1D with filtered values
        """
        filtered_values = filter_func(self.values, **kwargs)
        
        return Profile1D(
            distance=self.distance.copy(),
            values=filtered_values,
            stations=self.stations.copy() if self.stations is not None else None,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'distance': self.distance.tolist(),
            'values': self.values.tolist(),
            'stations': self.stations.tolist() if self.stations is not None else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Profile1D':
        """Create from dictionary"""
        return cls(
            distance=np.array(data['distance']),
            values=np.array(data['values']),
            stations=np.array(data['stations']) if data.get('stations') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class GridData2D:
    """
    2D grid data structure for geophysical maps
    
    Used for:
    - Gravity maps
    - Magnetic maps
    - Topography
    - Derivative maps
    
    Attributes:
        x: X coordinates (1D or 2D array)
        y: Y coordinates (1D or 2D array)
        values: Data values (2D array)
        metadata: Additional information
    """
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization"""
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        self.values = np.asarray(self.values)
        
        # Convert 1D coordinates to 2D if needed
        if self.x.ndim == 1 and self.y.ndim == 1:
            self.x, self.y = np.meshgrid(self.x, self.y)
        
        if self.x.shape != self.values.shape or self.y.shape != self.values.shape:
            raise ValueError("x, y, and values must have compatible shapes")
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (ny, nx)"""
        return self.values.shape
    
    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Grid extent (xmin, xmax, ymin, ymax)"""
        return (self.x.min(), self.x.max(), self.y.min(), self.y.max())
    
    @property
    def spacing(self) -> Tuple[float, float]:
        """Grid spacing (dx, dy)"""
        dx = np.mean(np.diff(self.x[0, :]))
        dy = np.mean(np.diff(self.y[:, 0]))
        return (dx, dy)
    
    def extract_profile(self, x0: float, y0: float, x1: float, y1: float, 
                       n_points: int = 100) -> Profile1D:
        """
        Extract 1D profile from 2D grid
        
        Args:
            x0, y0: Start coordinates
            x1, y1: End coordinates
            n_points: Number of points along profile
            
        Returns:
            Profile1D instance
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        x_unique = self.x[0, :]
        y_unique = self.y[:, 0]
        interp = RegularGridInterpolator((y_unique, x_unique), self.values)
        
        # Create profile points
        x_profile = np.linspace(x0, x1, n_points)
        y_profile = np.linspace(y0, y1, n_points)
        distance = np.linspace(0, np.sqrt((x1-x0)**2 + (y1-y0)**2), n_points)
        
        # Interpolate values
        points = np.column_stack([y_profile, x_profile])
        values = interp(points)
        
        # Create stations array
        stations = np.column_stack([x_profile, y_profile, np.zeros(n_points)])
        
        return Profile1D(
            distance=distance,
            values=values,
            stations=stations,
            metadata={'extracted_from': 'GridData2D', **self.metadata}
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'x': self.x.tolist(),
            'y': self.y.tolist(),
            'values': self.values.tolist(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GridData2D':
        """Create from dictionary"""
        return cls(
            x=np.array(data['x']),
            y=np.array(data['y']),
            values=np.array(data['values']),
            metadata=data.get('metadata', {})
        )


@dataclass
class VoxelModel3D:
    """
    3D voxel/volume data structure for geophysical models
    
    Used for:
    - 3D density models
    - 3D susceptibility models
    - Seismic velocity models
    - Inversion results
    
    Attributes:
        values: 3D array of property values (shape: nz, ny, nx)
        spacing: Grid spacing (dx, dy, dz) in meters
        origin: Origin coordinates (x0, y0, z0) in meters
        metadata: Additional information
    """
    values: np.ndarray
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization"""
        self.values = np.asarray(self.values)
        
        if self.values.ndim != 3:
            raise ValueError("Values must be a 3D array")
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Volume shape (nz, ny, nx)"""
        return self.values.shape
    
    @property
    def extent(self) -> Tuple[float, float, float, float, float, float]:
        """Volume extent (xmin, xmax, ymin, ymax, zmin, zmax)"""
        nz, ny, nx = self.shape
        dx, dy, dz = self.spacing
        x0, y0, z0 = self.origin
        
        return (
            x0, x0 + nx * dx,
            y0, y0 + ny * dy,
            z0, z0 + nz * dz
        )
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 3D coordinate arrays
        
        Returns:
            (X, Y, Z) 3D coordinate arrays
        """
        nz, ny, nx = self.shape
        dx, dy, dz = self.spacing
        x0, y0, z0 = self.origin
        
        x = x0 + np.arange(nx) * dx
        y = y0 + np.arange(ny) * dy
        z = z0 + np.arange(nz) * dz
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
        
        return X, Y, Z
    
    def extract_slice(self, axis: str, index: int) -> GridData2D:
        """
        Extract 2D slice from 3D volume
        
        Args:
            axis: 'x', 'y', or 'z'
            index: Index along specified axis
            
        Returns:
            GridData2D instance
        """
        nz, ny, nx = self.shape
        dx, dy, dz = self.spacing
        x0, y0, z0 = self.origin
        
        if axis == 'z':
            # Horizontal slice at depth z
            values = self.values[index, :, :]
            x = x0 + np.arange(nx) * dx
            y = y0 + np.arange(ny) * dy
            x, y = np.meshgrid(x, y)
            
        elif axis == 'y':
            # Vertical slice at y
            values = self.values[:, index, :]
            x = x0 + np.arange(nx) * dx
            z = z0 + np.arange(nz) * dz
            x, y = np.meshgrid(x, z)  # y is actually z here
            
        elif axis == 'x':
            # Vertical slice at x
            values = self.values[:, :, index]
            y = y0 + np.arange(ny) * dy
            z = z0 + np.arange(nz) * dz
            x, y = np.meshgrid(y, z)  # x is y, y is z
            
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        
        return GridData2D(
            x=x, y=y, values=values,
            metadata={'slice_axis': axis, 'slice_index': index, **self.metadata}
        )
    
    def add_prism(self, bounds: List[float], value: float):
        """
        Add a rectangular prism to the model
        
        Args:
            bounds: [xmin, xmax, ymin, ymax, zmin, zmax] in same units as origin/spacing
            value: Property value for prism (e.g., density, susceptibility)
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        dx, dy, dz = self.spacing
        x0, y0, z0 = self.origin
        
        # Convert bounds to indices
        ix_min = int((xmin - x0) / dx)
        ix_max = int((xmax - x0) / dx)
        iy_min = int((ymin - y0) / dy)
        iy_max = int((ymax - y0) / dy)
        iz_min = int((zmin - z0) / dz)
        iz_max = int((zmax - z0) / dz)
        
        # Clip to volume bounds
        nz, ny, nx = self.shape
        ix_min = max(0, min(ix_min, nx))
        ix_max = max(0, min(ix_max, nx))
        iy_min = max(0, min(iy_min, ny))
        iy_max = max(0, min(iy_max, ny))
        iz_min = max(0, min(iz_min, nz))
        iz_max = max(0, min(iz_max, nz))
        
        # Set values
        self.values[iz_min:iz_max, iy_min:iy_max, ix_min:ix_max] = value
        
        logger.info(f"Added prism: bounds={bounds}, value={value}")
    
    def to_prism_list(self, threshold: Optional[float] = None) -> List[Dict]:
        """
        Convert voxel model to list of prisms for forward modeling
        
        Args:
            threshold: Only include voxels with |value| > threshold
            
        Returns:
            List of prism dictionaries for use with Harmonica
        """
        prisms = []
        nz, ny, nx = self.shape
        dx, dy, dz = self.spacing
        x0, y0, z0 = self.origin
        
        # Iterate through voxels
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    value = self.values[iz, iy, ix]
                    
                    # Skip if below threshold
                    if threshold is not None and abs(value) < threshold:
                        continue
                    
                    # Skip if zero
                    if value == 0:
                        continue
                    
                    # Calculate bounds
                    xmin = x0 + ix * dx
                    xmax = x0 + (ix + 1) * dx
                    ymin = y0 + iy * dy
                    ymax = y0 + (iy + 1) * dy
                    zmin = z0 + iz * dz
                    zmax = z0 + (iz + 1) * dz
                    
                    prisms.append({
                        'bounds': [xmin, xmax, ymin, ymax, zmin, zmax],
                        'value': value
                    })
        
        logger.info(f"Converted to {len(prisms)} prisms (threshold={threshold})")
        return prisms
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'values': self.values.tolist(),
            'spacing': self.spacing,
            'origin': self.origin,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoxelModel3D':
        """Create from dictionary"""
        return cls(
            values=np.array(data['values']),
            spacing=tuple(data['spacing']),
            origin=tuple(data['origin']),
            metadata=data.get('metadata', {})
        )


# Convenience functions for creating data structures

def create_profile_from_points(x: np.ndarray, y: np.ndarray, 
                               values: np.ndarray, z: Optional[np.ndarray] = None) -> Profile1D:
    """
    Create Profile1D from scattered points
    
    Args:
        x, y: Coordinates
        values: Measured values
        z: Elevation (optional)
        
    Returns:
        Profile1D instance
    """
    # Calculate cumulative distance
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    distance = np.concatenate([[0], np.cumsum(distances)])
    
    # Create stations
    if z is None:
        z = np.zeros_like(x)
    stations = np.column_stack([x, y, z])
    
    return Profile1D(distance=distance, values=values, stations=stations)


def create_grid_from_points(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                            shape: Tuple[int, int], method: str = 'linear') -> GridData2D:
    """
    Create GridData2D from scattered points using interpolation
    
    Args:
        x, y: Scattered coordinates
        values: Measured values
        shape: Desired grid shape (ny, nx)
        method: Interpolation method
        
    Returns:
        GridData2D instance
    """
    from scipy.interpolate import griddata
    
    # Create regular grid
    xi = np.linspace(x.min(), x.max(), shape[1])
    yi = np.linspace(y.min(), y.max(), shape[0])
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate
    zi = griddata((x, y), values, (xi, yi), method=method)
    
    return GridData2D(x=xi, y=yi, values=zi)


def create_empty_voxel_model(shape: Tuple[int, int, int],
                             spacing: Tuple[float, float, float],
                             origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                             fill_value: float = 0.0) -> VoxelModel3D:
    """
    Create empty VoxelModel3D
    
    Args:
        shape: Volume shape (nz, ny, nx)
        spacing: Grid spacing (dx, dy, dz)
        origin: Origin coordinates
        fill_value: Initial value for all voxels
        
    Returns:
        VoxelModel3D instance
    """
    values = np.full(shape, fill_value, dtype=np.float32)
    
    return VoxelModel3D(values=values, spacing=spacing, origin=origin)
