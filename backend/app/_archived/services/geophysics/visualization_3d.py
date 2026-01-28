"""
3D Visualization Module using PyVista

This module provides comprehensive 3D visualization capabilities for geophysical data,
including volumetric rendering, prism models, and interactive visualization.

Based on PyVista (VTK wrapper) following scientific best practices.
"""

import numpy as np
import pyvista as pv
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Visualization3D:
    """
    3D visualization handler for geophysical data
    
    Provides methods for:
    - Volumetric rendering (voxel grids)
    - Prism model visualization
    - 3D grid surfaces
    - Interactive plotting
    """
    
    def __init__(self, notebook: bool = False, off_screen: bool = False):
        """
        Initialize 3D visualization
        
        Args:
            notebook: If True, use notebook-compatible rendering
            off_screen: If True, render without display (for saving images)
        """
        self.notebook = notebook
        self.off_screen = off_screen
        
        # Set PyVista preferences
        pv.set_plot_theme("document")
        if notebook:
            pv.set_jupyter_backend('static')  # Use static images in notebooks
    
    def create_plotter(self, 
                      window_size: Tuple[int, int] = (1024, 768),
                      title: str = "GeoBot 3D Visualization") -> pv.Plotter:
        """
        Create a PyVista plotter with standard settings
        
        Args:
            window_size: Window dimensions (width, height)
            title: Window title
            
        Returns:
            Configured PyVista plotter
        """
        plotter = pv.Plotter(
            notebook=self.notebook,
            off_screen=self.off_screen,
            window_size=window_size,
            title=title
        )
        
        # Add standard elements
        plotter.add_axes()
        plotter.show_grid()
        
        return plotter
    
    def visualize_grid_surface(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               z: np.ndarray,
                               scalars: np.ndarray,
                               title: str = "Grid Surface",
                               cmap: str = "viridis",
                               show_edges: bool = False,
                               opacity: float = 1.0,
                               save_path: Optional[str] = None) -> Optional[pv.Plotter]:
        """
        Visualize 2D grid as 3D surface
        
        Args:
            x: X coordinates (1D or 2D array)
            y: Y coordinates (1D or 2D array)
            z: Z coordinates/elevation (2D array)
            scalars: Data values to display (2D array)
            title: Plot title
            cmap: Colormap name
            show_edges: Show mesh edges
            opacity: Surface opacity (0-1)
            save_path: Path to save image (if provided)
            
        Returns:
            Plotter instance (if not off_screen)
        """
        # Create meshgrid if needed
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        
        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)
        grid.point_data['values'] = scalars.ravel()
        
        # Create plotter
        plotter = self.create_plotter(title=title)
        
        # Add surface
        plotter.add_mesh(
            grid,
            scalars='values',
            cmap=cmap,
            show_edges=show_edges,
            opacity=opacity,
            scalar_bar_args={'title': 'Values', 'vertical': True}
        )
        
        # Save or show
        if save_path:
            plotter.screenshot(save_path)
            logger.info(f"Saved 3D visualization to {save_path}")
            plotter.close()
            return None
        
        if not self.off_screen:
            plotter.show()
            return plotter
        
        return None
    
    def visualize_prism_model(self,
                             prisms: List[Dict],
                             field_data: Optional[Dict] = None,
                             title: str = "Prism Model",
                             prism_color: str = "lightblue",
                             prism_opacity: float = 0.6,
                             show_edges: bool = True,
                             save_path: Optional[str] = None) -> Optional[pv.Plotter]:
        """
        Visualize 3D prism model with optional field surface
        
        Args:
            prisms: List of prism dictionaries with keys:
                   - bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
                   - density: Density value (optional)
                   - magnetization: Magnetization vector (optional)
            field_data: Optional dict with keys:
                       - x, y: Coordinate arrays
                       - z: Elevation array
                       - values: Field values to display
            title: Plot title
            prism_color: Color for prisms
            prism_opacity: Prism transparency (0-1)
            show_edges: Show prism edges
            save_path: Path to save image
            
        Returns:
            Plotter instance (if not off_screen)
        """
        plotter = self.create_plotter(title=title)
        
        # Add prisms
        for i, prism in enumerate(prisms):
            bounds = prism['bounds']
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            
            # Create box for prism
            box = pv.Box(bounds=bounds)
            
            # Determine color based on density/magnetization
            color = prism_color
            if 'density' in prism:
                # Could color-code by density
                pass
            
            plotter.add_mesh(
                box,
                color=color,
                opacity=prism_opacity,
                show_edges=show_edges,
                label=f"Prism {i+1}"
            )
        
        # Add field surface if provided
        if field_data is not None:
            x = field_data['x']
            y = field_data['y']
            z = field_data.get('z', np.zeros_like(field_data['values']))
            values = field_data['values']
            
            if x.ndim == 1:
                x, y = np.meshgrid(x, y)
            
            grid = pv.StructuredGrid(x, y, z)
            grid.point_data['field'] = values.ravel()
            
            plotter.add_mesh(
                grid,
                scalars='field',
                cmap='seismic',
                opacity=0.8,
                scalar_bar_args={'title': 'Field Values', 'vertical': True}
            )
        
        # Add legend
        plotter.add_legend()
        
        # Save or show
        if save_path:
            plotter.screenshot(save_path)
            logger.info(f"Saved 3D prism model to {save_path}")
            plotter.close()
            return None
        
        if not self.off_screen:
            plotter.show()
            return plotter
        
        return None
    
    def visualize_volume(self,
                        volume_data: np.ndarray,
                        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                        title: str = "Volume Rendering",
                        cmap: str = "coolwarm",
                        opacity: str = "linear",
                        save_path: Optional[str] = None) -> Optional[pv.Plotter]:
        """
        Volumetric rendering of 3D data
        
        Args:
            volume_data: 3D numpy array (nz, ny, nx)
            spacing: Grid spacing (dx, dy, dz)
            origin: Origin coordinates (x0, y0, z0)
            title: Plot title
            cmap: Colormap name
            opacity: Opacity transfer function ('linear', 'sigmoid', or custom array)
            save_path: Path to save image
            
        Returns:
            Plotter instance (if not off_screen)
        """
        # Create uniform grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(volume_data.shape) + 1
        grid.spacing = spacing
        grid.origin = origin
        
        # Add volume data
        grid.cell_data['values'] = volume_data.flatten(order='F')
        
        # Create plotter
        plotter = self.create_plotter(title=title)
        
        # Add volume
        plotter.add_volume(
            grid,
            scalars='values',
            cmap=cmap,
            opacity=opacity,
            scalar_bar_args={'title': 'Values', 'vertical': True}
        )
        
        # Save or show
        if save_path:
            plotter.screenshot(save_path)
            logger.info(f"Saved volume rendering to {save_path}")
            plotter.close()
            return None
        
        if not self.off_screen:
            plotter.show()
            return plotter
        
        return None
    
    def visualize_isosurface(self,
                            volume_data: np.ndarray,
                            isovalues: Union[float, List[float]],
                            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                            origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                            title: str = "Isosurface",
                            colors: Optional[List[str]] = None,
                            opacity: float = 0.7,
                            save_path: Optional[str] = None) -> Optional[pv.Plotter]:
        """
        Visualize isosurfaces in 3D volume
        
        Args:
            volume_data: 3D numpy array
            isovalues: Single value or list of values for isosurfaces
            spacing: Grid spacing
            origin: Origin coordinates
            title: Plot title
            colors: Colors for each isosurface
            opacity: Surface opacity
            save_path: Path to save image
            
        Returns:
            Plotter instance (if not off_screen)
        """
        # Ensure isovalues is a list
        if isinstance(isovalues, (int, float)):
            isovalues = [isovalues]
        
        # Create uniform grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(volume_data.shape) + 1
        grid.spacing = spacing
        grid.origin = origin
        grid.cell_data['values'] = volume_data.flatten(order='F')
        
        # Create plotter
        plotter = self.create_plotter(title=title)
        
        # Default colors if not provided
        if colors is None:
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
        
        # Add isosurfaces
        for i, isovalue in enumerate(isovalues):
            contour = grid.contour(isosurfaces=[isovalue], scalars='values')
            color = colors[i % len(colors)]
            
            plotter.add_mesh(
                contour,
                color=color,
                opacity=opacity,
                show_edges=False,
                label=f"Iso = {isovalue}"
            )
        
        # Add legend
        plotter.add_legend()
        
        # Save or show
        if save_path:
            plotter.screenshot(save_path)
            logger.info(f"Saved isosurface to {save_path}")
            plotter.close()
            return None
        
        if not self.off_screen:
            plotter.show()
            return plotter
        
        return None
    
    def create_animation(self,
                        frames: List[np.ndarray],
                        output_path: str,
                        fps: int = 10,
                        title: str = "Animation") -> None:
        """
        Create animation from sequence of 2D arrays
        
        Args:
            frames: List of 2D numpy arrays (one per frame)
            output_path: Path to save GIF/MP4
            fps: Frames per second
            title: Animation title
        """
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        plotter.open_gif(output_path, fps=fps)
        
        # Create initial grid
        ny, nx = frames[0].shape
        x = np.arange(nx)
        y = np.arange(ny)
        x, y = np.meshgrid(x, y)
        
        for i, frame in enumerate(frames):
            plotter.clear()
            plotter.add_title(f"{title} - Frame {i+1}/{len(frames)}")
            
            grid = pv.StructuredGrid(x, y, np.zeros_like(x))
            grid.point_data['values'] = frame.ravel()
            
            plotter.add_mesh(
                grid,
                scalars='values',
                cmap='viridis',
                show_edges=False
            )
            
            plotter.write_frame()
        
        plotter.close()
        logger.info(f"Saved animation to {output_path}")


# Convenience functions

def visualize_grid_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                      scalars: np.ndarray, **kwargs) -> None:
    """Quick 3D grid visualization"""
    viz = Visualization3D()
    viz.visualize_grid_surface(x, y, z, scalars, **kwargs)


def visualize_prisms_3d(prisms: List[Dict], field_data: Optional[Dict] = None, 
                       **kwargs) -> None:
    """Quick prism model visualization"""
    viz = Visualization3D()
    viz.visualize_prism_model(prisms, field_data, **kwargs)


def visualize_volume_3d(volume_data: np.ndarray, **kwargs) -> None:
    """Quick volume rendering"""
    viz = Visualization3D()
    viz.visualize_volume(volume_data, **kwargs)
