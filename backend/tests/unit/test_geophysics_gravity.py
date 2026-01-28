"""
Unit Tests for Gravity Processing Functions
Tests for Bouguer, free-air, terrain corrections, etc.
"""

import pytest
import numpy as np
from app.services.geophysics.functions.gravity import (
    bouguer_correction,
    free_air_correction,
    terrain_correction,
    isostatic_correction,
    regional_residual_separation,
)


class TestBouguerCorrection:
    """Tests for Bouguer correction"""
    
    def test_bouguer_basic(self):
        """Test basic Bouguer correction"""
        # Create sample data
        data = np.random.rand(50, 50) * 100  # Gravity data in mGal
        elevation = np.random.rand(50, 50) * 500  # Elevation in meters
        
        result = bouguer_correction(data, elevation, density=2.67)
        
        # Check return structure
        assert 'result' in result
        assert 'correction' in result
        assert 'metadata' in result
        assert result['metadata']['function'] == 'bouguer_correction'
        assert result['result'].shape == data.shape
    
    def test_bouguer_formula(self):
        """Test Bouguer correction formula (BC = 0.04193 * ρ * h)"""
        # Create simple grid with known height
        height = 100.0  # meters
        density = 2.67  # g/cm³
        
        data = np.ones((10, 10)) * 50.0  # Constant gravity
        elevation = np.ones((10, 10)) * height  # Constant elevation
        
        result = bouguer_correction(data, elevation, density=density)
        
        expected_correction = 0.04193 * density * height  # mGal
        
        # Check correction values
        assert np.allclose(result['correction'], expected_correction, rtol=0.01)
        
        # Check result (data - correction)
        expected_result = data - expected_correction
        assert np.allclose(result['result'], expected_result, rtol=0.01)
    
    def test_bouguer_different_densities(self):
        """Test Bouguer with different densities"""
        data = np.ones((20, 20)) * 100.0
        elevation = np.ones((20, 20)) * 200.0
        densities = [2.0, 2.67, 3.0, 3.5]
        
        for density in densities:
            result = bouguer_correction(data, elevation, density=density)
            
            assert 'result' in result
            assert result['metadata']['density_g_cm3'] == density
            
            # Verify formula
            expected_correction = 0.04193 * density * 200.0
            assert np.allclose(result['correction'][0, 0], expected_correction, rtol=0.01)


class TestFreeAirCorrection:
    """Tests for free-air correction"""
    
    def test_free_air_basic(self):
        """Test basic free-air correction"""
        data = np.random.rand(50, 50) * 100
        elevation = np.random.rand(50, 50) * 500
        
        result = free_air_correction(data, elevation)
        
        assert 'result' in result
        assert 'correction' in result
        assert 'metadata' in result
        assert result['result'].shape == data.shape
    
    def test_free_air_formula(self):
        """Test free-air formula (-0.3086 mGal/m)"""
        height = 100.0  # meters
        
        data = np.ones((10, 10)) * 50.0
        elevation = np.ones((10, 10)) * height
        
        result = free_air_correction(data, elevation)
        
        expected_correction = -0.3086 * height  # mGal
        
        assert np.allclose(result['correction'], expected_correction, rtol=0.01)
    
    def test_free_air_negative_height(self):
        """Test free-air with negative height (below sea level)"""
        data = np.ones((10, 10)) * 50.0
        elevation = np.ones((10, 10)) * -50.0  # Below sea level
        
        result = free_air_correction(data, elevation)
        
        # Correction should be positive for negative heights
        assert np.all(result['correction'] > 0)


class TestTerrainCorrection:
    """Tests for terrain correction"""
    
    def test_terrain_basic(self):
        """Test basic terrain correction"""
        # Create sample data
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        
        data = np.random.rand(ny, nx) * 100
        elevation = 1000 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (2000**2))
        
        # Create DEM (same as elevation for this test)
        dem = elevation.copy()
        dem_x = x
        dem_y = y
        
        result = terrain_correction(
            data, x, y, elevation, dem, dem_x, dem_y,
            density=2.67,
            radius=5000.0
        )
        
        assert 'result' in result
        assert result['result'].shape == data.shape
    
    def test_terrain_flat(self):
        """Test terrain correction on flat surface (should be near zero)"""
        nx, ny = 20, 20
        x = np.linspace(0, 1000, nx)
        y = np.linspace(0, 1000, ny)
        
        data = np.ones((ny, nx)) * 50.0
        elevation = np.ones((ny, nx)) * 100.0  # Flat surface
        
        dem = elevation.copy()
        dem_x = x
        dem_y = y
        
        result = terrain_correction(
            data, x, y, elevation, dem, dem_x, dem_y,
            density=2.67,
            radius=1000.0
        )
        
        # Flat terrain should have near-zero correction
        assert np.allclose(result['correction'], 0, atol=0.1)


class TestIsostaticCorrection:
    """Tests for isostatic correction"""
    
    def test_isostatic_basic(self):
        """Test basic isostatic correction"""
        data = np.random.rand(50, 50) * 100
        elevation = np.random.rand(50, 50) * 1000
        
        result = isostatic_correction(
            data,
            elevation,
            crustal_thickness=30000.0,
            density_crust=2.67,
            density_mantle=3.3
        )
        
        assert 'result' in result
        assert result['result'].shape == data.shape
    
    def test_isostatic_params(self):
        """Test isostatic with different parameters"""
        data = np.random.rand(20, 20) * 100
        elevation = np.random.rand(20, 20) * 1000
        
        result = isostatic_correction(
            data,
            elevation,
            crustal_thickness=35000.0,
            density_crust=2.8,
            density_mantle=3.4
        )
        
        assert 'result' in result


class TestRegionalResidualSeparation:
    """Tests for regional-residual separation"""
    
    def test_rr_polynomial(self):
        """Test polynomial regional-residual"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        
        data = np.random.rand(ny, nx) * 100
        
        result = regional_residual_separation(
            data, xx, yy,
            method='polynomial',
            order=2
        )
        
        assert 'regional' in result
        assert 'residual' in result
        assert result['regional'].shape == data.shape
        assert result['residual'].shape == data.shape
    
    def test_rr_upward(self):
        """Test upward continuation regional-residual"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        
        data = np.random.rand(ny, nx) * 100
        
        result = regional_residual_separation(
            data, xx, yy,
            method='upward_continuation',
            continuation_height=2000.0
        )
        
        assert 'regional' in result
        assert 'residual' in result
    
    def test_rr_sum(self):
        """Test that regional + residual ≈ original"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        
        data = np.random.rand(ny, nx) * 100
        
        result = regional_residual_separation(
            data, xx, yy,
            method='polynomial',
            order=2
        )
        
        regional = result['regional']
        residual = result['residual']
        
        reconstructed = regional + residual
        
        # Should approximately sum to original
        assert np.allclose(reconstructed, data, rtol=0.1)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestGravityWorkflow:
    """Integration tests for gravity processing workflows"""
    
    def test_complete_gravity_reduction(self):
        """Test complete gravity reduction workflow"""
        # Create sample data
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        
        data = np.random.rand(ny, nx) * 100
        elevation = 500 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (2000**2))
        
        # Step 1: Free-air correction
        fa_result = free_air_correction(data, elevation)
        assert 'result' in fa_result
        
        # Step 2: Bouguer correction
        bc_result = bouguer_correction(fa_result['result'], elevation, density=2.67)
        assert 'result' in bc_result
        
        # Step 3: Terrain correction
        dem = elevation.copy()
        dem_x = x
        dem_y = y
        
        tc_result = terrain_correction(
            bc_result['result'], x, y, elevation,
            dem, dem_x, dem_y,
            density=2.67,
            radius=5000.0
        )
        assert 'result' in tc_result
        
        # Step 4: Regional-residual
        rr_result = regional_residual_separation(
            tc_result['result'], xx, yy,
            method='polynomial',
            order=2
        )
        assert 'regional' in rr_result
        assert 'residual' in rr_result

