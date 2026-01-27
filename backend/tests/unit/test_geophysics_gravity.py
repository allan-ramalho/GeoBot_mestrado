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
    
    def test_bouguer_basic(self, sample_grid_data):
        """Test basic Bouguer correction"""
        result = bouguer_correction(
            sample_grid_data,
            density=2.67
        )
        
        assert result['success'] is True
        assert 'data' in result['result']
        assert result['result']['metadata']['function'] == 'bouguer_correction'
    
    def test_bouguer_formula(self):
        """Test Bouguer correction formula (BC = 0.04193 * ρ * h)"""
        # Create simple grid with known height
        height = 100.0  # meters
        density = 2.67  # g/cm³
        
        data = {
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'z': [[height] * 3] * 3,
            'nx': 3,
            'ny': 3,
        }
        
        result = bouguer_correction(data, density=density)
        
        expected_correction = 0.04193 * density * height  # mGal
        correction = np.array(result['result']['data'])
        
        assert np.allclose(correction, expected_correction, rtol=0.01)
    
    def test_bouguer_different_densities(self, sample_grid_data):
        """Test Bouguer with different densities"""
        densities = [2.0, 2.67, 3.0, 3.5]
        
        for density in densities:
            result = bouguer_correction(sample_grid_data, density=density)
            
            assert result['success'] is True
            assert result['result']['metadata']['params']['density'] == density


class TestFreeAirCorrection:
    """Tests for free-air correction"""
    
    def test_free_air_basic(self, sample_grid_data):
        """Test basic free-air correction"""
        result = free_air_correction(sample_grid_data)
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_free_air_formula(self):
        """Test free-air formula (-0.3086 mGal/m)"""
        height = 100.0  # meters
        
        data = {
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'z': [[height] * 3] * 3,
            'nx': 3,
            'ny': 3,
        }
        
        result = free_air_correction(data)
        
        expected_correction = -0.3086 * height  # mGal
        correction = np.array(result['result']['data'])
        
        assert np.allclose(correction, expected_correction, rtol=0.01)
    
    def test_free_air_negative_height(self):
        """Test free-air with negative height (below sea level)"""
        data = {
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'z': [[-50.0] * 3] * 3,
            'nx': 3,
            'ny': 3,
        }
        
        result = free_air_correction(data)
        
        correction = np.array(result['result']['data'])
        
        # Correction should be positive for negative heights
        assert np.all(correction > 0)


class TestTerrainCorrection:
    """Tests for terrain correction"""
    
    def test_terrain_basic(self, sample_grid_data):
        """Test basic terrain correction"""
        result = terrain_correction(
            sample_grid_data,
            density=2.67,
            radius=5000.0
        )
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_terrain_flat(self):
        """Test terrain correction on flat surface (should be zero)"""
        # Flat surface
        data = {
            'x': np.linspace(0, 1000, 10).tolist(),
            'y': np.linspace(0, 1000, 10).tolist(),
            'z': [[100.0] * 10] * 10,
            'nx': 10,
            'ny': 10,
        }
        
        result = terrain_correction(data, density=2.67, radius=1000.0)
        
        correction = np.array(result['result']['data'])
        
        # Flat terrain should have near-zero correction
        assert np.allclose(correction, 0, atol=0.1)


class TestIsostaticCorrection:
    """Tests for isostatic correction"""
    
    def test_isostatic_basic(self, sample_grid_data):
        """Test basic isostatic correction"""
        result = isostatic_correction(
            sample_grid_data,
            crustal_thickness=30000.0,
            density_crust=2.67,
            density_mantle=3.3
        )
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_isostatic_params(self):
        """Test isostatic with different parameters"""
        data = {
            'x': np.linspace(0, 10000, 20).tolist(),
            'y': np.linspace(0, 10000, 20).tolist(),
            'z': (np.random.rand(20, 20) * 1000).tolist(),
            'nx': 20,
            'ny': 20,
        }
        
        result = isostatic_correction(
            data,
            crustal_thickness=35000.0,
            density_crust=2.8,
            density_mantle=3.4
        )
        
        assert result['success'] is True


class TestRegionalResidualSeparation:
    """Tests for regional-residual separation"""
    
    def test_rr_polynomial(self, sample_grid_data):
        """Test polynomial regional-residual"""
        result = regional_residual_separation(
            sample_grid_data,
            method='polynomial',
            order=2
        )
        
        assert result['success'] is True
        assert 'regional' in result['result']
        assert 'residual' in result['result']
    
    def test_rr_upward(self, sample_grid_data):
        """Test upward continuation regional-residual"""
        result = regional_residual_separation(
            sample_grid_data,
            method='upward',
            altitude=2000.0
        )
        
        assert result['success'] is True
        assert 'regional' in result['result']
        assert 'residual' in result['result']
    
    def test_rr_sum(self, sample_grid_data):
        """Test that regional + residual ≈ original"""
        result = regional_residual_separation(
            sample_grid_data,
            method='polynomial',
            order=2
        )
        
        regional = np.array(result['result']['regional'])
        residual = np.array(result['result']['residual'])
        original = np.array(sample_grid_data['z'])
        
        reconstructed = regional + residual
        
        # Should approximately sum to original
        assert np.allclose(reconstructed, original, rtol=0.1)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestGravityWorkflow:
    """Integration tests for gravity processing workflows"""
    
    def test_complete_gravity_reduction(self, sample_grid_data):
        """Test complete gravity reduction workflow"""
        # Step 1: Free-air correction
        fa_result = free_air_correction(sample_grid_data)
        assert fa_result['success'] is True
        
        # Step 2: Bouguer correction
        fa_data = {
            'x': sample_grid_data['x'],
            'y': sample_grid_data['y'],
            'z': fa_result['result']['data'],
            'nx': sample_grid_data['nx'],
            'ny': sample_grid_data['ny'],
        }
        
        bc_result = bouguer_correction(fa_data, density=2.67)
        assert bc_result['success'] is True
        
        # Step 3: Terrain correction
        bc_data = {
            'x': sample_grid_data['x'],
            'y': sample_grid_data['y'],
            'z': bc_result['result']['data'],
            'nx': sample_grid_data['nx'],
            'ny': sample_grid_data['ny'],
        }
        
        tc_result = terrain_correction(bc_data, density=2.67, radius=5000.0)
        assert tc_result['success'] is True
        
        # Step 4: Regional-residual
        tc_data = {
            'x': sample_grid_data['x'],
            'y': sample_grid_data['y'],
            'z': tc_result['result']['data'],
            'nx': sample_grid_data['nx'],
            'ny': sample_grid_data['ny'],
        }
        
        rr_result = regional_residual_separation(
            tc_data,
            method='polynomial',
            order=2
        )
        assert rr_result['success'] is True
        
        # Verify all steps completed
        assert 'regional' in rr_result['result']
        assert 'residual' in rr_result['result']
