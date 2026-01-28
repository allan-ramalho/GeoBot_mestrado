"""
Unit Tests for Magnetic Processing Functions
Tests for reduction to pole, derivatives, transforms, etc.
CORRECTED VERSION - Compatible with current API
"""

import pytest
import numpy as np
from app.services.geophysics.functions.magnetic import (
    reduction_to_pole,
    upward_continuation,
    analytic_signal,
    total_horizontal_derivative,
    vertical_derivative,
    tilt_derivative,
    pseudogravity,
    matched_filter,
)


# Helper function to extract numpy array from function result
def extract_result(func_output):
    """
    Extract numpy array from function output
    Magnetic functions return dict with 'z' field containing the data array
    """
    if isinstance(func_output, dict):
        # Magnetic functions return {'x': ..., 'y': ..., 'z': array, 'shape': ...}
        if 'z' in func_output:
            return np.array(func_output['z']) if not isinstance(func_output['z'], np.ndarray) else func_output['z']
        # Fallback for other formats
        if 'result' in func_output:
            result = func_output['result']
            if isinstance(result, dict) and 'z' in result:
                return np.array(result['z'])
            return np.array(result) if not isinstance(result, np.ndarray) else result
        if 'data' in func_output:
            return np.array(func_output['data'])
    # Direct numpy array
    return np.array(func_output) if not isinstance(func_output, np.ndarray) else func_output


class TestReductionToPole:
    """Tests for RTP function"""
    
    def test_rtp_basic(self):
        """Test basic RTP execution"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        
        result = reduction_to_pole(data, inclination=-30.0, declination=0.0)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape
    
    def test_rtp_preserves_shape(self):
        """Test that RTP preserves data shape"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = reduction_to_pole(data, inclination=-30.0, declination=0.0)
        result_array = extract_result(result)
        
        assert result_array.shape == z.shape
    
    def test_rtp_at_pole(self):
        """Test RTP at magnetic pole"""
        nx, ny = 30, 30
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = reduction_to_pole(data, inclination=90.0, declination=0.0)
        result_array = extract_result(result)
        
        assert result_array.shape == z.shape


class TestUpwardContinuation:
    """Tests for upward continuation"""
    
    def test_uc_basic(self):
        """Test basic upward continuation"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = upward_continuation(data, height=500.0)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape
    
    def test_uc_amplitude_decrease(self):
        """Test that UC decreases amplitude"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = upward_continuation(data, height=1000.0)
        result_array = extract_result(result)
        
        # Upward continuation should smooth and reduce amplitude
        assert np.max(np.abs(result_array)) < np.max(np.abs(z))
    
    def test_uc_zero_height(self):
        """Test UC with zero height (should be near identity)"""
        nx, ny = 30, 30
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = upward_continuation(data, height=0.0)
        result_array = extract_result(result)
        
        # At zero height, should be very close to original
        assert np.allclose(result_array, z, rtol=0.1)


class TestAnalyticSignal:
    """Tests for analytic signal"""
    
    def test_as_basic(self):
        """Test basic analytic signal"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = analytic_signal(data)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape
    
    def test_as_positive(self):
        """Test that AS is always positive"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = analytic_signal(data)
        result_array = extract_result(result)
        
        # Analytic signal should always be >= 0
        assert np.all(result_array >= -0.01)  # Allow small numerical errors


class TestTotalHorizontalDerivative:
    """Tests for THD"""
    
    def test_thd_basic(self):
        """Test basic THD"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = total_horizontal_derivative(data)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape


class TestVerticalDerivative:
    """Tests for vertical derivative"""
    
    def test_vd_basic(self):
        """Test basic vertical derivative"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = vertical_derivative(data, order=1)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape


class TestTiltDerivative:
    """Tests for tilt derivative"""
    
    def test_tilt_basic(self):
        """Test basic tilt derivative"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = tilt_derivative(data)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape


class TestPseudogravity:
    """Tests for pseudogravity transform"""
    
    def test_pg_basic(self):
        """Test basic pseudogravity"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = pseudogravity(data, inclination=-30.0, declination=0.0)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape


class TestMatchedFilter:
    """Tests for matched filter"""
    
    def test_mf_basic(self):
        """Test basic matched filter"""
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        result = matched_filter(data, target_depth=1000.0, si=3)
        
        assert isinstance(result, dict)
        result_array = extract_result(result)
        assert result_array.shape == z.shape


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestMagneticWorkflow:
    """Integration tests for magnetic processing workflows"""
    
    def test_complete_workflow(self):
        """Test complete magnetic processing workflow"""
        # Create test data
        nx, ny = 50, 50
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        xx, yy = np.meshgrid(x, y)
        z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
        
        data = {'x': x, 'y': y, 'z': z, 'shape': z.shape}
        
        # Step 1: RTP
        rtp_result = reduction_to_pole(data, inclination=-30.0, declination=0.0)
        assert isinstance(rtp_result, dict)
        assert 'z' in rtp_result
        
        # Step 2: Upward continuation (pass dict directly)
        uc_result = upward_continuation(rtp_result, height=500.0)
        assert isinstance(uc_result, dict)
        assert 'z' in uc_result
        
        # Step 3: THD (pass dict directly)
        thd_result = total_horizontal_derivative(uc_result)
        assert isinstance(thd_result, dict)
        assert 'z' in thd_result
