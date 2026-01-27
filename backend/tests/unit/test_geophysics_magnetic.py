"""
Unit Tests for Magnetic Processing Functions
Tests for reduction to pole, derivatives, transforms, etc.
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


class TestReductionToPole:
    """Tests for RTP function"""
    
    def test_rtp_basic(self, sample_grid_data):
        """Test basic RTP execution"""
        result = reduction_to_pole(
            sample_grid_data,
            inclination=-30.0,
            declination=0.0
        )
        
        assert result['success'] is True
        assert 'data' in result['result']
        assert result['result']['metadata']['function'] == 'reduction_to_pole'
    
    def test_rtp_preserves_shape(self, sample_grid_data):
        """Test that RTP preserves data shape"""
        result = reduction_to_pole(
            sample_grid_data,
            inclination=-30.0,
            declination=0.0
        )
        
        data = np.array(result['result']['data'])
        original = np.array(sample_grid_data['z'])
        
        assert data.shape == original.shape
    
    def test_rtp_at_pole(self, sample_grid_data):
        """Test RTP at magnetic pole (should be identity)"""
        result = reduction_to_pole(
            sample_grid_data,
            inclination=90.0,
            declination=0.0
        )
        
        data = np.array(result['result']['data'])
        original = np.array(sample_grid_data['z'])
        
        # At pole, RTP should change data minimally
        assert np.allclose(data, original, rtol=0.1)
    
    def test_rtp_invalid_inclination(self, sample_grid_data):
        """Test RTP with invalid inclination"""
        result = reduction_to_pole(
            sample_grid_data,
            inclination=100.0,  # Invalid
            declination=0.0
        )
        
        assert result['success'] is False
        assert result['error'] is not None


class TestUpwardContinuation:
    """Tests for upward continuation"""
    
    def test_uc_basic(self, sample_grid_data):
        """Test basic upward continuation"""
        result = upward_continuation(
            sample_grid_data,
            altitude=500.0
        )
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_uc_amplitude_decrease(self, sample_grid_data):
        """Test that UC decreases amplitude"""
        result = upward_continuation(
            sample_grid_data,
            altitude=1000.0
        )
        
        data = np.array(result['result']['data'])
        original = np.array(sample_grid_data['z'])
        
        # Upward continuation should smooth and reduce amplitude
        assert np.max(np.abs(data)) < np.max(np.abs(original))
    
    def test_uc_zero_altitude(self, sample_grid_data):
        """Test UC with zero altitude (identity)"""
        result = upward_continuation(
            sample_grid_data,
            altitude=0.0
        )
        
        data = np.array(result['result']['data'])
        original = np.array(sample_grid_data['z'])
        
        assert np.allclose(data, original, rtol=0.01)


class TestAnalyticSignal:
    """Tests for analytic signal"""
    
    def test_as_basic(self, sample_grid_data):
        """Test basic analytic signal"""
        result = analytic_signal(sample_grid_data)
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_as_positive(self, sample_grid_data):
        """Test that AS is always positive"""
        result = analytic_signal(sample_grid_data)
        
        data = np.array(result['result']['data'])
        
        # Analytic signal should always be >= 0
        assert np.all(data >= 0)
    
    def test_as_edge_detection(self, sample_grid_data):
        """Test that AS enhances edges"""
        result = analytic_signal(sample_grid_data)
        
        data = np.array(result['result']['data'])
        
        # AS should have peak at anomaly center/edges
        assert np.max(data) > 0


class TestTotalHorizontalDerivative:
    """Tests for THD"""
    
    def test_thd_basic(self, sample_grid_data):
        """Test basic THD"""
        result = total_horizontal_derivative(sample_grid_data)
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_thd_positive(self, sample_grid_data):
        """Test that THD is positive"""
        result = total_horizontal_derivative(sample_grid_data)
        
        data = np.array(result['result']['data'])
        
        assert np.all(data >= 0)


class TestVerticalDerivative:
    """Tests for vertical derivative"""
    
    def test_vd_basic(self, sample_grid_data):
        """Test basic vertical derivative"""
        result = vertical_derivative(
            sample_grid_data,
            order=1
        )
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_vd_orders(self, sample_grid_data):
        """Test different orders of VD"""
        for order in [1, 2, 3]:
            result = vertical_derivative(
                sample_grid_data,
                order=order
            )
            
            assert result['success'] is True
            assert result['result']['metadata']['params']['order'] == order


class TestTiltDerivative:
    """Tests for tilt derivative"""
    
    def test_tilt_basic(self, sample_grid_data):
        """Test basic tilt derivative"""
        result = tilt_derivative(sample_grid_data)
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_tilt_range(self, sample_grid_data):
        """Test that tilt is in valid range"""
        result = tilt_derivative(sample_grid_data)
        
        data = np.array(result['result']['data'])
        
        # Tilt should be between -90 and 90 degrees
        assert np.all(data >= -90)
        assert np.all(data <= 90)


class TestPseudogravity:
    """Tests for pseudogravity transform"""
    
    def test_pg_basic(self, sample_grid_data):
        """Test basic pseudogravity"""
        result = pseudogravity(
            sample_grid_data,
            inclination=-30.0,
            declination=0.0
        )
        
        assert result['success'] is True
        assert 'data' in result['result']


class TestMatchedFilter:
    """Tests for matched filter"""
    
    def test_mf_basic(self, sample_grid_data):
        """Test basic matched filter"""
        result = matched_filter(
            sample_grid_data,
            target_depth=1000.0,
            si=3
        )
        
        assert result['success'] is True
        assert 'data' in result['result']
    
    def test_mf_si_values(self, sample_grid_data):
        """Test different SI values"""
        for si in [0, 1, 2, 3]:
            result = matched_filter(
                sample_grid_data,
                target_depth=1000.0,
                si=si
            )
            
            assert result['success'] is True


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestMagneticWorkflow:
    """Integration tests for magnetic processing workflows"""
    
    def test_complete_workflow(self, sample_grid_data):
        """Test complete magnetic processing workflow"""
        # Step 1: RTP
        rtp_result = reduction_to_pole(
            sample_grid_data,
            inclination=-30.0,
            declination=0.0
        )
        assert rtp_result['success'] is True
        
        # Step 2: Upward continuation on RTP result
        rtp_data = {
            'x': sample_grid_data['x'],
            'y': sample_grid_data['y'],
            'z': rtp_result['result']['data'],
            'nx': sample_grid_data['nx'],
            'ny': sample_grid_data['ny'],
        }
        
        uc_result = upward_continuation(rtp_data, altitude=500.0)
        assert uc_result['success'] is True
        
        # Step 3: THD on UC result
        uc_data = {
            'x': sample_grid_data['x'],
            'y': sample_grid_data['y'],
            'z': uc_result['result']['data'],
            'nx': sample_grid_data['nx'],
            'ny': sample_grid_data['ny'],
        }
        
        thd_result = total_horizontal_derivative(uc_data)
        assert thd_result['success'] is True
        
        # Verify metadata chain
        assert 'function' in thd_result['result']['metadata']
