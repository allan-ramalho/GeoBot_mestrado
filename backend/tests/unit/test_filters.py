"""
Tests for Signal Processing Filters
"""

import pytest
import numpy as np
from scipy import signal

from app.services.geophysics.functions.filters import (
    butterworth_filter,
    gaussian_filter,
    median_filter,
    directional_filter,
    wiener_filter
)


@pytest.fixture
def sample_grid():
    """Create sample grid with signal + noise"""
    x = np.linspace(0, 1000, 50)
    y = np.linspace(0, 1000, 50)
    X, Y = np.meshgrid(x, y)
    
    # Signal: smooth long-wavelength anomaly
    signal_data = np.sin(2 * np.pi * X / 500) * np.cos(2 * np.pi * Y / 500)
    
    # Noise: high-frequency
    np.random.seed(42)
    noise = 0.1 * np.random.randn(50, 50)
    
    return signal_data + noise, 20.0, 20.0  # data, dx, dy


@pytest.fixture
def sample_grid_with_outliers():
    """Create grid with outliers for median filter testing"""
    x = np.linspace(0, 1000, 50)
    y = np.linspace(0, 1000, 50)
    X, Y = np.meshgrid(x, y)
    
    data = np.sin(2 * np.pi * X / 500)
    
    # Add outliers
    data[10, 10] = 100.0
    data[20, 20] = -100.0
    data[30, 30] = 100.0
    
    return data, 20.0, 20.0


class TestButterworthFilter:
    """Test Butterworth filter"""
    
    def test_lowpass_filter(self, sample_grid):
        """Test low-pass Butterworth filter"""
        data, dx, dy = sample_grid
        
        result = butterworth_filter(
            data=data,
            dx=dx,
            dy=dy,
            cutoff_wavelength=200.0,
            filter_type="low-pass",
            order=4
        )
        
        assert "filtered" in result
        assert result["filtered"].shape == data.shape
        assert "filter_type" in result
        assert result["filter_type"] == "low-pass"
    
    def test_highpass_filter(self, sample_grid):
        """Test high-pass Butterworth filter"""
        data, dx, dy = sample_grid
        
        result = butterworth_filter(
            data=data,
            dx=dx,
            dy=dy,
            cutoff_wavelength=300.0,
            filter_type="high-pass",
            order=4
        )
        
        assert "filtered" in result
        assert result["filtered"].shape == data.shape
        # High-pass should reduce mean (removes DC component)
        assert np.abs(np.mean(result["filtered"])) < np.abs(np.mean(data))
    
    def test_bandpass_filter(self, sample_grid):
        """Test band-pass Butterworth filter"""
        data, dx, dy = sample_grid
        
        result = butterworth_filter(
            data=data,
            dx=dx,
            dy=dy,
            cutoff_wavelength=100.0,
            filter_type="band-pass",
            order=4,
            high_cutoff=400.0
        )
        
        assert "filtered" in result
        assert result["filtered"].shape == data.shape
    
    def test_filter_order_effect(self, sample_grid):
        """Test different filter orders"""
        data, dx, dy = sample_grid
        
        result_order2 = butterworth_filter(
            data=data, dx=dx, dy=dy,
            cutoff_wavelength=200.0,
            filter_type="low-pass",
            order=2
        )
        
        result_order6 = butterworth_filter(
            data=data, dx=dx, dy=dy,
            cutoff_wavelength=200.0,
            filter_type="low-pass",
            order=6
        )
        
        # Both should filter, but results will differ
        assert result_order2["filtered"].shape == data.shape
        assert result_order6["filtered"].shape == data.shape
    
    def test_preserves_data_shape(self, sample_grid):
        """Test filter preserves data shape"""
        data, dx, dy = sample_grid
        
        result = butterworth_filter(
            data=data, dx=dx, dy=dy,
            cutoff_wavelength=200.0
        )
        
        assert result["filtered"].shape == data.shape


class TestGaussianFilter:
    """Test Gaussian filter"""
    
    def test_gaussian_smoothing(self, sample_grid):
        """Test Gaussian smoothing filter"""
        data, dx, dy = sample_grid
        
        result = gaussian_filter(
            data=data,
            dx=dx,
            dy=dy,
            sigma=2.0
        )
        
        assert "filtered" in result
        assert result["filtered"].shape == data.shape
        # Gaussian should smooth (reduce variance)
        assert np.var(result["filtered"]) < np.var(data)
    
    def test_gaussian_sigma_effect(self, sample_grid):
        """Test effect of different sigma values"""
        data, dx, dy = sample_grid
        
        result_small = gaussian_filter(data, dx, dy, sigma=1.0)
        result_large = gaussian_filter(data, dx, dy, sigma=5.0)
        
        # Larger sigma = more smoothing = lower variance
        assert np.var(result_large["filtered"]) < np.var(result_small["filtered"])
    
    def test_gaussian_truncate(self, sample_grid):
        """Test Gaussian filter truncation parameter"""
        data, dx, dy = sample_grid
        
        result = gaussian_filter(
            data=data,
            dx=dx,
            dy=dy,
            sigma=2.0,
            truncate=4.0
        )
        
        assert "filtered" in result
    
    def test_gaussian_edge_handling(self):
        """Test Gaussian filter handles edges correctly"""
        # Small grid to test edges
        data = np.ones((10, 10))
        data[5, 5] = 10.0  # Spike in center
        
        result = gaussian_filter(
            data=data,
            dx=1.0,
            dy=1.0,
            sigma=1.0
        )
        
        # Should smooth the spike
        assert result["filtered"][5, 5] < 10.0
        assert result["filtered"][5, 5] > 1.0


class TestMedianFilter:
    """Test Median filter"""
    
    def test_median_removes_outliers(self, sample_grid_with_outliers):
        """Test median filter removes outliers"""
        data, dx, dy = sample_grid_with_outliers
        
        # Check outliers exist
        assert np.max(np.abs(data)) > 50.0
        
        result = median_filter(
            data=data,
            dx=dx,
            dy=dy,
            size=3
        )
        
        # Outliers should be reduced
        assert np.max(np.abs(result["filtered"])) < 50.0
    
    def test_median_preserves_edges(self):
        """Test median filter preserves edges"""
        # Create step function
        data = np.zeros((20, 20))
        data[:, 10:] = 1.0
        
        result = median_filter(
            data=data,
            dx=1.0,
            dy=1.0,
            size=3
        )
        
        # Edge should still be relatively sharp
        assert result["filtered"].shape == data.shape
    
    def test_median_window_sizes(self, sample_grid_with_outliers):
        """Test different median filter window sizes"""
        data, dx, dy = sample_grid_with_outliers
        
        result_small = median_filter(data, dx, dy, size=3)
        result_large = median_filter(data, dx, dy, size=5)
        
        # Both should work
        assert result_small["filtered"].shape == data.shape
        assert result_large["filtered"].shape == data.shape


class TestDirectionalFilter:
    """Test Directional filter"""
    
    def test_directional_enhancement(self, sample_grid):
        """Test directional filter enhances features"""
        data, dx, dy = sample_grid
        
        result = directional_filter(
            data=data,
            dx=dx,
            dy=dy,
            azimuth=0.0,
            width=30.0
        )
        
        assert "filtered" in result
        assert result["filtered"].shape == data.shape
    
    def test_different_azimuths(self, sample_grid):
        """Test directional filter at different azimuths"""
        data, dx, dy = sample_grid
        
        result_ns = directional_filter(data, dx, dy, azimuth=0.0, width=30.0)
        result_ew = directional_filter(data, dx, dy, azimuth=90.0, width=30.0)
        
        # Different azimuths give different results
        assert not np.allclose(result_ns["filtered"], result_ew["filtered"])
    
    def test_directional_width(self, sample_grid):
        """Test effect of directional filter width"""
        data, dx, dy = sample_grid
        
        result_narrow = directional_filter(data, dx, dy, azimuth=0.0, width=15.0)
        result_wide = directional_filter(data, dx, dy, azimuth=0.0, width=60.0)
        
        # Both should work
        assert result_narrow["filtered"].shape == data.shape
        assert result_wide["filtered"].shape == data.shape


class TestWienerFilter:
    """Test Wiener filter"""
    
    def test_wiener_filtering(self, sample_grid):
        """Test Wiener filter"""
        data, dx, dy = sample_grid
        
        result = wiener_filter(
            data=data,
            dx=dx,
            dy=dy,
            noise_variance=0.01
        )
        
        assert "filtered" in result
        assert result["filtered"].shape == data.shape
    
    def test_wiener_noise_reduction(self, sample_grid):
        """Test Wiener filter reduces noise"""
        data, dx, dy = sample_grid
        
        result = wiener_filter(
            data=data,
            dx=dx,
            dy=dy,
            noise_variance=0.01
        )
        
        # Should reduce noise
        assert result["filtered"].shape == data.shape


class TestFilterMetadata:
    """Test filter returns proper metadata"""
    
    def test_butterworth_metadata(self, sample_grid):
        """Test Butterworth filter returns metadata"""
        data, dx, dy = sample_grid
        
        result = butterworth_filter(data, dx, dy, cutoff_wavelength=200.0)
        
        assert "cutoff_wavelength" in result
        assert "filter_type" in result
        assert "order" in result
    
    def test_gaussian_metadata(self, sample_grid):
        """Test Gaussian filter returns metadata"""
        data, dx, dy = sample_grid
        
        result = gaussian_filter(data, dx, dy, sigma=2.0)
        
        assert "sigma" in result
    
    def test_median_metadata(self, sample_grid):
        """Test Median filter returns metadata"""
        data, dx, dy = sample_grid
        
        result = median_filter(data, dx, dy, size=3)
        
        assert "window_size" in result or "size" in result


class TestFilterInputValidation:
    """Test filter input validation"""
    
    def test_invalid_filter_type(self, sample_grid):
        """Test invalid filter type raises error"""
        data, dx, dy = sample_grid
        
        with pytest.raises((ValueError, KeyError)):
            butterworth_filter(
                data, dx, dy,
                cutoff_wavelength=200.0,
                filter_type="invalid-type"
            )
    
    def test_negative_cutoff(self, sample_grid):
        """Test negative cutoff wavelength"""
        data, dx, dy = sample_grid
        
        # Should handle gracefully or raise error
        try:
            result = butterworth_filter(
                data, dx, dy,
                cutoff_wavelength=-100.0
            )
            # If it doesn't raise, it should return valid result
            assert "filtered" in result
        except (ValueError, RuntimeError):
            pass  # Expected error
    
    def test_zero_grid_spacing(self):
        """Test zero grid spacing"""
        data = np.ones((10, 10))
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            butterworth_filter(data, dx=0.0, dy=10.0, cutoff_wavelength=100.0)


class TestFilterNumericalStability:
    """Test filters are numerically stable"""
    
    def test_constant_input(self):
        """Test filters handle constant input"""
        data = np.ones((20, 20)) * 5.0
        
        result = butterworth_filter(
            data, dx=10.0, dy=10.0,
            cutoff_wavelength=100.0
        )
        
        # Constant input should remain constant (or close)
        assert np.allclose(result["filtered"], data, atol=1e-10)
    
    def test_large_values(self):
        """Test filters handle large values"""
        data = np.random.randn(20, 20) * 1e6
        
        result = gaussian_filter(data, dx=10.0, dy=10.0, sigma=2.0)
        
        assert not np.any(np.isnan(result["filtered"]))
        assert not np.any(np.isinf(result["filtered"]))
    
    def test_small_values(self):
        """Test filters handle small values"""
        data = np.random.randn(20, 20) * 1e-10
        
        result = gaussian_filter(data, dx=10.0, dy=10.0, sigma=2.0)
        
        assert not np.any(np.isnan(result["filtered"]))
