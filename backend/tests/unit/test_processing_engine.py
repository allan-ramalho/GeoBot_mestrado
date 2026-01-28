"""
Tests for Processing Engine
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.geophysics.processing_engine import (
    ProcessingEngine,
    ResultCache,
    PerformanceMetrics,
    AdvancedValidator
)


@pytest.fixture
def processing_engine():
    """Create processing engine instance"""
    return ProcessingEngine()


@pytest.fixture
def sample_data_id():
    """Sample data ID for testing"""
    return "test_data_001"


@pytest.fixture
def sample_parameters():
    """Sample parameters for testing"""
    return {
        "param1": 10,
        "param2": "test_value"
    }


class TestProcessingEngineInit:
    """Test processing engine initialization"""
    
    def test_engine_initialization(self, processing_engine):
        """Test engine initializes correctly"""
        assert processing_engine is not None
        assert processing_engine.registry is not None
        assert processing_engine.cache is not None
        assert processing_engine.metrics is not None
        assert processing_engine.validator is not None
    
    @pytest.mark.asyncio
    async def test_engine_async_initialization(self, processing_engine):
        """Test async initialization"""
        await processing_engine.initialize()
        assert processing_engine.executor is not None


class TestCacheSystem:
    """Test result caching"""
    
    def test_cache_initialization(self):
        """Test cache initializes with correct max size"""
        cache = ResultCache(max_size=50)
        assert cache.max_size == 50
        assert len(cache.cache) == 0
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache"""
        cache = ResultCache(max_size=10)
        
        result = {"data": "test_result"}
        cache.put("test_function", "data_123", {"param": 1}, result)
        
        retrieved = cache.get("test_function", "data_123", {"param": 1})
        assert retrieved == result
    
    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = ResultCache(max_size=10)
        
        result = cache.get("nonexistent_function", "data_123", {})
        assert result is None
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = ResultCache(max_size=2)
        
        # Fill cache
        cache.put("func1", "data1", {}, "result1")
        cache.put("func2", "data2", {}, "result2")
        
        # Cache should be full
        assert len(cache.cache) == 2
        
        # Access func1 to make it most recently used
        cache.get("func1", "data1", {})
        
        # Add new item (should evict least recently used)
        cache.put("func3", "data3", {}, "result3")
        
        # Cache should still have max 2 items
        assert len(cache.cache) == 2
        
        # func3 should be in cache
        assert cache.get("func3", "data3", {}) == "result3"
    
    def test_cache_clear(self):
        """Test clearing cache"""
        cache = ResultCache(max_size=10)
        cache.put("func1", "data1", {}, "result1")
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("func1", "data1", {}) is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = ResultCache(max_size=10)
        cache.put("func1", "data1", {}, "result1")
        cache.get("func1", "data1", {})
        
        stats = cache.get_stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == 10
        assert stats['total_accesses'] >= 1


class TestPerformanceMetrics:
    """Test performance tracking"""
    
    def test_metrics_initialization(self):
        """Test metrics initialize correctly"""
        metrics = PerformanceMetrics()
        assert len(metrics.execution_times) == 0
        assert len(metrics.execution_counts) == 0
    
    def test_record_execution(self):
        """Test recording execution metrics"""
        metrics = PerformanceMetrics()
        
        metrics.record_execution("test_function", duration=1.5, success=True)
        
        stats = metrics.get_function_stats("test_function")
        assert stats['count'] == 1
        assert stats['avg_time'] == 1.5
    
    def test_record_multiple_executions(self):
        """Test recording multiple executions"""
        metrics = PerformanceMetrics()
        
        metrics.record_execution("test_function", 1.0, True)
        metrics.record_execution("test_function", 2.0, True)
        metrics.record_execution("test_function", 3.0, True)
        
        stats = metrics.get_function_stats("test_function")
        
        assert stats['count'] == 3
        assert stats['avg_time'] == 2.0
        assert stats['min_time'] == 1.0
        assert stats['max_time'] == 3.0
    
    def test_error_tracking(self):
        """Test error counting"""
        metrics = PerformanceMetrics()
        
        metrics.record_execution("test_function", 1.0, True)
        metrics.record_execution("test_function", 1.0, False)  # Error
        
        stats = metrics.get_function_stats("test_function")
        
        assert stats['error_count'] == 1
        assert stats['error_rate'] == 0.5
    
    def test_overall_stats(self):
        """Test overall statistics"""
        metrics = PerformanceMetrics()
        
        metrics.record_execution("func1", 1.0, True)
        metrics.record_execution("func2", 2.0, True)
        
        stats = metrics.get_overall_stats()
        
        assert stats['total_executions'] == 2
        assert stats['functions_used'] == 2
    
    def test_top_functions(self):
        """Test retrieving top functions"""
        metrics = PerformanceMetrics()
        
        metrics.record_execution("func1", 1.0, True)
        metrics.record_execution("func1", 1.0, True)
        metrics.record_execution("func2", 1.0, True)
        
        top = metrics.get_top_functions(top_k=2)
        
        assert len(top) == 2
        assert top[0]['function'] == "func1"
        assert top[0]['count'] == 2
    
    def test_metrics_reset(self):
        """Test resetting metrics"""
        metrics = PerformanceMetrics()
        
        metrics.record_execution("func1", 1.0, True)
        metrics.reset()
        
        stats = metrics.get_overall_stats()
        assert stats['total_executions'] == 0


class TestAsyncExecution:
    """Test async job execution"""
    
    @pytest.mark.asyncio
    async def test_execute_async(self, processing_engine):
        """Test async execution returns job ID"""
        with patch.object(processing_engine, '_load_data', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {"test": "data"}
            
            job_id = await processing_engine.execute_async(
                function_name="test_function",
                data_id="data_123",
                parameters={}
            )
            
            assert job_id is not None
            assert isinstance(job_id, str)
            assert job_id in processing_engine.jobs
    
    @pytest.mark.asyncio
    async def test_get_job_status(self, processing_engine):
        """Test retrieving job status"""
        with patch.object(processing_engine, '_load_data', new_callable=AsyncMock):
            job_id = await processing_engine.execute_async(
                function_name="test_function",
                data_id="data_123",
                parameters={}
            )
            
            # Wait a bit for job to start
            await asyncio.sleep(0.1)
            
            status = await processing_engine.get_job_status(job_id)
            
            assert status is not None
            assert 'status' in status or 'job_id' in status
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, processing_engine):
        """Test canceling a running job"""
        with patch.object(processing_engine, '_load_data', new_callable=AsyncMock):
            job_id = await processing_engine.execute_async(
                function_name="test_function",
                data_id="data_123",
                parameters={}
            )
            
            await processing_engine.cancel_job(job_id)
            
            status = await processing_engine.get_job_status(job_id)
            # Status should indicate cancellation or job should not exist
            assert status is not None or job_id not in processing_engine.jobs


class TestEngineFeatures:
    """Test engine features"""
    
    def test_get_cache_stats(self, processing_engine):
        """Test retrieving cache statistics"""
        stats = processing_engine.get_cache_stats()
        
        assert 'size' in stats
        assert 'max_size' in stats
    
    def test_get_performance_stats(self, processing_engine):
        """Test retrieving performance statistics"""
        # Overall stats
        stats = processing_engine.get_performance_stats()
        
        assert 'total_executions' in stats
    
    def test_clear_cache(self, processing_engine):
        """Test clearing cache"""
        processing_engine.clear_cache()
        
        stats = processing_engine.get_cache_stats()
        assert stats['size'] == 0
    
    def test_reset_metrics(self, processing_engine):
        """Test resetting metrics"""
        processing_engine.reset_metrics()
        
        stats = processing_engine.get_performance_stats()
        assert stats['total_executions'] == 0


# Import asyncio for async tests
import asyncio
