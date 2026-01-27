"""
Performance Optimizations
Lazy loading, streaming, caching, and memory management
"""

import asyncio
import logging
from typing import Any, Dict, List, AsyncIterator, Optional
from functools import lru_cache, wraps
from datetime import datetime, timedelta
import hashlib
import json
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory usage optimization
    Monitors and manages memory consumption
    """
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.memory_usage: Dict[str, int] = {}
    
    def track_allocation(self, key: str, size_mb: float):
        """Track memory allocation"""
        self.memory_usage[key] = size_mb
        total = sum(self.memory_usage.values())
        
        if total > self.max_memory_mb:
            logger.warning(f"Memory usage ({total:.1f} MB) exceeds limit ({self.max_memory_mb} MB)")
            self._cleanup_old_allocations()
    
    def _cleanup_old_allocations(self):
        """Remove old allocations to free memory"""
        # Sort by size and remove largest first
        sorted_keys = sorted(
            self.memory_usage.keys(),
            key=lambda k: self.memory_usage[k],
            reverse=True
        )
        
        for key in sorted_keys[:len(sorted_keys)//2]:
            logger.info(f"Freeing memory: {key} ({self.memory_usage[key]:.1f} MB)")
            del self.memory_usage[key]
    
    def estimate_grid_size(self, nx: int, ny: int, dtype=np.float64) -> float:
        """Estimate memory size of grid in MB"""
        bytes_per_element = np.dtype(dtype).itemsize
        total_bytes = nx * ny * bytes_per_element
        return total_bytes / (1024 * 1024)


class LazyGrid:
    """
    Lazy loading for large grids
    Only loads data chunks as needed
    """
    
    def __init__(self, file_path: Path, chunk_size: int = 1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.metadata = self._load_metadata()
        self._cache: Dict[tuple, np.ndarray] = {}
    
    def _load_metadata(self) -> Dict:
        """Load grid metadata without loading data"""
        # Read header only
        with open(self.file_path, 'rb') as f:
            header_size = pickle.load(f)
            metadata = pickle.load(f)
        
        return metadata
    
    def get_chunk(self, x_start: int, x_end: int, y_start: int, y_end: int) -> np.ndarray:
        """Load a specific chunk of data"""
        key = (x_start, x_end, y_start, y_end)
        
        if key in self._cache:
            return self._cache[key]
        
        # Load chunk from file
        with open(self.file_path, 'rb') as f:
            # Skip to chunk position
            offset = self._calculate_offset(x_start, y_start)
            f.seek(offset)
            
            # Read chunk
            chunk_data = np.load(f)
        
        self._cache[key] = chunk_data
        return chunk_data
    
    def _calculate_offset(self, x: int, y: int) -> int:
        """Calculate byte offset for position"""
        # Simplified - actual implementation would be more complex
        return x * self.metadata['ny'] * 8 + y * 8
    
    @property
    def shape(self) -> tuple:
        """Grid shape"""
        return (self.metadata['nx'], self.metadata['ny'])


class StreamProcessor:
    """
    Streaming data processor
    Process large datasets in chunks
    """
    
    @staticmethod
    async def process_stream(
        data: np.ndarray,
        process_func: callable,
        chunk_size: int = 1000
    ) -> AsyncIterator[np.ndarray]:
        """
        Process data in chunks (streaming)
        
        Args:
            data: Input data array
            process_func: Function to apply to each chunk
            chunk_size: Size of chunks
        
        Yields:
            Processed chunks
        """
        nx, ny = data.shape
        
        for i in range(0, nx, chunk_size):
            for j in range(0, ny, chunk_size):
                # Extract chunk
                chunk = data[
                    i:min(i+chunk_size, nx),
                    j:min(j+chunk_size, ny)
                ]
                
                # Process chunk
                processed = await asyncio.to_thread(process_func, chunk)
                
                yield processed
                
                # Allow other coroutines to run
                await asyncio.sleep(0)
    
    @staticmethod
    async def process_large_grid(
        data: np.ndarray,
        process_func: callable,
        chunk_size: int = 1000,
        overlap: int = 50
    ) -> np.ndarray:
        """
        Process large grid with overlapping chunks
        
        Args:
            data: Input grid
            process_func: Processing function
            chunk_size: Chunk size
            overlap: Overlap between chunks
        
        Returns:
            Processed grid
        """
        nx, ny = data.shape
        result = np.zeros_like(data)
        
        for i in range(0, nx, chunk_size - overlap):
            for j in range(0, ny, chunk_size - overlap):
                # Extract chunk with overlap
                i_end = min(i + chunk_size, nx)
                j_end = min(j + chunk_size, ny)
                
                chunk = data[i:i_end, j:j_end]
                
                # Process
                processed = await asyncio.to_thread(process_func, chunk)
                
                # Merge (without overlap edges)
                result[
                    i+overlap//2:i_end-overlap//2,
                    j+overlap//2:j_end-overlap//2
                ] = processed[
                    overlap//2:-overlap//2 if i_end < nx else None,
                    overlap//2:-overlap//2 if j_end < ny else None
                ]
        
        return result


class ResultCache:
    """
    Cache for processing results
    Avoids recomputing identical operations
    """
    
    def __init__(self, cache_dir: Path = None, ttl_hours: int = 24):
        self.cache_dir = cache_dir or Path.home() / ".geobot" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _compute_key(self, function_id: str, params: Dict, data_hash: str) -> str:
        """Compute cache key"""
        key_data = {
            'function': function_id,
            'params': params,
            'data_hash': data_hash,
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _hash_data(self, data: np.ndarray) -> str:
        """Compute hash of data"""
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]
    
    def get(self, function_id: str, params: Dict, data: np.ndarray) -> Optional[Any]:
        """Get cached result"""
        data_hash = self._hash_data(data)
        key = self._compute_key(function_id, params, data_hash)
        
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check TTL
        modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - modified_time > self.ttl:
            cache_file.unlink()
            return None
        
        # Load cached result
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            
            logger.info(f"Cache hit: {function_id}")
            return result
            
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    def set(self, function_id: str, params: Dict, data: np.ndarray, result: Any):
        """Store result in cache"""
        data_hash = self._hash_data(data)
        key = self._compute_key(function_id, params, data_hash)
        
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            logger.info(f"Cached result: {function_id}")
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def get_size(self) -> float:
        """Get cache size in MB"""
        total_bytes = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*.pkl")
        )
        return total_bytes / (1024 * 1024)


# Decorator for caching function results
def cached_processing(cache: ResultCache):
    """
    Decorator to cache processing function results
    
    Usage:
        @cached_processing(cache_instance)
        def my_function(data, param1, param2):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(data: np.ndarray, **params) -> Any:
            # Try to get from cache
            cached = cache.get(func.__name__, params, data)
            if cached is not None:
                return cached
            
            # Compute result
            result = func(data, **params)
            
            # Store in cache
            cache.set(func.__name__, params, data, result)
            
            return result
        
        return wrapper
    return decorator


class ProgressTracker:
    """Track progress of long operations"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step: int = 1):
        """Update progress"""
        self.current_step += step
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current_step > 0:
            eta = elapsed / self.current_step * (self.total_steps - self.current_step)
        else:
            eta = 0
        
        return {
            'current': self.current_step,
            'total': self.total_steps,
            'percent': (self.current_step / self.total_steps) * 100,
            'elapsed': elapsed,
            'eta': eta,
        }


# Global instances
memory_manager = MemoryManager()
result_cache = ResultCache()


# Example usage
async def example_optimized_processing():
    """Example of using optimization features"""
    
    # Large dataset
    data = np.random.rand(10000, 10000)
    
    # Estimate memory
    size_mb = memory_manager.estimate_grid_size(*data.shape)
    memory_manager.track_allocation('input_data', size_mb)
    
    # Define processing function
    def process_chunk(chunk):
        return np.fft.fft2(chunk).real
    
    # Process with streaming
    result_chunks = []
    async for chunk in StreamProcessor.process_stream(data, process_chunk, chunk_size=1000):
        result_chunks.append(chunk)
    
    logger.info("Processing complete")
    
    # Cache example
    @cached_processing(result_cache)
    def expensive_operation(data, param1=1.0):
        return data * param1
    
    result = expensive_operation(data, param1=2.0)
