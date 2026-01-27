"""
Processing Engine
Executes geophysical processing functions
Handles async processing, job management, and workflows
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
import json
import hashlib
import time
from functools import lru_cache
from collections import defaultdict

from app.services.geophysics.function_registry import get_registry
from app.core.config import settings

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Cache for processing results
    Avoids re-computing same operations
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.access_time: Dict[str, float] = {}
        logger.info(f"🗄️ Result cache initialized (max_size={max_size})")
    
    def _generate_key(
        self,
        function_name: str,
        data_id: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Generate cache key from inputs"""
        # Create deterministic hash
        params_str = json.dumps(parameters, sort_keys=True)
        key_str = f"{function_name}:{data_id}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self,
        function_name: str,
        data_id: str,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(function_name, data_id, parameters)
        
        if key in self.cache:
            self.access_count[key] += 1
            self.access_time[key] = time.time()
            logger.debug(f"✅ Cache hit: {function_name}")
            return self.cache[key]
        
        logger.debug(f"❌ Cache miss: {function_name}")
        return None
    
    def put(
        self,
        function_name: str,
        data_id: str,
        parameters: Dict[str, Any],
        result: Any
    ):
        """Store result in cache"""
        key = self._generate_key(function_name, data_id, parameters)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict least recently used
            lru_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
            del self.access_time[lru_key]
            logger.debug(f"🗑️ Evicted cache entry: {lru_key}")
        
        self.cache[key] = result
        self.access_count[key] = 1
        self.access_time[key] = time.time()
        logger.debug(f"💾 Cached result: {function_name}")
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.access_count.clear()
        self.access_time.clear()
        logger.info("🧹 Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'total_accesses': sum(self.access_count.values()),
            'unique_entries': len(self.cache)
        }


class PerformanceMetrics:
    """
    Track performance metrics for processing operations
    """
    
    def __init__(self):
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.execution_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.total_data_processed: int = 0
        logger.info("📊 Performance metrics initialized")
    
    def record_execution(
        self,
        function_name: str,
        duration: float,
        success: bool = True
    ):
        """Record function execution metrics"""
        self.execution_times[function_name].append(duration)
        self.execution_counts[function_name] += 1
        
        if not success:
            self.error_counts[function_name] += 1
        
        self.total_data_processed += 1
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistics for specific function"""
        times = self.execution_times.get(function_name, [])
        
        if not times:
            return {
                'count': 0,
                'avg_time': 0,
                'min_time': 0,
                'max_time': 0,
                'error_rate': 0
            }
        
        import statistics
        
        count = self.execution_counts[function_name]
        errors = self.error_counts.get(function_name, 0)
        
        return {
            'count': count,
            'avg_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times),
            'error_count': errors,
            'error_rate': errors / count if count > 0 else 0
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        all_times = []
        for times in self.execution_times.values():
            all_times.extend(times)
        
        total_errors = sum(self.error_counts.values())
        
        if not all_times:
            return {
                'total_executions': 0,
                'total_errors': 0,
                'avg_time': 0
            }
        
        import statistics
        
        return {
            'total_executions': sum(self.execution_counts.values()),
            'total_errors': total_errors,
            'avg_time': statistics.mean(all_times),
            'total_time': sum(all_times),
            'functions_used': len(self.execution_counts),
            'error_rate': total_errors / sum(self.execution_counts.values())
        }
    
    def get_top_functions(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get most used functions"""
        sorted_functions = sorted(
            self.execution_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {
                'function': func,
                'count': count,
                'stats': self.get_function_stats(func)
            }
            for func, count in sorted_functions
        ]
    
    def reset(self):
        """Reset all metrics"""
        self.execution_times.clear()
        self.execution_counts.clear()
        self.error_counts.clear()
        self.total_data_processed = 0
        logger.info("🔄 Metrics reset")


class AdvancedValidator:
    """
    Advanced parameter validation
    """
    
    @staticmethod
    def validate_parameters(
        function_name: str,
        parameters: Dict[str, Any],
        function_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate parameters against function requirements
        
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        
        # Check required parameters
        required_params = function_metadata.get('required_parameters', [])
        for param in required_params:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Check parameter types
        param_types = function_metadata.get('parameter_types', {})
        for param, value in parameters.items():
            expected_type = param_types.get(param)
            if expected_type and not isinstance(value, expected_type):
                errors.append(
                    f"Parameter '{param}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
        
        # Check parameter ranges
        param_ranges = function_metadata.get('parameter_ranges', {})
        for param, (min_val, max_val) in param_ranges.items():
            value = parameters.get(param)
            if value is not None:
                if value < min_val or value > max_val:
                    errors.append(
                        f"Parameter '{param}' out of range: "
                        f"{value} not in [{min_val}, {max_val}]"
                    )
        
        # Add warnings for best practices
        best_practices = function_metadata.get('best_practices', [])
        if best_practices:
            warnings.extend([f"Best practice: {bp}" for bp in best_practices])
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Processing job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingJob:
    """Processing job tracker"""
    
    def __init__(
        self,
        job_id: str,
        function_name: str,
        data_id: str,
        parameters: Dict[str, Any]
    ):
        self.job_id = job_id
        self.function_name = function_name
        self.data_id = data_id
        self.parameters = parameters
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "function_name": self.function_name,
            "data_id": self.data_id,
            "parameters": self.parameters,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "error": self.error
        }


class ProcessingEngine:
    """
    Enhanced geophysical data processing engine
    
    Features:
    - Result caching for performance
    - Performance metrics tracking
    - Advanced parameter validation
    - Job management and monitoring
    """
    
    def __init__(self):
        self.registry = get_registry()
        self.jobs: Dict[str, ProcessingJob] = {}
        self.executor = None
        
        # Enhanced features
        self.cache = ResultCache(max_size=100)
        self.metrics = PerformanceMetrics()
        self.validator = AdvancedValidator()
        
        logger.info("🚀 Enhanced processing engine initialized")
    
    async def initialize(self):
        """Initialize processing engine"""
        if not self.executor:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
            logger.info(f"✅ Processing engine initialized with {settings.MAX_WORKERS} workers")
    
    async def execute(
        self,
        function_name: str,
        data_id: str,
        parameters: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Execute processing function with caching and metrics
        
        Args:
            function_name: Name of function to execute
            data_id: ID of data to process
            parameters: Function parameters
            use_cache: Use cached result if available
        
        Returns:
            Processing result
        """
        start_time = time.time()
        success = True
        
        try:
            # Check cache first
            if use_cache:
                cached_result = self.cache.get(function_name, data_id, parameters)
                if cached_result is not None:
                    logger.info(f"✅ Using cached result for {function_name}")
                    return cached_result
            
            # Get function from registry
            func = self.registry.get_function(function_name)
            
            if not func:
                raise ValueError(f"Function not found: {function_name}")
            
            # Validate parameters
            func_metadata = self.registry.get_function_metadata(function_name)
            validation = self.validator.validate_parameters(
                function_name, parameters, func_metadata or {}
            )
            
            if not validation['valid']:
                raise ValueError(f"Parameter validation failed: {validation['errors']}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    logger.warning(f"⚠️ {warning}")
            
            # Load data
            data = await self._load_data(data_id)
            
            # Execute function
            logger.info(f"🔧 Executing {function_name} on data {data_id}")
            result = await self._execute_function(func, data, parameters)
            
            # Save result
            result_id = await self._save_result(result, data_id, function_name)
            
            response = {
                "status": "success",
                "result_id": result_id,
                "function": function_name,
                "summary": self._summarize_result(result),
                "cached": False
            }
            
            # Cache result
            if use_cache:
                self.cache.put(function_name, data_id, parameters, response)
            
            logger.info(f"✅ Processing completed: {function_name}")
            
            return response
            
        except Exception as e:
            success = False
            logger.error(f"❌ Processing failed: {e}")
            raise
        
        finally:
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_execution(function_name, duration, success)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def get_performance_stats(
        self,
        function_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Args:
            function_name: Specific function or None for overall stats
        
        Returns:
            Performance statistics
        """
        if function_name:
            return self.metrics.get_function_stats(function_name)
        else:
            return self.metrics.get_overall_stats()
    
    def get_top_functions(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get most used functions"""
        return self.metrics.get_top_functions(top_k)
    
    def clear_cache(self):
        """Clear result cache"""
        self.cache.clear()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset()
    
    async def execute_async(
        self,
        function_name: str,
        data_id: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Execute processing function asynchronously
        
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        job = ProcessingJob(
            job_id=job_id,
            function_name=function_name,
            data_id=data_id,
            parameters=parameters
        )
        
        self.jobs[job_id] = job
        
        # Start async execution
        asyncio.create_task(self._execute_job(job))
        
        logger.info(f"🚀 Started async job: {job_id}")
        
        return job_id
    
    async def _execute_job(self, job: ProcessingJob):
        """Execute job asynchronously"""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Execute
            result = await self.execute(
                function_name=job.function_name,
                data_id=job.data_id,
                parameters=job.parameters
            )
            
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 100.0
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            logger.error(f"❌ Job {job.job_id} failed: {e}")
    
    async def execute_workflow(
        self,
        data_id: str,
        steps: List[Dict[str, Any]],
        async_processing: bool = True
    ) -> str:
        """
        Execute workflow (chain of processing steps)
        
        Args:
            data_id: Initial data ID
            steps: List of processing steps
            async_processing: Execute asynchronously
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        if async_processing:
            asyncio.create_task(self._execute_workflow_async(job_id, data_id, steps))
            return job_id
        else:
            result = await self._execute_workflow_sync(data_id, steps)
            return result["result_id"]
    
    async def _execute_workflow_sync(
        self,
        data_id: str,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute workflow synchronously"""
        current_data_id = data_id
        results = []
        
        for i, step in enumerate(steps):
            logger.info(f"📋 Workflow step {i+1}/{len(steps)}: {step['function']}")
            
            result = await self.execute(
                function_name=step["function"],
                data_id=current_data_id,
                parameters=step.get("parameters", {})
            )
            
            results.append(result)
            current_data_id = result["result_id"]
        
        return {
            "status": "success",
            "result_id": current_data_id,
            "steps_completed": len(steps),
            "results": results
        }
    
    async def _execute_workflow_async(
        self,
        job_id: str,
        data_id: str,
        steps: List[Dict[str, Any]]
    ):
        """Execute workflow asynchronously"""
        job = ProcessingJob(
            job_id=job_id,
            function_name="workflow",
            data_id=data_id,
            parameters={"steps": steps}
        )
        
        self.jobs[job_id] = job
        
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            result = await self._execute_workflow_sync(data_id, steps)
            
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 100.0
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None
    
    async def cancel_job(self, job_id: str):
        """Cancel running job"""
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            logger.info(f"🛑 Job cancelled: {job_id}")
    
    async def interpret_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpret natural language command
        Maps to processing functions
        """
        # Search for matching functions
        functions = await self.registry.search_functions(command, top_k=3)
        
        if not functions:
            return {
                "understood": False,
                "message": "Could not understand command"
            }
        
        best_match = functions[0]
        
        return {
            "understood": True,
            "function": best_match["name"],
            "description": best_match["description"],
            "parameters": best_match["parameters"],
            "confidence": best_match["similarity_score"]
        }
    
    async def _execute_function(
        self,
        func: callable,
        data: Any,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute function with data and parameters"""
        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            func,
            data,
            **parameters
        )
        return result
    
    async def _load_data(self, data_id: str) -> Any:
        """Load data by ID"""
        # TODO: Implement data loading from storage
        # For now, return mock data
        import numpy as np
        return {
            "id": data_id,
            "x": np.arange(0, 100, 1),
            "y": np.arange(0, 100, 1),
            "z": np.random.randn(100, 100)
        }
    
    async def _save_result(
        self,
        result: Any,
        original_data_id: str,
        function_name: str
    ) -> str:
        """Save processing result"""
        result_id = f"{original_data_id}_{function_name}_{uuid.uuid4().hex[:8]}"
        
        # TODO: Implement result storage
        logger.info(f"💾 Saved result: {result_id}")
        
        return result_id
    
    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create summary of result"""
        # TODO: Implement smart summarization
        return {
            "type": str(type(result).__name__),
            "size": "unknown"
        }
