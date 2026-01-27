"""
Batch Processing System for Geophysical Data
Process multiple datasets in parallel with progress tracking
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a single batch processing job"""
    job_id: str
    input_data: Dict[str, Any]
    function_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'job_id': self.job_id,
            'function_name': self.function_name,
            'parameters': self.parameters,
            'status': self.status,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration
        }


class BatchProcessor:
    """
    Batch processing system for geophysical data
    
    Features:
    - Parallel execution with configurable workers
    - Progress tracking and monitoring
    - Error handling per job
    - Result aggregation
    - Export to various formats
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.jobs: List[BatchJob] = []
        self.function_registry = {}
        logger.info(f"🔧 Batch processor initialized with {max_workers} workers")
    
    def register_function(self, name: str, func: Callable):
        """
        Register a processing function
        
        Args:
            name: Function name
            func: Callable processing function
        """
        self.function_registry[name] = func
        logger.info(f"✅ Function registered: {name}")
    
    def add_job(
        self,
        job_id: str,
        input_data: Dict[str, Any],
        function_name: str,
        parameters: Dict[str, Any] = None
    ) -> BatchJob:
        """
        Add a job to the batch
        
        Args:
            job_id: Unique job identifier
            input_data: Input data dictionary
            function_name: Name of processing function
            parameters: Function parameters
            
        Returns:
            Created BatchJob object
        """
        if function_name not in self.function_registry:
            raise ValueError(f"Function '{function_name}' not registered")
        
        job = BatchJob(
            job_id=job_id,
            input_data=input_data,
            function_name=function_name,
            parameters=parameters or {}
        )
        self.jobs.append(job)
        logger.info(f"📋 Job added: {job_id} ({function_name})")
        return job
    
    def add_jobs_from_list(
        self,
        data_list: List[Dict[str, Any]],
        function_name: str,
        parameters: Dict[str, Any] = None,
        job_prefix: str = "job"
    ) -> List[BatchJob]:
        """
        Add multiple jobs from a list of datasets
        
        Args:
            data_list: List of input data dictionaries
            function_name: Processing function name
            parameters: Common parameters for all jobs
            job_prefix: Prefix for auto-generated job IDs
            
        Returns:
            List of created BatchJob objects
        """
        jobs = []
        for i, data in enumerate(data_list):
            job_id = f"{job_prefix}_{i:04d}"
            job = self.add_job(job_id, data, function_name, parameters)
            jobs.append(job)
        
        logger.info(f"📋 Added {len(jobs)} jobs to batch")
        return jobs
    
    def _execute_job(self, job: BatchJob) -> BatchJob:
        """
        Execute a single job
        
        Args:
            job: BatchJob to execute
            
        Returns:
            Updated BatchJob with results
        """
        job.status = "running"
        job.start_time = datetime.now()
        
        try:
            # Get function
            func = self.function_registry[job.function_name]
            
            # Execute
            logger.info(f"🚀 Executing job: {job.job_id}")
            result = func(job.input_data, **job.parameters)
            
            # Store result
            job.result = result
            job.status = "completed"
            logger.info(f"✅ Job completed: {job.job_id}")
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"❌ Job failed: {job.job_id} - {e}")
        
        finally:
            job.end_time = datetime.now()
            job.duration = (job.end_time - job.start_time).total_seconds()
        
        return job
    
    def execute(
        self,
        progress_callback: Optional[Callable[[int, int, BatchJob], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute all jobs in parallel
        
        Args:
            progress_callback: Optional callback for progress updates
                               Signature: callback(completed, total, current_job)
        
        Returns:
            Dictionary with execution summary
        """
        if not self.jobs:
            logger.warning("⚠️ No jobs to execute")
            return {'total': 0, 'completed': 0, 'failed': 0}
        
        logger.info(f"🚀 Starting batch execution: {len(self.jobs)} jobs")
        start_time = time.time()
        
        completed_count = 0
        failed_count = 0
        
        # Execute jobs in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._execute_job, job): job
                for job in self.jobs
            }
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    updated_job = future.result()
                    
                    if updated_job.status == "completed":
                        completed_count += 1
                    else:
                        failed_count += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            completed_count + failed_count,
                            len(self.jobs),
                            updated_job
                        )
                
                except Exception as e:
                    logger.error(f"❌ Unexpected error in job: {e}")
                    failed_count += 1
        
        total_time = time.time() - start_time
        
        summary = {
            'total': len(self.jobs),
            'completed': completed_count,
            'failed': failed_count,
            'success_rate': completed_count / len(self.jobs) * 100,
            'total_time': total_time,
            'avg_time_per_job': total_time / len(self.jobs)
        }
        
        logger.info(f"✅ Batch execution completed: {completed_count}/{len(self.jobs)} successful")
        logger.info(f"⏱️ Total time: {total_time:.2f}s, Avg: {summary['avg_time_per_job']:.2f}s/job")
        
        return summary
    
    def get_results(self, include_failed: bool = False) -> List[Dict[str, Any]]:
        """
        Get all job results
        
        Args:
            include_failed: Include failed jobs in results
            
        Returns:
            List of job result dictionaries
        """
        results = []
        for job in self.jobs:
            if job.status == "completed" or (include_failed and job.status == "failed"):
                results.append({
                    'job_id': job.job_id,
                    'status': job.status,
                    'result': job.result,
                    'error': job.error,
                    'duration': job.duration
                })
        return results
    
    def get_failed_jobs(self) -> List[BatchJob]:
        """Get list of failed jobs"""
        return [job for job in self.jobs if job.status == "failed"]
    
    def retry_failed_jobs(self) -> Dict[str, Any]:
        """
        Retry all failed jobs
        
        Returns:
            Execution summary for retried jobs
        """
        failed_jobs = self.get_failed_jobs()
        
        if not failed_jobs:
            logger.info("ℹ️ No failed jobs to retry")
            return {'total': 0, 'completed': 0, 'failed': 0}
        
        logger.info(f"🔄 Retrying {len(failed_jobs)} failed jobs")
        
        # Reset failed jobs
        for job in failed_jobs:
            job.status = "pending"
            job.error = None
            job.result = None
        
        # Create temporary processor for retry
        temp_processor = BatchProcessor(max_workers=self.max_workers)
        temp_processor.function_registry = self.function_registry
        temp_processor.jobs = failed_jobs
        
        return temp_processor.execute()
    
    def export_summary(self, output_path: str):
        """
        Export batch execution summary to JSON
        
        Args:
            output_path: Path to output JSON file
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_jobs': len(self.jobs),
            'completed': sum(1 for j in self.jobs if j.status == "completed"),
            'failed': sum(1 for j in self.jobs if j.status == "failed"),
            'jobs': [job.to_dict() for job in self.jobs]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Summary exported to: {output_path}")
    
    def clear(self):
        """Clear all jobs from the processor"""
        self.jobs.clear()
        logger.info("🧹 Batch processor cleared")


class BatchProcessingPipeline:
    """
    Multi-stage batch processing pipeline
    
    Process datasets through multiple sequential processing steps
    with intermediate result caching.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize pipeline
        
        Args:
            max_workers: Maximum parallel workers
        """
        self.stages: List[Dict[str, Any]] = []
        self.max_workers = max_workers
        self.results_cache: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("🔧 Batch processing pipeline initialized")
    
    def add_stage(
        self,
        stage_name: str,
        function_name: str,
        parameters: Dict[str, Any] = None
    ):
        """
        Add a processing stage to the pipeline
        
        Args:
            stage_name: Name of the stage
            function_name: Processing function name
            parameters: Function parameters
        """
        self.stages.append({
            'name': stage_name,
            'function': function_name,
            'parameters': parameters or {}
        })
        logger.info(f"➕ Stage added: {stage_name} ({function_name})")
    
    def execute_pipeline(
        self,
        input_datasets: List[Dict[str, Any]],
        function_registry: Dict[str, Callable],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete pipeline
        
        Args:
            input_datasets: List of input datasets
            function_registry: Dictionary of available functions
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with pipeline execution summary
        """
        logger.info(f"🚀 Executing pipeline: {len(self.stages)} stages, {len(input_datasets)} datasets")
        
        current_data = input_datasets
        stage_summaries = []
        
        for stage_idx, stage in enumerate(self.stages):
            logger.info(f"▶️ Stage {stage_idx + 1}/{len(self.stages)}: {stage['name']}")
            
            # Create batch processor for this stage
            processor = BatchProcessor(max_workers=self.max_workers)
            
            # Register function
            func = function_registry.get(stage['function'])
            if not func:
                raise ValueError(f"Function '{stage['function']}' not found in registry")
            
            processor.register_function(stage['function'], func)
            
            # Add jobs
            processor.add_jobs_from_list(
                current_data,
                stage['function'],
                stage['parameters'],
                job_prefix=f"stage_{stage_idx}"
            )
            
            # Execute stage
            summary = processor.execute(progress_callback)
            stage_summaries.append({
                'stage': stage['name'],
                'summary': summary
            })
            
            # Get results for next stage
            results = processor.get_results()
            current_data = [r['result'] for r in results if r['result']]
            
            # Cache results
            self.results_cache[stage['name']] = current_data
            
            logger.info(f"✅ Stage '{stage['name']}' completed: {len(current_data)} outputs")
        
        pipeline_summary = {
            'stages': stage_summaries,
            'input_count': len(input_datasets),
            'output_count': len(current_data),
            'final_results': current_data
        }
        
        logger.info(f"✅ Pipeline completed: {len(current_data)} final results")
        return pipeline_summary
    
    def get_stage_results(self, stage_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results from a specific stage
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            List of results or None if stage not found
        """
        return self.results_cache.get(stage_name)
    
    def clear_cache(self):
        """Clear all cached results"""
        self.results_cache.clear()
        logger.info("🧹 Pipeline cache cleared")


# Initialize logger
logger.info("📦 Batch processing system initialized")
