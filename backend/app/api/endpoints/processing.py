"""
Processing Endpoints
Geophysical data processing and function execution
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from app.services.geophysics.processing_engine import ProcessingEngine
from app.services.geophysics.function_registry import FunctionRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


class ProcessingRequest(BaseModel):
    """Processing request"""
    function_name: str
    data_id: str
    parameters: Optional[Dict[str, Any]] = None
    async_processing: bool = True


class WorkflowRequest(BaseModel):
    """Workflow request - chain multiple processing steps"""
    data_id: str
    steps: List[Dict[str, Any]]
    async_processing: bool = True


@router.get("/functions")
async def list_functions():
    """
    List all available processing functions
    """
    try:
        registry = FunctionRegistry()
        functions = registry.list_functions()
        
        return {"functions": functions}
        
    except Exception as e:
        logger.error(f"❌ Failed to list functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/functions/{function_name}")
async def get_function_info(function_name: str):
    """
    Get detailed information about a specific function
    """
    try:
        registry = FunctionRegistry()
        info = registry.get_function_info(function_name)
        
        if not info:
            raise HTTPException(status_code=404, detail="Function not found")
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get function info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute a processing function
    Can be synchronous or asynchronous
    """
    try:
        engine = ProcessingEngine()
        
        if request.async_processing:
            # Execute in background
            job_id = await engine.execute_async(
                function_name=request.function_name,
                data_id=request.data_id,
                parameters=request.parameters or {}
            )
            
            return {
                "status": "processing",
                "job_id": job_id,
                "message": "Processing started in background"
            }
        else:
            # Execute synchronously
            result = await engine.execute(
                function_name=request.function_name,
                data_id=request.data_id,
                parameters=request.parameters or {}
            )
            
            return {
                "status": "completed",
                "result": result
            }
        
    except Exception as e:
        logger.error(f"❌ Processing execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow")
async def execute_workflow(request: WorkflowRequest):
    """
    Execute a workflow (chain of processing steps)
    """
    try:
        engine = ProcessingEngine()
        
        job_id = await engine.execute_workflow(
            data_id=request.data_id,
            steps=request.steps,
            async_processing=request.async_processing
        )
        
        return {
            "status": "processing",
            "job_id": job_id,
            "message": f"Workflow with {len(request.steps)} steps started"
        }
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get processing job status
    """
    try:
        engine = ProcessingEngine()
        status = await engine.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running processing job
    """
    try:
        engine = ProcessingEngine()
        await engine.cancel_job(job_id)
        
        return {"status": "success", "message": f"Job {job_id} cancelled"}
        
    except Exception as e:
        logger.error(f"❌ Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret")
async def interpret_command(command: str, context: Optional[Dict[str, Any]] = None):
    """
    Interpret natural language command and map to processing functions
    Uses AI to understand user intent
    """
    try:
        engine = ProcessingEngine()
        
        interpretation = await engine.interpret_command(command, context)
        
        return {
            "command": command,
            "interpretation": interpretation
        }
        
    except Exception as e:
        logger.error(f"❌ Command interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
