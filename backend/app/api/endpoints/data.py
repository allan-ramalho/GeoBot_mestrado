"""
Data Management Endpoints
File upload, download, and data management
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import logging
from pathlib import Path

from app.services.storage.data_manager import DataManager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    project_id: str = None,
    data_type: str = "raw"  # raw, processed, interpretation
):
    """
    Upload geophysical data file
    Supports various formats: CSV, XYZ, GRD, etc.
    """
    try:
        data_manager = DataManager()
        
        # Save file
        file_path = await data_manager.save_file(
            file=file,
            project_id=project_id,
            data_type=data_type
        )
        
        # Parse file metadata
        metadata = await data_manager.parse_file_metadata(file_path)
        
        logger.info(f"✅ File uploaded: {file.filename}")
        
        return {
            "status": "success",
            "file_id": str(file_path.stem),
            "filename": file.filename,
            "path": str(file_path),
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{project_id}")
async def list_files(project_id: str):
    """
    List all files in a project
    """
    try:
        data_manager = DataManager()
        files = await data_manager.list_project_files(project_id)
        
        return {"project_id": project_id, "files": files}
        
    except Exception as e:
        logger.error(f"❌ Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """
    Download a file by ID
    """
    try:
        data_manager = DataManager()
        file_path = await data_manager.get_file_path(file_id)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"❌ File download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete a file
    """
    try:
        data_manager = DataManager()
        await data_manager.delete_file(file_id)
        
        logger.info(f"🗑️ File deleted: {file_id}")
        return {"status": "success", "message": f"File {file_id} deleted"}
        
    except Exception as e:
        logger.error(f"❌ File deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import")
async def import_data(
    file: UploadFile = File(...),
    format: str = "xyz",  # xyz, csv, grd
    project_id: str = None
):
    """
    Import and parse geophysical data
    Automatically detects format and structure
    """
    try:
        data_manager = DataManager()
        
        # Import and parse data
        data = await data_manager.import_data(
            file=file,
            format=format,
            project_id=project_id
        )
        
        logger.info(f"✅ Data imported: {file.filename}")
        
        return {
            "status": "success",
            "data_id": data["id"],
            "summary": data["summary"]
        }
        
    except Exception as e:
        logger.error(f"❌ Data import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
