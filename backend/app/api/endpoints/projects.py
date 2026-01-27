"""
Project Management Endpoints
Create, manage, and organize geophysical projects
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from app.services.storage.project_manager import ProjectManager

router = APIRouter()
logger = logging.getLogger(__name__)


class ProjectCreate(BaseModel):
    """Create project request"""
    name: str
    description: Optional[str] = None
    project_type: str = "magnetic"  # magnetic, gravity, combined
    location: Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    """Update project request"""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/create")
async def create_project(project: ProjectCreate):
    """
    Create a new geophysical project
    """
    try:
        project_manager = ProjectManager()
        
        project_data = await project_manager.create_project(
            name=project.name,
            description=project.description,
            project_type=project.project_type,
            location=project.location
        )
        
        logger.info(f"✅ Project created: {project.name}")
        
        return project_data
        
    except Exception as e:
        logger.error(f"❌ Project creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_projects():
    """
    List all projects
    """
    try:
        project_manager = ProjectManager()
        projects = await project_manager.list_projects()
        
        return {"projects": projects}
        
    except Exception as e:
        logger.error(f"❌ Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}")
async def get_project(project_id: str):
    """
    Get project details
    """
    try:
        project_manager = ProjectManager()
        project = await project_manager.get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return project
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{project_id}")
async def update_project(project_id: str, update: ProjectUpdate):
    """
    Update project details
    """
    try:
        project_manager = ProjectManager()
        project = await project_manager.update_project(project_id, update.dict(exclude_unset=True))
        
        logger.info(f"✅ Project updated: {project_id}")
        
        return project
        
    except Exception as e:
        logger.error(f"❌ Project update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """
    Delete a project and all its data
    """
    try:
        project_manager = ProjectManager()
        await project_manager.delete_project(project_id)
        
        logger.info(f"🗑️ Project deleted: {project_id}")
        
        return {"status": "success", "message": f"Project {project_id} deleted"}
        
    except Exception as e:
        logger.error(f"❌ Project deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/tree")
async def get_project_tree(project_id: str):
    """
    Get project file tree structure
    """
    try:
        project_manager = ProjectManager()
        tree = await project_manager.get_project_tree(project_id)
        
        return tree
        
    except Exception as e:
        logger.error(f"❌ Failed to get project tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))
