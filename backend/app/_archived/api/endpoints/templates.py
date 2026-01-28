"""
Templates API Endpoints
FastAPI routes for project templates
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict, Any
import logging

from app.core.templates import template_manager

router = APIRouter(prefix="/api/templates", tags=["Templates"])
logger = logging.getLogger(__name__)


@router.get("/list")
async def list_templates() -> Dict[str, Any]:
    """List all available templates"""
    try:
        templates = template_manager.list_templates()
        
        return {
            "success": True,
            "templates": templates,
            "count": len(templates),
        }
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_id}")
async def get_template_info(template_id: str) -> Dict[str, Any]:
    """Get template information"""
    try:
        template = template_manager.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "success": True,
            "template": template.to_dict(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create/{template_id}")
async def create_from_template(
    template_id: str,
    project_name: str,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Create project from template
    
    Args:
        template_id: Template identifier
        project_name: Name for new project
        output_path: Optional custom output path
    """
    try:
        # Determine output directory
        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = Path.home() / "GeoBot" / "Projects" / project_name
        
        # Create project
        project_info = template_manager.create_from_template(template_id, output_dir)
        
        return {
            "success": True,
            "message": f"Project created: {project_name}",
            "project": project_info,
            "path": str(output_dir),
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to create project from template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
