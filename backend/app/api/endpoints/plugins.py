"""
Plugin API Endpoints
FastAPI routes for plugin system
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
from typing import List, Dict, Any
import logging

from app.core.plugin_system import plugin_manager, PluginError

router = APIRouter(prefix="/api/plugins", tags=["Plugins"])
logger = logging.getLogger(__name__)


@router.get("/list")
async def list_plugins() -> Dict[str, Any]:
    """List all loaded plugins"""
    try:
        plugins = plugin_manager.list_plugins()
        
        return {
            "success": True,
            "plugins": plugins,
            "count": len(plugins),
        }
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{plugin_id}")
async def get_plugin_info(plugin_id: str) -> Dict[str, Any]:
    """Get detailed plugin information"""
    try:
        info = plugin_manager.get_plugin_info(plugin_id)
        
        if not info:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        return {
            "success": True,
            "plugin": info,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_plugin(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and load a new plugin
    
    Args:
        file: Plugin file (.py)
    """
    if not file.filename.endswith('.py'):
        raise HTTPException(status_code=400, detail="Only .py files are allowed")
    
    try:
        # Save uploaded file
        plugin_path = plugin_manager.plugin_dir / file.filename
        
        with open(plugin_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Load plugin
        plugin_manager.load_plugin(plugin_path)
        
        return {
            "success": True,
            "message": f"Plugin {file.filename} uploaded and loaded",
            "plugin_id": plugin_path.stem,
        }
        
    except PluginError as e:
        # Remove invalid plugin file
        if plugin_path.exists():
            plugin_path.unlink()
        
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to upload plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute/{plugin_id}")
async def execute_plugin(
    plugin_id: str,
    data: Dict[str, Any],
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute a plugin function
    
    Args:
        plugin_id: Plugin identifier
        data: Input grid data
        params: Function parameters
    """
    try:
        result = plugin_manager.execute_plugin(
            plugin_id=plugin_id,
            data=data,
            params=params or {}
        )
        
        return result
        
    except PluginError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to execute plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{plugin_id}")
async def unload_plugin(plugin_id: str) -> Dict[str, Any]:
    """Unload a plugin"""
    try:
        success = plugin_manager.unload_plugin(plugin_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        return {
            "success": True,
            "message": f"Plugin {plugin_id} unloaded",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_all_plugins() -> Dict[str, Any]:
    """Reload all plugins from directory"""
    try:
        loaded, failed = plugin_manager.load_all_plugins()
        
        return {
            "success": True,
            "loaded": loaded,
            "failed": failed,
            "message": f"Reloaded {loaded} plugins ({failed} failed)",
        }
        
    except Exception as e:
        logger.error(f"Failed to reload plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template/download")
async def download_template() -> Dict[str, Any]:
    """Get plugin template as string"""
    try:
        template_path = Path(__file__).parent.parent / "core" / "plugin_template.py"
        
        if not template_path.exists():
            # Create template
            plugin_manager.create_plugin_template(template_path)
        
        template_content = template_path.read_text()
        
        return {
            "success": True,
            "template": template_content,
        }
        
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
