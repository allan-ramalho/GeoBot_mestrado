"""
Configuration Endpoints
System-wide configuration management
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class SystemConfig(BaseModel):
    """System configuration"""
    theme: str = "dark"
    language: str = "en"
    auto_save: bool = True
    max_workers: int = 4
    cache_enabled: bool = True


@router.get("/system")
async def get_system_config():
    """Get current system configuration"""
    # TODO: Load from configuration file
    return {
        "theme": "dark",
        "language": "pt-br",
        "auto_save": True,
        "max_workers": 4,
        "cache_enabled": True
    }


@router.put("/system")
async def update_system_config(config: SystemConfig):
    """Update system configuration"""
    try:
        # TODO: Save to configuration file
        logger.info(f"✅ System configuration updated: {config.dict()}")
        return {"status": "success", "config": config}
    except Exception as e:
        logger.error(f"❌ Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supabase")
async def get_supabase_config():
    """Get Supabase configuration status"""
    from app.core.config import settings
    
    return {
        "configured": bool(settings.SUPABASE_URL and settings.SUPABASE_KEY),
        "url": settings.SUPABASE_URL[:20] + "..." if settings.SUPABASE_URL else None,
        "bucket": settings.SUPABASE_BUCKET
    }


@router.post("/supabase")
async def configure_supabase(url: str, key: str, bucket: str = "pdfs"):
    """Configure Supabase connection"""
    try:
        from app.core.config import settings
        
        # Update settings
        settings.SUPABASE_URL = url
        settings.SUPABASE_KEY = key
        settings.SUPABASE_BUCKET = bucket
        
        # TODO: Validate connection
        # TODO: Save to secure storage
        
        logger.info("✅ Supabase configured successfully")
        return {"status": "success", "message": "Supabase configured"}
        
    except Exception as e:
        logger.error(f"❌ Supabase configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
