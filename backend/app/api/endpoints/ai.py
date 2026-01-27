"""
AI Provider Configuration Endpoints
Manages AI provider setup, model listing, and configuration
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.services.ai.provider_manager import ProviderManager, AIProvider

router = APIRouter()
logger = logging.getLogger(__name__)


class ProviderConfig(BaseModel):
    """AI Provider configuration"""
    provider: str  # groq, openai, claude, gemini
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    name: str
    context_window: int
    supports_functions: bool


@router.get("/providers", response_model=List[str])
async def list_providers():
    """
    List all available AI providers
    """
    return ["groq", "openai", "claude", "gemini"]


@router.post("/providers/configure")
async def configure_provider(config: ProviderConfig):
    """
    Configure AI provider with API key and model
    Validates connection and saves configuration
    """
    try:
        provider_manager = ProviderManager()
        
        # Validate API key and get available models
        is_valid = await provider_manager.validate_api_key(
            config.provider,
            config.api_key
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid API key or provider not accessible"
            )
        
        # Save configuration
        await provider_manager.save_configuration(config.dict())
        
        logger.info(f"✅ AI Provider configured: {config.provider} - {config.model}")
        
        return {
            "status": "success",
            "message": f"Provider {config.provider} configured successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to configure provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers/{provider}/models", response_model=List[ModelInfo])
async def list_models(provider: str, api_key: str):
    """
    List available models for a specific provider
    Requires API key for authentication
    """
    try:
        provider_manager = ProviderManager()
        models = await provider_manager.list_models(provider, api_key)
        return models
        
    except Exception as e:
        logger.error(f"❌ Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/current")
async def get_current_config():
    """
    Get current AI provider configuration
    API key is masked for security
    """
    try:
        provider_manager = ProviderManager()
        config = await provider_manager.get_configuration()
        
        if config:
            # Mask API key
            config["api_key"] = "***" + config["api_key"][-4:] if config.get("api_key") else None
        
        return config or {"configured": False}
        
    except Exception as e:
        logger.error(f"❌ Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config")
async def clear_configuration():
    """
    Clear AI provider configuration
    """
    try:
        provider_manager = ProviderManager()
        await provider_manager.clear_configuration()
        
        return {"status": "success", "message": "Configuration cleared"}
        
    except Exception as e:
        logger.error(f"❌ Failed to clear configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))
