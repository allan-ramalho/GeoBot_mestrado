"""
AI Provider Manager
Manages multiple AI providers (Groq, OpenAI, Claude, Gemini)
Handles provider switching, API key management, and model listing
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import httpx
import logging
from pathlib import Path
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """Supported AI providers"""
    GROQ = "groq"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


class ProviderManager:
    """
    Manages AI provider configuration and operations
    """
    
    def __init__(self):
        self.config_file = settings.DATA_DIR / "config" / "ai_provider.json"
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self._current_config: Optional[Dict] = None
    
    async def validate_api_key(self, provider: str, api_key: str) -> bool:
        """
        Validate API key for a specific provider
        """
        try:
            if provider == AIProvider.GROQ:
                return await self._validate_groq(api_key)
            elif provider == AIProvider.OPENAI:
                return await self._validate_openai(api_key)
            elif provider == AIProvider.CLAUDE:
                return await self._validate_claude(api_key)
            elif provider == AIProvider.GEMINI:
                return await self._validate_gemini(api_key)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"API key validation failed for {provider}: {e}")
            return False
    
    async def list_models(self, provider: str, api_key: str) -> List[Dict[str, Any]]:
        """
        List available models for a provider
        """
        try:
            if provider == AIProvider.GROQ:
                return await self._list_groq_models(api_key)
            elif provider == AIProvider.OPENAI:
                return await self._list_openai_models(api_key)
            elif provider == AIProvider.CLAUDE:
                return await self._list_claude_models(api_key)
            elif provider == AIProvider.GEMINI:
                return await self._list_gemini_models(api_key)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to list models for {provider}: {e}")
            raise
    
    async def save_configuration(self, config: Dict[str, Any]):
        """
        Save provider configuration securely
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self._current_config = config
            logger.info(f"✅ Configuration saved for provider: {config['provider']}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    async def get_configuration(self) -> Optional[Dict[str, Any]]:
        """
        Load current configuration
        """
        try:
            if self._current_config:
                return self._current_config
            
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self._current_config = json.load(f)
                return self._current_config
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    async def clear_configuration(self):
        """
        Clear provider configuration
        """
        if self.config_file.exists():
            self.config_file.unlink()
        self._current_config = None
    
    # Provider-specific validation methods
    
    async def _validate_groq(self, api_key: str) -> bool:
        """Validate Groq API key"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
            return response.status_code == 200
    
    async def _validate_openai(self, api_key: str) -> bool:
        """Validate OpenAI API key"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
            return response.status_code == 200
    
    async def _validate_claude(self, api_key: str) -> bool:
        """Validate Anthropic Claude API key"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=10.0
            )
            return response.status_code in [200, 400]  # 400 is ok, means auth worked
    
    async def _validate_gemini(self, api_key: str) -> bool:
        """Validate Google Gemini API key"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://generativelanguage.googleapis.com/v1/models?key={api_key}",
                timeout=10.0
            )
            return response.status_code == 200
    
    # Model listing methods
    
    async def _list_groq_models(self, api_key: str) -> List[Dict[str, Any]]:
        """List Groq models"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            data = response.json()
            
            models = []
            for model in data.get("data", []):
                models.append({
                    "id": model["id"],
                    "name": model["id"],
                    "context_window": model.get("context_window", 8192),
                    "supports_functions": True
                })
            
            return models
    
    async def _list_openai_models(self, api_key: str) -> List[Dict[str, Any]]:
        """List OpenAI models"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            data = response.json()
            
            # Filter for chat models
            chat_models = [m for m in data.get("data", []) if "gpt" in m["id"]]
            
            models = []
            for model in chat_models:
                models.append({
                    "id": model["id"],
                    "name": model["id"],
                    "context_window": 128000 if "gpt-4" in model["id"] else 16385,
                    "supports_functions": True
                })
            
            return models
    
    async def _list_claude_models(self, api_key: str) -> List[Dict[str, Any]]:
        """List Claude models"""
        # Anthropic doesn't have a models endpoint, return known models
        return [
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "context_window": 200000,
                "supports_functions": True
            },
            {
                "id": "claude-3-sonnet-20240229",
                "name": "Claude 3 Sonnet",
                "context_window": 200000,
                "supports_functions": True
            },
            {
                "id": "claude-3-haiku-20240307",
                "name": "Claude 3 Haiku",
                "context_window": 200000,
                "supports_functions": True
            }
        ]
    
    async def _list_gemini_models(self, api_key: str) -> List[Dict[str, Any]]:
        """List Gemini models"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
            )
            data = response.json()
            
            models = []
            for model in data.get("models", []):
                if "gemini" in model["name"]:
                    models.append({
                        "id": model["name"].split("/")[-1],
                        "name": model.get("displayName", model["name"]),
                        "context_window": 32768,
                        "supports_functions": True
                    })
            
            return models
