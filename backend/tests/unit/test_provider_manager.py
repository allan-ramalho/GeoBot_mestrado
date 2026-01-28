"""
Tests for AI Provider Manager
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import httpx

from app.services.ai.provider_manager import ProviderManager, AIProvider


@pytest.fixture
def provider_manager():
    """Create a provider manager instance"""
    return ProviderManager()


class TestProviderValidation:
    """Test API key validation"""
    
    @pytest.mark.asyncio
    async def test_validate_groq_success(self, provider_manager):
        """Test GROQ API key validation success"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await provider_manager.validate_api_key("groq", "test_key")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_groq_failure(self, provider_manager):
        """Test GROQ API key validation failure"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await provider_manager.validate_api_key("groq", "invalid_key")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_openai_success(self, provider_manager):
        """Test OpenAI API key validation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await provider_manager.validate_api_key("openai", "test_key")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_unsupported_provider(self, provider_manager):
        """Test validation with unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported provider"):
            await provider_manager.validate_api_key("unsupported", "key")
    
    @pytest.mark.asyncio
    async def test_validate_with_network_error(self, provider_manager):
        """Test validation handles network errors"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Network error")
            )
            
            result = await provider_manager.validate_api_key("groq", "test_key")
            assert result is False


class TestModelListing:
    """Test model listing functionality"""
    
    @pytest.mark.asyncio
    async def test_list_groq_models_success(self, provider_manager):
        """Test listing GROQ models"""
        mock_models = {
            "data": [
                {"id": "llama-3.3-70b-versatile", "owned_by": "groq"},
                {"id": "mixtral-8x7b-32768", "owned_by": "groq"}
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json = Mock(return_value=mock_models)
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            models = await provider_manager.list_models("groq", "test_key")
            
            assert len(models) == 2
            assert models[0]["id"] == "llama-3.3-70b-versatile"
    
    @pytest.mark.asyncio
    async def test_list_models_failure(self, provider_manager):
        """Test model listing handles failures"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            with pytest.raises(Exception):
                await provider_manager.list_models("groq", "invalid_key")


class TestConfiguration:
    """Test configuration management"""
    
    @pytest.mark.asyncio
    async def test_save_configuration(self, provider_manager, tmp_path):
        """Test saving provider configuration"""
        # Override config file path for testing
        provider_manager.config_file = tmp_path / "test_config.json"
        
        config = {
            "provider": "groq",
            "api_key": "test_key",
            "model": "llama-3.3-70b-versatile"
        }
        
        await provider_manager.save_configuration(config)
        
        # Verify file was created
        assert provider_manager.config_file.exists()
        
        # Verify configuration is cached
        assert provider_manager._current_config == config
    
    @pytest.mark.asyncio
    async def test_get_configuration_cached(self, provider_manager):
        """Test getting cached configuration"""
        provider_manager._current_config = {"provider": "groq"}
        
        config = await provider_manager.get_configuration()
        
        assert config == {"provider": "groq"}
    
    @pytest.mark.asyncio
    async def test_get_configuration_from_file(self, provider_manager, tmp_path):
        """Test loading configuration from file"""
        import json
        
        config_file = tmp_path / "config.json"
        config_data = {"provider": "groq", "model": "test"}
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        provider_manager.config_file = config_file
        provider_manager._current_config = None
        
        config = await provider_manager.get_configuration()
        
        assert config["provider"] == "groq"
    
    @pytest.mark.asyncio
    async def test_get_configuration_no_file(self, provider_manager, tmp_path):
        """Test getting configuration when no file exists"""
        provider_manager.config_file = tmp_path / "nonexistent.json"
        provider_manager._current_config = None
        
        config = await provider_manager.get_configuration()
        
        assert config is None


class TestProviderMethods:
    """Test individual provider methods"""
    
    @pytest.mark.asyncio
    async def test_validate_claude(self, provider_manager):
        """Test Claude API validation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await provider_manager._validate_claude("test_key")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_claude_auth_error(self, provider_manager):
        """Test Claude validation with auth error (400 still means auth worked)"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await provider_manager._validate_claude("test_key")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_gemini(self, provider_manager):
        """Test Gemini API validation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await provider_manager._validate_gemini("test_key")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_list_openai_models(self, provider_manager):
        """Test listing OpenAI models"""
        mock_data = {
            "data": [
                {"id": "gpt-4", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "owned_by": "openai"}
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json = Mock(return_value=mock_data)
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            models = await provider_manager._list_openai_models("test_key")
            
            assert len(models) == 2
            assert models[0]["id"] == "gpt-4"
