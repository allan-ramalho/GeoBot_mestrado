"""
Tests for Chat Service
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.ai.chat_service import ChatService


@pytest.fixture
def chat_service():
    """Create chat service instance"""
    return ChatService()


@pytest.fixture
def sample_message():
    """Sample user message"""
    return "What is magnetic reduction to pole?"


@pytest.fixture
def sample_context():
    """Sample processing context"""
    return {
        "project_id": "test_project",
        "current_data": "data_001"
    }


class TestChatServiceInit:
    """Test chat service initialization"""
    
    def test_service_initialization(self, chat_service):
        """Test service initializes correctly"""
        assert chat_service is not None
        assert chat_service.provider_manager is not None
        assert chat_service.rag_engine is not None
        assert chat_service.function_registry is not None
        assert chat_service.processing_engine is not None
        assert isinstance(chat_service.conversations, dict)
        assert len(chat_service.conversations) == 0


class TestLanguageDetection:
    """Test language detection"""
    
    def test_detect_portuguese(self, chat_service):
        """Test detecting Portuguese"""
        lang = chat_service._detect_language("Olá, como vai?")
        assert lang == "pt"
    
    def test_detect_english(self, chat_service):
        """Test detecting English"""
        lang = chat_service._detect_language("Hello, how are you?")
        assert lang == "en"
    
    def test_detect_spanish(self, chat_service):
        """Test detecting Spanish"""
        lang = chat_service._detect_language("Hola, ¿cómo estás?")
        assert lang == "es"
    
    def test_detect_fallback(self, chat_service):
        """Test language detection fallback"""
        # Very short text might fail detection
        lang = chat_service._detect_language("xyz")
        assert lang in ["en", "pt", "es"]  # Should fallback to one


class TestSystemPromptBuilding:
    """Test system prompt construction"""
    
    def test_build_prompt_portuguese(self, chat_service):
        """Test building prompt in Portuguese"""
        prompt = chat_service._build_system_prompt("pt", "", None)
        
        assert "GeoBot" in prompt
        assert "português" in prompt.lower()
        assert "geofísica" in prompt.lower()
    
    def test_build_prompt_english(self, chat_service):
        """Test building prompt in English"""
        prompt = chat_service._build_system_prompt("en", "", None)
        
        assert "GeoBot" in prompt
        assert "English" in prompt
        assert "geophysics" in prompt.lower()
    
    def test_build_prompt_with_rag_context(self, chat_service):
        """Test prompt includes RAG context"""
        rag_context = "Scientific paper about magnetic data"
        prompt = chat_service._build_system_prompt("en", rag_context, None)
        
        assert rag_context in prompt
    
    def test_build_prompt_with_context(self, chat_service, sample_context):
        """Test prompt includes processing context"""
        prompt = chat_service._build_system_prompt("en", "", sample_context)
        
        assert "test_project" in prompt or "Context" in prompt


class TestConversationManagement:
    """Test conversation tracking"""
    
    @pytest.mark.asyncio
    async def test_create_new_conversation(self, chat_service, sample_message):
        """Test creating new conversation"""
        with patch.object(chat_service, '_detect_language', return_value='en'):
            with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, return_value=[]):
                    mock_ai.return_value = {"content": "Test response"}
                    
                    response = await chat_service.process_message(sample_message)
                    
                    assert "conversation_id" in response
                    assert response["conversation_id"] in chat_service.conversations
    
    @pytest.mark.asyncio
    async def test_continue_existing_conversation(self, chat_service):
        """Test continuing existing conversation"""
        conv_id = "test_conv_123"
        
        with patch.object(chat_service, '_detect_language', return_value='en'):
            with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, return_value=[]):
                    mock_ai.return_value = {"content": "Response"}
                    
                    # First message
                    await chat_service.process_message("First message", conversation_id=conv_id)
                    
                    # Second message in same conversation
                    await chat_service.process_message("Second message", conversation_id=conv_id)
                    
                    history = chat_service.conversations[conv_id]
                    assert len(history) == 4  # 2 user + 2 assistant
    
    def test_conversation_history_format(self, chat_service):
        """Test conversation history has correct format"""
        conv_id = "test_conv"
        chat_service.conversations[conv_id] = []
        
        chat_service.conversations[conv_id].append({
            "role": "user",
            "content": "Test message",
            "timestamp": datetime.now().isoformat()
        })
        
        message = chat_service.conversations[conv_id][0]
        assert message["role"] == "user"
        assert message["content"] == "Test message"
        assert "timestamp" in message


class TestRAGIntegration:
    """Test RAG integration"""
    
    @pytest.mark.asyncio
    async def test_rag_enabled(self, chat_service, sample_message):
        """Test RAG is used when enabled"""
        with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock) as mock_rag:
            with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                with patch.object(chat_service, '_detect_language', return_value='en'):
                    mock_rag.return_value = [
                        {"content": "Scientific content", "citation": "Paper 2020"}
                    ]
                    mock_ai.return_value = {"content": "Response"}
                    
                    response = await chat_service.process_message(
                        sample_message,
                        use_rag=True
                    )
                    
                    mock_rag.assert_called_once()
                    assert "sources" in response
                    assert len(response["sources"]) > 0
    
    @pytest.mark.asyncio
    async def test_rag_disabled(self, chat_service, sample_message):
        """Test RAG is not used when disabled"""
        with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock) as mock_rag:
            with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                with patch.object(chat_service, '_detect_language', return_value='en'):
                    mock_ai.return_value = {"content": "Response"}
                    
                    response = await chat_service.process_message(
                        sample_message,
                        use_rag=False
                    )
                    
                    mock_rag.assert_not_called()
                    assert response["sources"] == []
    
    @pytest.mark.asyncio
    async def test_rag_sources_in_response(self, chat_service, sample_message):
        """Test RAG sources are included in response"""
        with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock) as mock_rag:
            with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                with patch.object(chat_service, '_detect_language', return_value='en'):
                    mock_rag.return_value = [
                        {"content": "Paper 1 content", "citation": "Author 2020"},
                        {"content": "Paper 2 content", "citation": "Author 2021"}
                    ]
                    mock_ai.return_value = {"content": "Response"}
                    
                    response = await chat_service.process_message(sample_message)
                    
                    assert len(response["sources"]) == 2
                    assert response["sources"][0]["citation"] == "Author 2020"


class TestFunctionCalling:
    """Test function calling integration"""
    
    @pytest.mark.asyncio
    async def test_interpret_function_call(self, chat_service):
        """Test interpreting messages as function calls"""
        message = "Apply reduction to pole with inclination -30"
        
        with patch.object(chat_service, '_interpret_as_function_call', new_callable=AsyncMock) as mock_interpret:
            with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                with patch.object(chat_service, '_detect_language', return_value='en'):
                    with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, return_value=[]):
                        mock_interpret.return_value = []
                        mock_ai.return_value = {"content": "Response"}
                        
                        await chat_service.process_message(message)
                        
                        mock_interpret.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_function_call(self, chat_service):
        """Test executing function when AI requests it"""
        with patch.object(chat_service, '_detect_language', return_value='en'):
            with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, return_value=[]):
                with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                    with patch.object(chat_service, '_execute_function', new_callable=AsyncMock) as mock_exec:
                        # First response requests function call
                        mock_ai.side_effect = [
                            {"content": "Calling function", "function_call": {"name": "test_func"}},
                            {"content": "Function executed"}
                        ]
                        mock_exec.return_value = {"status": "success"}
                        
                        response = await chat_service.process_message("Test message")
                        
                        mock_exec.assert_called_once()
                        assert len(response["function_calls"]) > 0


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_language_detection_error(self, chat_service):
        """Test handling language detection errors"""
        with patch.object(chat_service, '_detect_language', side_effect=Exception("Detection error")):
            with pytest.raises(Exception):
                await chat_service.process_message("Test message")
    
    @pytest.mark.asyncio
    async def test_rag_error(self, chat_service, sample_message):
        """Test handling RAG errors gracefully"""
        with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, side_effect=Exception("RAG error")):
            with pytest.raises(Exception):
                await chat_service.process_message(sample_message)
    
    @pytest.mark.asyncio
    async def test_ai_response_error(self, chat_service, sample_message):
        """Test handling AI response errors"""
        with patch.object(chat_service, '_detect_language', return_value='en'):
            with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, return_value=[]):
                with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock, side_effect=Exception("AI error")):
                    with pytest.raises(Exception):
                        await chat_service.process_message(sample_message)


class TestStreamingResponse:
    """Test streaming responses"""
    
    @pytest.mark.asyncio
    async def test_stream_message(self, chat_service, sample_message):
        """Test streaming message response"""
        with patch.object(chat_service, 'process_message', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"message": "Streamed response"}
            
            result = []
            async for chunk in chat_service.stream_message(sample_message):
                result.append(chunk)
            
            assert len(result) > 0
            assert result[0]["message"] == "Streamed response"


class TestResponseFormat:
    """Test response format"""
    
    @pytest.mark.asyncio
    async def test_response_has_required_fields(self, chat_service, sample_message):
        """Test response contains all required fields"""
        with patch.object(chat_service, '_detect_language', return_value='en'):
            with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, return_value=[]):
                with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                    mock_ai.return_value = {"content": "Test response"}
                    
                    response = await chat_service.process_message(sample_message)
                    
                    assert "message" in response
                    assert "conversation_id" in response
                    assert "sources" in response
                    assert "function_calls" in response
                    assert "language" in response
    
    @pytest.mark.asyncio
    async def test_message_content(self, chat_service, sample_message):
        """Test response message content"""
        expected_content = "This is the AI response"
        
        with patch.object(chat_service, '_detect_language', return_value='en'):
            with patch.object(chat_service.rag_engine, 'search', new_callable=AsyncMock, return_value=[]):
                with patch.object(chat_service, '_get_ai_response', new_callable=AsyncMock) as mock_ai:
                    mock_ai.return_value = {"content": expected_content}
                    
                    response = await chat_service.process_message(sample_message)
                    
                    assert response["message"] == expected_content
