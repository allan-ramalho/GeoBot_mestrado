"""
Integration Tests for API Endpoints
Tests for FastAPI routes and WebSocket connections
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
import json


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self, client: TestClient):
        """Test health check returns 200"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestChatEndpoints:
    """Tests for chat API endpoints"""
    
    @pytest.mark.asyncio
    async def test_chat_history(self, async_client: AsyncClient):
        """Test retrieving chat history"""
        response = await async_client.get("/api/chat/history")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, async_client: AsyncClient):
        """Test creating new conversation"""
        payload = {
            "title": "Test Conversation",
            "provider": "openai"
        }
        
        response = await async_client.post("/api/chat/conversations", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["title"] == "Test Conversation"


class TestProcessingEndpoints:
    """Tests for processing API endpoints"""
    
    @pytest.mark.asyncio
    async def test_list_functions(self, async_client: AsyncClient):
        """Test listing available functions"""
        response = await async_client.get("/api/processing/functions")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check function structure
        func = data[0]
        assert "id" in func
        assert "name" in func
        assert "category" in func
    
    @pytest.mark.asyncio
    async def test_execute_function(self, async_client: AsyncClient, sample_grid_data):
        """Test executing a processing function"""
        payload = {
            "function_id": "reduction_to_pole",
            "data": sample_grid_data,
            "params": {
                "inclination": -30.0,
                "declination": 0.0
            }
        }
        
        response = await async_client.post("/api/processing/execute", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, async_client: AsyncClient, sample_grid_data):
        """Test batch processing"""
        payload = {
            "jobs": [
                {
                    "id": "job1",
                    "function_id": "reduction_to_pole",
                    "data": sample_grid_data,
                    "params": {"inclination": -30.0, "declination": 0.0}
                },
                {
                    "id": "job2",
                    "function_id": "upward_continuation",
                    "data": sample_grid_data,
                    "params": {"altitude": 500.0}
                }
            ]
        }
        
        response = await async_client.post("/api/processing/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2


class TestWorkflowEndpoints:
    """Tests for workflow API endpoints"""
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, async_client: AsyncClient):
        """Test listing workflows"""
        response = await async_client.get("/api/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_workflow_template(self, async_client: AsyncClient):
        """Test getting workflow template"""
        response = await async_client.get("/api/workflows/templates/magnetic_enhancement")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "steps" in data
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, async_client: AsyncClient, sample_grid_data):
        """Test executing workflow"""
        payload = {
            "workflow_id": "magnetic_enhancement",
            "data": sample_grid_data,
            "params": {
                "inclination": -30.0,
                "declination": 0.0,
                "altitude": 500.0
            }
        }
        
        response = await async_client.post("/api/workflows/execute", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data


class TestRAGEndpoints:
    """Tests for RAG system endpoints"""
    
    @pytest.mark.asyncio
    async def test_search_documents(self, async_client: AsyncClient):
        """Test document search"""
        payload = {
            "query": "reduction to pole",
            "limit": 5
        }
        
        response = await async_client.post("/api/rag/search", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    @pytest.mark.ai
    async def test_query_with_rag(self, async_client: AsyncClient, skip_if_no_api_key):
        """Test query with RAG"""
        skip_if_no_api_key("groq")
        
        payload = {
            "query": "What is magnetic reduction to pole?",
            "use_rag": True,
            "provider": "groq"
        }
        
        response = await async_client.post("/api/chat/query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "citations" in data


# ============================================================================
# WebSocket Tests
# ============================================================================

class TestChatWebSocket:
    """Tests for chat WebSocket"""
    
    def test_websocket_connection(self, client: TestClient):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws/chat") as websocket:
            # Send test message
            websocket.send_json({
                "type": "message",
                "content": "Hello",
                "provider": "groq"
            })
            
            # Receive response
            data = websocket.receive_json()
            
            assert "type" in data
    
    @pytest.mark.asyncio
    @pytest.mark.ai
    async def test_websocket_streaming(self, client: TestClient, skip_if_no_api_key):
        """Test WebSocket streaming response"""
        skip_if_no_api_key("groq")
        
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({
                "type": "message",
                "content": "What is RTP?",
                "provider": "groq",
                "stream": True
            })
            
            chunks = []
            while True:
                data = websocket.receive_json()
                chunks.append(data)
                
                if data.get("type") == "done":
                    break
            
            assert len(chunks) > 0
            assert any(chunk.get("type") == "token" for chunk in chunks)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_function_id(self, async_client: AsyncClient, sample_grid_data):
        """Test invalid function ID"""
        payload = {
            "function_id": "nonexistent_function",
            "data": sample_grid_data,
            "params": {}
        }
        
        response = await async_client.post("/api/processing/execute", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    @pytest.mark.asyncio
    async def test_missing_required_params(self, async_client: AsyncClient, sample_grid_data):
        """Test missing required parameters"""
        payload = {
            "function_id": "reduction_to_pole",
            "data": sample_grid_data,
            "params": {}  # Missing inclination and declination
        }
        
        response = await async_client.post("/api/processing/execute", json=payload)
        
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_invalid_data_format(self, async_client: AsyncClient):
        """Test invalid data format"""
        payload = {
            "function_id": "reduction_to_pole",
            "data": {"invalid": "format"},
            "params": {"inclination": -30.0, "declination": 0.0}
        }
        
        response = await async_client.post("/api/processing/execute", json=payload)
        
        assert response.status_code == 400
