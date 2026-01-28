"""
Chat Endpoints
Handles AI Assistant conversations with RAG and function calling
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json

from app.services.ai.chat_service import ChatService
from app.services.ai.rag_engine import RAGEngine

router = APIRouter()
logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Chat message"""
    role: str  # user, assistant, system
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat request"""
    message: str
    conversation_id: Optional[str] = None
    use_rag: bool = True
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response"""
    message: str
    conversation_id: str
    sources: Optional[List[Dict[str, Any]]] = None
    function_calls: Optional[List[Dict[str, Any]]] = None
    language: str = "en"


class ConversationCreate(BaseModel):
    """Create conversation request"""
    title: str
    provider: Optional[str] = "openai"


@router.post("/message")
async def send_message(request: ChatRequest):
    """
    Send a message to GeoBot Assistant
    Supports RAG, function calling, and multi-language
    
    Returns:
        ChatResponse with message, citations, and function calls
    """
    try:
        chat_service = ChatService()
        
        # Process message with RAG and function calling
        response = await chat_service.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
            use_rag=request.use_rag,
            context=request.context
        )
        
        return {
            "message_id": response.get("message_id"),
            "conversation_id": response.get("conversation_id"),
            "response": response.get("response"),
            "citations": response.get("sources", []),
            "function_calls": response.get("function_calls", []),
            "language": response.get("language", "pt"),
            "timestamp": response.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with streaming
    
    Client sends:
        {
            "message": "user message",
            "conversation_id": "optional-id",
            "use_rag": true,
            "context": {}
        }
    
    Server sends chunks:
        {
            "type": "start",
            "conversation_id": "id"
        }
        {
            "type": "content",
            "content": "partial response"
        }
        {
            "type": "citation",
            "citation": {...}
        }
        {
            "type": "end",
            "message_id": "id",
            "timestamp": "..."
        }
        {
            "type": "error",
            "error": "message"
        }
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection established")
    
    chat_service = ChatService()
    conversation_id = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            logger.info(f"📨 Received: {request_data.get('message', '')[:50]}...")
            
            # Extract request parameters
            message = request_data.get("message")
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "error": "Message is required"
                })
                continue
            
            conversation_id = request_data.get("conversation_id", conversation_id)
            use_rag = request_data.get("use_rag", True)
            context = request_data.get("context")
            
            try:
                # Send start event
                await websocket.send_json({
                    "type": "start",
                    "conversation_id": conversation_id
                })
                
                # Process message with streaming
                async for chunk in chat_service.stream_message(
                    message=message,
                    conversation_id=conversation_id,
                    use_rag=use_rag,
                    context=context
                ):
                    await websocket.send_json(chunk)
                    
                    # Update conversation ID if new
                    if chunk.get("type") == "end" and chunk.get("conversation_id"):
                        conversation_id = chunk["conversation_id"]
                
                logger.info(f"✅ Message processed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket connection closed normally")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation history
    """
    try:
        chat_service = ChatService()
        history = await chat_service.get_conversation_history(conversation_id)
        
        return {"conversation_id": conversation_id, "messages": history}
        
    except Exception as e:
        logger.error(f"❌ Failed to get conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_chat_history():
    """
    Get list of all conversations
    """
    try:
        # For now, return empty list (no persistence implemented)
        # TODO: Implement conversation persistence
        return []
        
    except Exception as e:
        logger.error(f"❌ Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations")
async def create_conversation(request: ConversationCreate):
    """
    Create a new conversation
    """
    try:
        import uuid
        conversation_id = str(uuid.uuid4())
        
        # For now, just return the created conversation
        # TODO: Implement conversation persistence
        return {
            "id": conversation_id,
            "title": request.title,
            "provider": request.provider,
            "created_at": None  # Will be added when persistence is implemented
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Get conversation history
    """
    try:
        chat_service = ChatService()
        history = await chat_service.get_conversation_history(conversation_id)
        
        return {"conversation_id": conversation_id, "messages": history}
        
    except Exception as e:
        logger.error(f"❌ Failed to get conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    """
    try:
        chat_service = ChatService()
        await chat_service.delete_conversation(conversation_id)
        
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
        
    except Exception as e:
        logger.error(f"❌ Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_knowledge_base(query: str, top_k: int = 5):
    """
    Search RAG knowledge base directly
    Returns relevant documents with citations
    """
    try:
        rag_engine = RAGEngine()
        results = await rag_engine.search(query, top_k=top_k)
        
        return {
            "query": query,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Knowledge base search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    """Query with RAG request"""
    query: str
    use_rag: bool = True
    provider: Optional[str] = "openai"


@router.post("/query")
async def query_with_rag(request: QueryRequest):
    """
    Query with RAG support.
    Returns AI response with citations.
    
    This is a mock implementation. In a real scenario, this would:
    1. Search relevant documents using RAG
    2. Generate response using LLM with context
    3. Return response with citations
    """
    try:
        # TODO: Implement actual RAG query
        # For now, return mock response with citations
        return {
            "message": "Mock response about magnetic reduction to pole",
            "citations": [
                {
                    "source": "geophysics_manual.pdf",
                    "page": 1,
                    "text": "Reduction to pole is a technique..."
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
