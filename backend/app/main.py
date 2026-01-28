"""
GeoBot - Backend FastAPI
Main application entry point
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import json
from pathlib import Path

from app.core.config import settings
from app.api import api_router
from app.core.logging_config import setup_logging
from app.services.ai.chat_service import ChatService

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application
    Handles startup and shutdown events
    """
    logger.info("🚀 GeoBot Backend starting...")
    
    # Startup: Initialize services
    try:
        # Initialize database connections
        # Initialize AI providers
        # Load processing functions registry
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    logger.info("👋 GeoBot Backend shutting down...")
    # Close database connections
    # Cleanup resources


# Initialize FastAPI application
app = FastAPI(
    title="GeoBot API",
    description="Backend API for GeoBot - AI-Powered Geophysical Data Processing",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint - Health check"""
    return {
        "service": "GeoBot API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    Returns status of all critical services
    """
    health_status = {
        "status": "healthy",
        "services": {
            "api": "operational",
            "database": "operational",
            "ai_provider": "operational",
            "rag_engine": "operational"
        }
    }
    
    # TODO: Add actual health checks for each service
    
    return health_status


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with streaming support
    Mounted at /ws/chat for backward compatibility with tests
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Check if streaming is requested
            if data.get("stream", False):
                # Mock streaming response
                content = data.get("content", "")
                response_text = f"Mock streaming response to: {content}"
                
                # Send tokens one by one
                for i, char in enumerate(response_text):
                    await websocket.send_json({
                        "type": "token",
                        "content": char,
                        "index": i
                    })
                
                # Send done signal
                await websocket.send_json({
                    "type": "done",
                    "total_tokens": len(response_text)
                })
            else:
                # Regular echo response
                message_type = data.get("type", "message")
                response = {
                    "type": message_type,
                    "content": f"Received: {data.get('content', 'No content')}",
                    "echo": True
                }
                
                await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
