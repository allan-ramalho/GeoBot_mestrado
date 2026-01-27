"""
API Router aggregation
"""

from fastapi import APIRouter

from app.api.endpoints import (
    ai,
    config,
    projects,
    processing,
    data,
    chat
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(ai.router, prefix="/ai", tags=["AI"])
api_router.include_router(config.router, prefix="/config", tags=["Configuration"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(processing.router, prefix="/processing", tags=["Processing"])
api_router.include_router(data.router, prefix="/data", tags=["Data"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
