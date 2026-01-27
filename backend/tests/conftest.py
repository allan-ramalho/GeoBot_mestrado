"""
Pytest Configuration and Fixtures
Shared fixtures for all tests
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from httpx import AsyncClient
from fastapi.testclient import TestClient
import numpy as np

# Import app
from app.main import app
from app.core.config import settings


# ============================================================================
# Pytest Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# API Client Fixtures
# ============================================================================

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Synchronous test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Asynchronous test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_grid_data():
    """Sample grid data for geophysics tests"""
    nx, ny = 100, 100
    x = np.linspace(0, 10000, nx)
    y = np.linspace(0, 10000, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Synthetic magnetic anomaly
    z = 100 * np.exp(-((xx - 5000)**2 + (yy - 5000)**2) / (1000**2))
    
    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'z': z.tolist(),
        'nx': nx,
        'ny': ny,
    }


@pytest.fixture
def sample_geophysics_params():
    """Sample parameters for geophysics functions"""
    return {
        'magnetic': {
            'inclination': -30.0,
            'declination': 0.0,
            'altitude': 500.0,
        },
        'gravity': {
            'density': 2.67,
            'height': 100.0,
        },
        'filter': {
            'wavelength': 1000.0,
            'order': 4,
        }
    }


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for AI tests"""
    return [
        {
            "role": "user",
            "content": "What is reduction to pole in magnetics?"
        },
        {
            "role": "assistant",
            "content": "Reduction to Pole (RTP) is a magnetic data processing technique..."
        }
    ]


@pytest.fixture
def sample_pdf_metadata():
    """Sample PDF metadata for RAG tests"""
    return {
        "title": "Geophysical Methods in Mineral Exploration",
        "author": "Test Author",
        "year": 2024,
        "pages": 150,
        "abstract": "This paper discusses geophysical methods..."
    }


# ============================================================================
# Mock Services
# ============================================================================

@pytest.fixture
def mock_ai_response():
    """Mock AI service response"""
    return {
        "message": "This is a test response from the AI",
        "model": "gpt-4",
        "provider": "openai",
        "citations": []
    }


@pytest.fixture
def mock_processing_result():
    """Mock processing result"""
    return {
        "success": True,
        "result": {
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "metadata": {
                "function": "test_function",
                "params": {},
                "execution_time": 0.123
            }
        },
        "error": None
    }


# ============================================================================
# Database Fixtures (if needed)
# ============================================================================

@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client"""
    class MockSupabase:
        def table(self, name):
            return self
        
        def select(self, *args):
            return self
        
        def insert(self, data):
            return self
        
        def execute(self):
            return {"data": [], "error": None}
    
    return MockSupabase()


# ============================================================================
# Environment Configuration
# ============================================================================

@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings before each test"""
    # Save original values
    original_env = settings.ENVIRONMENT
    
    # Set test environment
    settings.ENVIRONMENT = "test"
    
    yield
    
    # Restore
    settings.ENVIRONMENT = original_env


# ============================================================================
# Markers and Utilities
# ============================================================================

@pytest.fixture
def skip_if_no_api_key():
    """Skip test if API keys are not configured"""
    def _skip(provider: str):
        key_map = {
            "openai": settings.OPENAI_API_KEY,
            "anthropic": settings.ANTHROPIC_API_KEY,
            "google": settings.GOOGLE_API_KEY,
        }
        if not key_map.get(provider):
            pytest.skip(f"{provider} API key not configured")
    return _skip
