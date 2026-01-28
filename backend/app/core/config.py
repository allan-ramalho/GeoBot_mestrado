"""
Core configuration for GeoBot Backend
Environment variables and application settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    
    # Application
    APP_NAME: str = "GeoBot"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="127.0.0.1", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="ALLOWED_ORIGINS"
    )
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = Field(default=None, env="SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = Field(default=None, env="SUPABASE_KEY")
    SUPABASE_BUCKET: str = Field(default="pdfs", env="SUPABASE_BUCKET")
    
    # Database (Supabase Postgres with pgvector)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # AI Providers Configuration
    AI_PROVIDER: Optional[str] = Field(default="groq", env="AI_PROVIDER")
    AI_MODEL: Optional[str] = Field(default="llama-3.3-70b-versatile", env="AI_MODEL")
    
    # AI Provider API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    
    # RAG Configuration
    EMBEDDING_MODEL: str = Field(
        default="intfloat/e5-large-v2",
        env="EMBEDDING_MODEL"
    )
    EMBEDDING_DIMENSION: int = 1024
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    # Storage
    DATA_DIR: Path = Field(
        default=Path.home() / "GeoBot" / "data",
        env="DATA_DIR"
    )
    PROJECTS_DIR: Path = Field(
        default=Path.home() / "GeoBot" / "projects",
        env="PROJECTS_DIR"
    )
    CACHE_DIR: Path = Field(
        default=Path.home() / "GeoBot" / "cache",
        env="CACHE_DIR"
    )
    
    # Processing
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    ASYNC_PROCESSING: bool = True
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[Path] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
