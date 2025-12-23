"""
Application configuration.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


def _get_project_root() -> Path:
    """Get project root, supporting both local dev and Docker."""
    # In Docker, data is mounted at /app/data
    if Path("/app/data").exists():
        return Path("/app")
    # Local development: go up from config.py
    return Path(__file__).parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings."""
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "To-Peek"
    DEBUG: bool = True
    
    # CORS - allow frontend in dev and prod
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://frontend:3000",  # Docker network
    ]
    
    # ML Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # LLM (for topic naming)
    DSPY_MODEL: str = "openai/gpt-4o-mini"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def PROJECT_ROOT(self) -> Path:
        return _get_project_root()
    
    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"
    
    @property
    def MODELING_DIR(self) -> Path:
        return self.PROJECT_ROOT / "modeling"
    
    @property
    def DATASETS_DIR(self) -> Path:
        return self.MODELING_DIR / "datasets"
    
    @property
    def CACHE_DIR(self) -> Path:
        return self.MODELING_DIR / "cache"


settings = Settings()

