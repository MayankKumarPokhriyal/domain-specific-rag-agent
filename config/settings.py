"""
Application configuration and defaults.

Uses environment variables for overrides; keeps sensible local defaults for dev.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_PATH = DATA_DIR / "faiss_index"


class AppSettings(BaseSettings):
    """Runtime settings for the RAG system."""

    # ---------- Storage ----------
    data_dir: Path = Field(default=DATA_DIR)
    vector_store_path: Path = Field(default=DEFAULT_INDEX_PATH)

    # ---------- Chunking ----------
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)

    # ---------- Embeddings ----------
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_batch_size: int = Field(default=16)

    # ---------- Ollama ----------
    # IMPORTANT: base URL only — NOT /api/generate
    ollama_api_url: str = Field(
        default="http://localhost:11434"
    )
    ollama_model: str = Field(default="llama3")
    ollama_temperature: float = Field(default=0.2)
    ollama_max_tokens: int = Field(default=512)

    # ---------- Retrieval ----------
    retriever_top_k: int = Field(default=4)
    retriever_score_threshold: float = Field(default=0.45)

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached settings instance."""
    return AppSettings()