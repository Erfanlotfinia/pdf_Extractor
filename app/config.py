# app/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App
    app_name: str = "PDF Vectorization Service"
    environment: str = Field("dev", env="ENVIRONMENT")

    # Qdrant
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_collection_default: str = Field("pdf_documents", env="QDRANT_COLLECTION")

    # OpenAI / embeddings
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(1536, env="EMBEDDING_DIM")  # text-embedding-3-small

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()