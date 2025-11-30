from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

load_dotenv()

class Settings(BaseSettings):
    """
    Configuration class for the application.
    Reads environment variables from a .env file or the environment.
    """
    # OpenAI / Embedding Settings
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.avalai.ir/v1"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536 # Default for text-embedding-3-small
     
    # GitHub Settings
    GITHUB_BASE_URL: str = "https://models.inference.ai.azure.com"
    GITHUB_TOKEN: str

    # Qdrant Settings
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "pdf_documents"

    # MinIO Settings
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_HOST: str = "minio"
    MINIO_PORT: int = 9000
    MINIO_BUCKET: str = "docs"
    
    # PDF Processing Settings
    OCR_LANGUAGE: str = "fas" # Persian language for OCR

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False # Standard for env vars
    )

def get_settings() -> Settings:
    return settings

settings = Settings()
