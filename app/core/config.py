from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

load_dotenv()

class Settings(BaseSettings):
    """
    Configuration class for the application.
    Reads environment variables from a .env file or the environment.
    """
    OPENAI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    QDRANT_COLLECTION_NAME: str = "pdf-vectors"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_HOST: str = "minio"
    MINIO_PORT: int = 9000
    MINIO_BUCKET: str = "docs"
    OPENAI_BASE_URL: str = "https://api.avalai.ir/v1"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "pdf_documents"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

def get_settings() -> Settings:
    return settings

settings = Settings()