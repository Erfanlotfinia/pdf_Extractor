from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration class for the application.
    Reads environment variables from a .env file or the environment.
    """
    OPENAI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    QDRANT_COLLECTION_NAME: str = "pdf-vectors"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
