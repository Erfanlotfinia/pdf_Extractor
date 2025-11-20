from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from api.endpoints import router as api_router
from fastapi.responses import JSONResponse
from services.vector_service import VectorService
from services.storage_service import MinioStorageService
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up FastAPI application...")
    try:
        # Initialize Vector Service
        vector_service = VectorService()
        await vector_service.initialize_collection()
        logger.info("Vector service initialized successfully.")
        app.state.vector_service = vector_service
    except Exception as e:
        logger.exception("Failed to initialize vector service during startup: %s", e)
        raise
    
    try:
        # Initialize Storage Service
        storage_service = MinioStorageService()
        if storage_service.client:
            logger.info("Storage service (MinIO) initialized successfully.")
        else:
            logger.info("Storage service (MinIO) not initialized - uploads/downloads will not be available.")
        app.state.storage_service = storage_service
    except Exception as e:
        logger.exception("Failed to initialize storage service during startup: %s", e)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    try:
        if hasattr(app.state, 'vector_service'):
            await app.state.vector_service.qdrant_client.close()
            logger.info("Vector service closed successfully.")
    except Exception as e:
        logger.exception("Error closing vector service: %s", e)

app = FastAPI(
    title="PDF Vectorization Microservice",
    description="An API to extract content from PDFs, create embeddings, and store them in Qdrant.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1", tags=["Vectorization"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception for request %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error"},
    )

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    A simple health check endpoint.
    """
    return {"status": "ok"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
