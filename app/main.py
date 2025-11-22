import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.endpoints import router as api_router
from processing.pdf_processor import PDFProcessorService
from storage.storage_service import MinioStorageService
from vector_db.vector_service import VectorService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    - Initializes and closes service connections (Qdrant, MinIO).
    """
    logger.info("Application startup...")

    # Initialize services
    storage_service = MinioStorageService()
    await storage_service.initialize()
    
    vector_service = VectorService()
    await vector_service.initialize()

    pdf_processor_service = PDFProcessorService()

    # Store service instances in the app state for dependency injection
    app.state.storage_service = storage_service
    app.state.vector_service = vector_service
    app.state.pdf_processor_service = pdf_processor_service
    
    yield
    
    logger.info("Application shutdown...")
    # Cleanly close service connections
    await storage_service.close()
    await vector_service.close()


app = FastAPI(
    title="PDF Vectorization Microservice",
    description="An API to extract content from PDFs, create vector embeddings, and store them in Qdrant.",
    version="1.0.0",
    lifespan=lifespan,
)

# Include the main API router
app.include_router(api_router, prefix="/api/v1")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """A global exception handler to catch unhandled errors."""
    logger.exception("Unhandled exception for request %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "An internal server error occurred."},
    )

@app.get("/", tags=["Health Check"])
async def health_check():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
