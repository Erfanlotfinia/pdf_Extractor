import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# --- Path Setup (If running as a script vs module) ---
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# -----------------------------------------------------

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.processing.pdf_processor import PDFProcessorService
from app.storage.storage_service import MinioStorageService
from app.vector_db.vector_service import VectorService


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the application.
    Initializes connections on startup and closes them on shutdown.
    """
    logger.info("--- Application Startup ---")

    try:
        # 1. Storage (MinIO)
        storage_service = MinioStorageService()
        await storage_service.initialize()
        app.state.storage_service = storage_service
        logger.info("Storage Service: OK")

        # 2. Vector DB (Qdrant + OpenAI)
        vector_service = VectorService()
        await vector_service.initialize()
        app.state.vector_service = vector_service
        logger.info("Vector Service: OK")

        # 3. PDF Processor (Unstructured)
        # (Usually stateless, but good to instantiate once)
        pdf_processor_service = PDFProcessorService()
        app.state.pdf_processor_service = pdf_processor_service
        logger.info("PDF Processor: OK")

        yield

    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise e
    
    finally:
        logger.info("--- Application Shutdown ---")
        # Graceful cleanup
        if hasattr(app.state, "vector_service"):
            await app.state.vector_service.close()
        if hasattr(app.state, "storage_service"):
            await app.state.storage_service.close()


app = FastAPI(
    title="RAG PDF Microservice",
    description="API for ingesting PDFs, extracting semantic chunks, and vector retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

# Optional: CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "path": str(request.url)},
    )

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy", "service": "pdf-vectorizer"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )