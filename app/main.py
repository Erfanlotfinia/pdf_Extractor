import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import grpc
import uvicorn

# Ensure the project root is in the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.endpoints import router as api_router
from app.core.config import get_settings
from app.processing.pdf_processor import PDFProcessorService
from app.storage.storage_service import MinioStorageService
from app.vector_db.vector_service import VectorService
from app.grpc_server import VectorizeServiceServicer
from app.grpc.generated import vectorizer_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Application State and Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    - Initializes and closes service connections (Qdrant, MinIO).
    """
    logger.info("Application startup: Initializing services...")

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
    
    logger.info("Application shutdown: Closing service connections...")
    # Cleanly close service connections
    await storage_service.close()
    await vector_service.close()
    logger.info("Service connections closed.")

# --- FastAPI Application Setup ---

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

# --- Server Runner ---

async def serve():
    """
    Starts both the gRPC and FastAPI servers concurrently.
    The application lifespan manager is used to initialize services, which are
    then shared by both servers.
    """
    settings = get_settings()

    # The app's lifespan context will manage service initialization and cleanup.
    async with app.router.lifespan_context(app):
        # --- Configure and start gRPC server ---
        grpc_server = grpc.aio.server()
        vectorizer_pb2_grpc.add_VectorizeServiceServicer_to_server(
            VectorizeServiceServicer(
                storage_service=app.state.storage_service,
                pdf_processor_service=app.state.pdf_processor_service,
                vector_service=app.state.vector_service,
            ),
            grpc_server,
        )
        grpc_listen_addr = f"[::]:{settings.GRPC_PORT}"
        grpc_server.add_insecure_port(grpc_listen_addr)

        async def run_grpc():
            logger.info(f"[gRPC] Starting server on {grpc_listen_addr}")
            await grpc_server.start()
            await grpc_server.wait_for_termination()

        # --- Configure and start FastAPI server ---
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        uvicorn_server = uvicorn.Server(uvicorn_config)

        async def run_fastapi():
             logger.info("[REST] Starting FastAPI server on http://0.0.0.0:8000")
             await uvicorn_server.serve()

        # --- Run servers concurrently ---
        try:
            await asyncio.gather(run_grpc(), run_fastapi())
        finally:
            logger.info("Servers are shutting down...")
            await grpc_server.stop(grace=1) # Graceful shutdown for gRPC

if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting.")
