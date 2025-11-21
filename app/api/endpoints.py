import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request

from app.models.schemas import (
    VectorizeRequest,
    VectorizeResponse,
    ErrorResponse,
    UploadResponse,
)
from app.processing.pdf_processor import PDFProcessorService
from app.storage.storage_service import StorageService
from app.vector_db.vector_service import VectorService

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection ---
# These dependencies will be managed by the application's lifespan context.
# This ensures a single, shared instance of each service is used per request.
def get_pdf_processor(request: Request) -> PDFProcessorService:
    return request.app.state.pdf_processor_service

def get_vector_service(request: Request) -> VectorService:
    return request.app.state.vector_service

def get_storage_service(request: Request) -> StorageService:
    return request.app.state.storage_service
# --------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF to Storage",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type (only PDF allowed)"},
        503: {"model": ErrorResponse, "description": "Storage service is unavailable"},
    },
)
async def upload_pdf(
    file: UploadFile = File(..., description="The PDF file to upload."),
    storage_service: StorageService = Depends(get_storage_service),
):
    """
    Accepts a PDF file, streams it to the configured object storage (MinIO),
    and returns a unique `file_key` for later processing.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_key = f"{uuid4()}.pdf"

    # Get the file size for the ContentLength header, crucial for S3-compatible services
    file.file.seek(0, 2)
    file_size = file.file.tell()
    await file.seek(0)

    try:
        # The async upload method now takes an async iterator
        await storage_service.upload_file(file.file, file_key, file_size)
        return UploadResponse(file_key=file_key, message="File uploaded successfully.")
    except RuntimeError as e:
        logger.exception("Storage service error during upload for file_key '%s': %s", file_key, e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("An unexpected error occurred during upload for file_key '%s': %s", file_key, e)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

@router.post(
    "/vectorize",
    response_model=VectorizeResponse,
    summary="Vectorize a PDF from Storage or URL",
    responses={
        404: {"model": ErrorResponse, "description": "The specified file key or URL was not found"},
        422: {"model": ErrorResponse, "description": "PDF processing failed; file may be corrupt"},
        503: {"model": ErrorResponse, "description": "A downstream service is unavailable"},
    },
)
async def vectorize_pdf(
    request: VectorizeRequest,
    pdf_processor: PDFProcessorService = Depends(get_pdf_processor),
    vector_service: VectorService = Depends(get_vector_service),
    storage_service: StorageService = Depends(get_storage_service),
):
    """
    This endpoint processes a PDF identified by a `file_key` (from `/upload`)
    or a public `source_url`. It extracts content, generates embeddings, and
    stores them in the Qdrant vector database.
    """
    source = request.file_key or request.source_url
    if not source:
        raise HTTPException(status_code=400, detail="Either 'file_key' or 'source_url' must be provided.")

    try:
        # 1. Download the PDF content
        pdf_content = await storage_service.download_file(source)

        # 2. Process the PDF to extract structured content
        file_hash, contents = await pdf_processor.process_pdf(pdf_content)

        # 3. Check if this document has already been processed
        existing_ids = await vector_service.check_document_exists(file_hash)
        if existing_ids:
            return VectorizeResponse(
                message="Document has already been processed.",
                document_ids=existing_ids,
            )

        # 4. Vectorize the content and upsert into the database
        await vector_service.vectorize_and_upsert(contents)

        document_ids = [str(c.id) for c in contents]
        return VectorizeResponse(
            message="PDF processed and vectorized successfully.",
            document_ids=document_ids,
        )
    except RuntimeError as e:
        logger.warning("A runtime error occurred in the vectorize endpoint for source '%s': %s", source, e)
        # Check for common error messages to provide better status codes
        if "not found" in str(e).lower() or "404" in str(e):
             raise HTTPException(status_code=404, detail=f"File not found: {source}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        logger.warning("PDF processing failed for source '%s': %s", source, e)
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {e}")
    except Exception as e:
        logger.exception("An unexpected error occurred in the vectorize endpoint for source '%s': %s", source, e)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
