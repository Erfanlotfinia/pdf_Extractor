import logging
import os  # <--- Added
import tempfile # <--- Added
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
    
    # Get the file size for the ContentLength header
    file.file.seek(0, 2)
    file_size = file.file.tell()
    await file.seek(0)
    
    try:
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
    Downloads the PDF from storage/URL to a temporary file, 
    processes it, and vectors the content.
    """
    source = request.file_key or request.source_url
    if not source:
        raise HTTPException(status_code=400, detail="Either 'file_key' or 'source_url' must be provided.")

    tmp_path = None # Initialize for cleanup in finally block

    try:
        # 1. Download the PDF content (Bytes)
        pdf_bytes = await storage_service.download_file(source)

        # 2. WRITE BYTES TO A TEMP FILE
        # The processor needs a file path string, not raw bytes.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name # e.g. /tmp/tmpxyz.pdf

        # 3. Process the PDF using the FILE PATH
        file_hash, contents = await pdf_processor.process_pdf(tmp_path)

        # 4. Check if this document has already been processed
        existing_ids = await vector_service.check_document_exists(file_hash)
        if existing_ids:
            return VectorizeResponse(
                message="Document has already been processed.",
                document_ids=existing_ids,
            )

        # 5. Handle case where no extractable content was found
        if not contents:
            logger.warning("PDF processed but no extractable content found for source '%s'", source)
            return VectorizeResponse(
                message="PDF processed successfully, but no extractable text content was found.",
                document_ids=[],
            )

        # 6. Vectorize the content and upsert into the database
        await vector_service.vectorize_and_upsert(contents)

        document_ids = [str(c.id) for c in contents]
        return VectorizeResponse(
            message="PDF processed and vectorized successfully.",
            document_ids=document_ids,
        )

    except RuntimeError as e:
        logger.warning("Runtime error for source '%s': %s", source, e)
        if "not found" in str(e).lower() or "404" in str(e):
             raise HTTPException(status_code=404, detail=f"File not found: {source}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        # This catches specific processing errors (like corrupt PDFs)
        logger.warning("PDF processing failed for source '%s': %s", source, e)
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {e}")
    except Exception as e:
        logger.exception("Unexpected error for source '%s': %s", source, e)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    
    finally:
        # 7. CLEANUP: Delete the temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {tmp_path}: {e}")