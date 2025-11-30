import logging
import os
import tempfile
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request

from app.models.schemas import (
    VectorizeRequest,
    VectorizeResponse,
    ErrorResponse,
    UploadResponse,
    SearchRequest,
    SearchResponse
)
from app.processing.pdf_processor import PDFProcessorService
from app.storage.storage_service import StorageService
from app.vector_db.vector_service import VectorService

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection Helpers ---

def get_pdf_processor(request: Request) -> PDFProcessorService:
    return request.app.state.pdf_processor_service

def get_vector_service(request: Request) -> VectorService:
    return request.app.state.vector_service

def get_storage_service(request: Request) -> StorageService:
    return request.app.state.storage_service

# ------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF to Storage",
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        503: {"model": ErrorResponse, "description": "Storage unavailable"}
    }
)
async def upload_pdf(
    file: UploadFile = File(..., description="The PDF file to upload."),
    storage_service: StorageService = Depends(get_storage_service),
):
    """
    Streams a PDF file to object storage (MinIO) and returns a file_key.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_key = f"{uuid4()}.pdf"
    
    try:
        # Determine file size (Efficiently, without reading into memory if possible)
        file.file.seek(0, 2)
        file_size = file.file.tell()
        await file.seek(0)
        
        # Upload using the optimized stream method
        await storage_service.upload_file(file.file, file_key, file_size)
        
        return UploadResponse(file_key=file_key, message="File uploaded successfully.")
    
    except Exception as e:
        logger.exception(f"Upload failed for {file_key}: {e}")
        raise HTTPException(status_code=503, detail="Storage service unavailable.")


@router.post(
    "/vectorize",
    response_model=VectorizeResponse,
    summary="Process and Vectorize a PDF",
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
        422: {"model": ErrorResponse, "description": "Processing failed"},
        500: {"model": ErrorResponse, "description": "Internal error"}
    }
)
async def vectorize_pdf(
    request: VectorizeRequest,
    pdf_processor: PDFProcessorService = Depends(get_pdf_processor),
    vector_service: VectorService = Depends(get_vector_service),
    storage_service: StorageService = Depends(get_storage_service),
):
    """
    1. Streams PDF from Storage/URL -> Local Temp File (Low RAM usage).
    2. Extracts text/images.
    3. Checks for duplicates (Idempotency).
    4. Embeds and Upserts to Qdrant.
    """
    source = request.file_key or request.source_url
    # Validator in schema handles mutual exclusivity, but double check doesn't hurt
    if not source:
        raise HTTPException(status_code=400, detail="Provide 'file_key' or 'source_url'.")

    tmp_path = None

    try:
        # 1. Prepare Temp File Path
        # delete=False is required so we can close it and let the processor open it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
        
        # 2. Stream Download to Disk (Memory Optimized)
        try:
            await storage_service.download_to_path(source, tmp_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found: {source}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise HTTPException(status_code=503, detail="Failed to retrieve file from storage/url.")

        # 3. Process PDF (Compute Hash + Extract)
        # processing relies on the file on disk
        try:
            file_hash, contents = await pdf_processor.process_pdf(tmp_path)
        except ValueError as e:
             raise HTTPException(status_code=422, detail=f"PDF Processing failed: {str(e)}")

        # 4. Idempotency Check
        # We check the DB using the hash calculated from the actual file
        existing_ids = await vector_service.check_document_exists(file_hash)
        
        if existing_ids and not request.force_reload:
            logger.info(f"File hash {file_hash} already exists. Returning existing IDs.")
            return VectorizeResponse(
                message="Document already processed.",
                document_ids=existing_ids,
                file_hash=file_hash
            )

        if not contents:
            logger.warning(f"No content extracted from {source}")
            return VectorizeResponse(
                message="No extractable content found.",
                document_ids=[],
                file_hash=file_hash
            )

        # 5. Vectorize & Upsert
        # Pass file_hash to allow cleaning old vectors before write (Consistency)
        await vector_service.vectorize_and_upsert(contents, file_hash)
        
        new_ids = [str(c.id) for c in contents]
        
        return VectorizeResponse(
            message="Successfully processed and vectorized.",
            document_ids=new_ids,
            file_hash=file_hash
        )

    except HTTPException:
        raise # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.exception(f"Unexpected error processing {source}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
    
    finally:
        # 6. Cleanup Temp File
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {tmp_path}: {e}")

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Search",
    description="Retrieve relevant context chunks for a given query."
)
async def search_documents(
    request: SearchRequest,
    vector_service: VectorService = Depends(get_vector_service),
):
    """
    Performs a cosine-similarity search in Qdrant.
    """
    try:
        results = await vector_service.search(
            query=request.query,
            limit=request.limit,
            file_hash=request.file_hash
        )
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Search service failed.")
