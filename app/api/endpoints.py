import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from uuid import uuid4
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

from models.schemas import VectorizeRequest, VectorizeResponse, ErrorResponse, UploadResponse
from services.pdf_processor import PDFProcessor
from services.vector_service import VectorService
from services.storage_service import StorageService, MinioStorageService

router = APIRouter()

# --- Dependency Injection ---
def get_pdf_processor() -> PDFProcessor:
    return PDFProcessor()

def get_vector_service() -> VectorService:
    return VectorService()

def get_storage_service() -> StorageService:
    return MinioStorageService()
# --------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        503: {"model": ErrorResponse, "description": "Storage service unavailable"},
    },
    tags=["Upload"],
)
async def upload_pdf(
    file: UploadFile = File(...),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Upload a PDF file, stream it to MinIO, and return a unique file key.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_key = f"{str(uuid4())}.pdf"

    try:
        storage_service.upload_file(file.file, file_key)
        return UploadResponse(file_key=file_key)
    except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
        # Catch specific boto3 exceptions for storage issues
        raise HTTPException(status_code=503, detail=f"Storage service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.post(
    "/vectorize",
    response_model=VectorizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL or file type"},
        422: {"model": ErrorResponse, "description": "PDF Parsing failed"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Storage service unavailable"},
    },
)
async def vectorize_pdf(
    request: VectorizeRequest,
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    vector_service: VectorService = Depends(get_vector_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    This endpoint accepts a file key or an external URL for a PDF, processes it,
    and stores its vectorized content in a Qdrant database.
    """
    try:
        file_key = request.source_url

        try:
            pdf_file_obj = storage_service.download_file(file_key)
        except ValueError as e:
            logging.warning("Download error for '%s': %s", file_key, e)
            raise HTTPException(status_code=400, detail=str(e))
        except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
            logging.exception("Storage credentials or client error: %s", e)
            raise HTTPException(status_code=503, detail=f"Storage service error: {e}")
        except RuntimeError as e:
            logging.exception("Storage service runtime error: %s", e)
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logging.exception("Unexpected error downloading file '%s': %s", file_key, e)
            raise HTTPException(status_code=500, detail=f"Unexpected storage error: {e}")

        # 1. Process the PDF to extract content
        try:
            file_hash, contents = await pdf_processor.process_pdf(pdf_file_obj)
        except ValueError as e:
            logging.warning("PDF processing failed for '%s': %s", file_key, e)
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logging.exception("Unexpected PDF processing error: %s", e)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while parsing PDF: {e}")

        # 2. Check if the document has already been processed
        try:
            existing_ids = await vector_service.check_document_exists(file_hash)
        except RuntimeError as e:
            logging.exception("Vector service check failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logging.exception("Unexpected error checking document existence: %s", e)
            raise HTTPException(status_code=500, detail=f"Vector service error: {e}")

        if existing_ids:
            return VectorizeResponse(
                message="Document already processed.",
                document_ids=existing_ids
            )

        # 3. If not, vectorize the content and store it
        try:
            await vector_service.vectorize_and_upsert(contents)
        except RuntimeError as e:
            logging.exception("Vectorization/upsert failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logging.exception("Unexpected vectorization error: %s", e)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while vectorizing: {e}")

        return VectorizeResponse(
            message="PDF processed and vectorized successfully.",
            document_ids=[content.id for content in contents]
        )

    except HTTPException:
        # re-raise HTTPExceptions unchanged
        raise
    except Exception as e:
        logging.exception("Unhandled exception in vectorize endpoint: %s", e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
