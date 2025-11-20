import httpx
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
    This endpoint accepts a file key for a PDF stored in MinIO, processes it, 
    and stores its vectorized content in a Qdrant database.
    """
    try:
        # The source_url is now the object key in MinIO
        file_key = request.source_url
        
        # Download the file from storage into a temporary buffer
        pdf_file_obj = storage_service.download_file(file_key)

        # 1. Process the PDF to extract content
        # The PDFProcessor will now read from the file-like object
        file_hash, contents = await pdf_processor.process_pdf(pdf_file_obj)
        
        # 2. Check if the document has already been processed
        existing_ids = await vector_service.check_document_exists(file_hash)
        if existing_ids:
            return VectorizeResponse(
                message="Document already processed.",
                document_ids=existing_ids
            )
            
        # 3. If not, vectorize the content and store it
        await vector_service.vectorize_and_upsert(contents)
        
        return VectorizeResponse(
            message="PDF processed and vectorized successfully.",
            document_ids=[content.id for content in contents]
        )

    except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
        raise HTTPException(status_code=503, detail=f"Storage service error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
