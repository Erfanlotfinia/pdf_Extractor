import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import VectorizeRequest, VectorizeResponse, ErrorResponse
from app.services.pdf_processor import PDFProcessor
from app.services.vector_service import VectorService

router = APIRouter()

# --- Dependency Injection ---
def get_pdf_processor() -> PDFProcessor:
    # This creates a new client for each request. For high-volume services,
    # you might consider a shared client instance.
    return PDFProcessor(client=httpx.AsyncClient())

def get_vector_service() -> VectorService:
    return VectorService()
# --------------------------

@router.post(
    "/vectorize",
    response_model=VectorizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL or file type"},
        422: {"model": ErrorResponse, "description": "PDF Parsing failed"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def vectorize_pdf(
    request: VectorizeRequest,
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    vector_service: VectorService = Depends(get_vector_service),
):
    """
    This endpoint accepts a URL to a PDF, processes it, and stores its
    vectorized content in a Qdrant database.
    """
    try:
        # 1. Process the PDF to extract content
        file_hash, contents = await pdf_processor.process_pdf(request.source_url)

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

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # This is a broad exception handler. In a production environment,
        # it's better to have more specific handlers for different errors
        # (e.g., Qdrant connection errors, OpenAI API errors).
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
