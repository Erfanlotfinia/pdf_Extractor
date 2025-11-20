from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid

class VectorizeRequest(BaseModel):
    """
    Pydantic model for the /vectorize endpoint request body.
    """
    source_url: str = Field(..., description="The URL of the PDF to be processed.")

class VectorizeResponse(BaseModel):
    """
    Pydantic model for a successful /vectorize endpoint response.
    """
    status: str = "success"
    message: str
    document_ids: List[str]

class ProcessedContent(BaseModel):
    """
    Internal model to structure content extracted from the PDF before vectorization.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: str  # "text", "image", or "table"
    text_content: str
    metadata: Dict[str, Any]  # Should include: page, file_hash, section, related_images

class ErrorResponse(BaseModel):
    """
    Pydantic model for a generic error response.
    """
    status: str = "error"
    detail: str
    code: Optional[int] = None