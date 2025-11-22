from typing import List, Optional, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator

# --- Internal Data Structures ---

class DocumentMetadata(BaseModel):
    """A structured model for the metadata of a processed content chunk."""
    page: int = Field(..., description="The page number where the content was found.")
    section: Optional[str] = Field(None, description="The logical section of the document (e.g., 'Introduction').")
    related_images: List[str] = Field(default_factory=list, description="A list of IDs for images on the same page.")
    file_hash: str = Field(..., description="The SHA256 hash of the original PDF file.")

class ProcessedContent(BaseModel):
    """An internal model representing a single chunk of processed content to be vectorized."""
    id: UUID = Field(default_factory=uuid4, description="The unique identifier for this content chunk.")
    content_type: str = Field(..., description="The type of content, e.g., 'text', 'image', 'table'.")
    text_content: str = Field(..., description="The textual representation of the content.")
    metadata: DocumentMetadata = Field(..., description="The structured metadata associated with the content.")

# --- API Request Models ---

class VectorizeRequest(BaseModel):
    """Defines the request body for the /vectorize endpoint."""
    file_key: Optional[str] = Field(None, description="The key of a file previously uploaded via /upload.")
    source_url: Optional[str] = Field(None, description="A public URL to a PDF file.")

    @model_validator(mode='before')
    @classmethod
    def check_exactly_one_source_provided(cls, data: Any) -> Any:
        """Ensures that either 'file_key' or 'source_url' is provided, but not both."""
        if isinstance(data, dict):
            if ('file_key' in data and data.get('file_key')) and \
               ('source_url' in data and data.get('source_url')):
                raise ValueError("Provide either 'file_key' or 'source_url', but not both.")
            if not ('file_key' in data and data.get('file_key')) and \
               not ('source_url' in data and data.get('source_url')):
                raise ValueError("Either 'file_key' or 'source_url' must be provided.")
        return data

# --- API Response Models ---

class ApiResponse(BaseModel):
    """A standard base for API responses."""
    status: str = "success"
    message: Optional[str] = None

class UploadResponse(ApiResponse):
    """The response for a successful file upload."""
    file_key: str = Field(..., description="The unique key assigned to the uploaded file.")
    message: str = "File uploaded successfully."

class VectorizeResponse(ApiResponse):
    """The response for a successful vectorization request."""
    document_ids: List[str] = Field(..., description="A list of unique IDs for the vectorized content chunks.")
    
class ErrorResponse(BaseModel):
    """A standard model for error responses."""
    status: str = "error"
    detail: str
