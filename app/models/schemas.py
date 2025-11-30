from typing import List, Optional, Any, Dict
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator, ConfigDict

# --- Internal Data Structures ---

class DocumentMetadata(BaseModel):
    """
    A structured model for the metadata of a processed content chunk.
    This matches the schema stored in Qdrant.
    """
    page: int = Field(..., description="The page number where the content was found.")
    section: Optional[str] = Field("General", description="The logical section (e.g., 'Introduction').")
    related_images: List[str] = Field(default_factory=list, description="IDs/Descriptions of images on the same page.")
    file_hash: str = Field(..., description="The SHA256 hash of the original PDF file.")

    # Allow extra fields if Unstructured adds dynamic metadata later
    model_config = ConfigDict(extra='ignore') 

class ProcessedContent(BaseModel):
    """
    Represents a single chunk of processed content ready for vectorization.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique ID for this chunk.")
    content_type: str = Field(..., description="Type of content: 'text', 'table', 'image'.")
    text_content: str = Field(..., description="The actual text to embed.")
    image_data: Optional[str] = Field(None, description="Base64 string of the image (if applicable).")
    metadata: DocumentMetadata = Field(..., description="Structured metadata.")

# --- API Request Models ---

class VectorizeRequest(BaseModel):
    """
    Request body for the /vectorize endpoint.
    """
    file_key: Optional[str] = Field(None, description="The MinIO key of an uploaded file.")
    source_url: Optional[str] = Field(None, description="A direct URL to a PDF.")
    force_reload: bool = Field(
        False, 
        description="If True, re-processes the file even if the hash exists in the DB."
    )

    @model_validator(mode='after')
    def check_exactly_one_source(self) -> 'VectorizeRequest':
        """Ensures that either 'file_key' or 'source_url' is provided, but not both."""
        has_key = bool(self.file_key and self.file_key.strip())
        has_url = bool(self.source_url and self.source_url.strip())

        if has_key and has_url:
            raise ValueError("Provide either 'file_key' or 'source_url', but not both.")
        if not has_key and not has_url:
            raise ValueError("Either 'file_key' or 'source_url' must be provided.")
        return self

class SearchRequest(BaseModel):
    """
    Request body for the /search endpoint.
    """
    query: str = Field(..., min_length=3, description="The semantic search query.")
    limit: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve.")
    file_hash: Optional[str] = Field(None, description="Optional: Limit search to a specific document hash.")

# --- API Response Models ---

class ApiResponse(BaseModel):
    """Base response model."""
    status: str = "success"
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = "error"
    detail: str

class UploadResponse(ApiResponse):
    """Response for /upload."""
    file_key: str = Field(..., description="The unique key assigned to the uploaded file.")

class VectorizeResponse(ApiResponse):
    """Response for /vectorize."""
    document_ids: List[str] = Field(default_factory=list, description="IDs of the vectorized chunks.")
    file_hash: Optional[str] = Field(None, description="The hash of the processed file (use this for filtering searches).")

class SearchResult(BaseModel):
    """Represents a single hit from the vector database."""
    score: float = Field(..., description="Similarity score (0 to 1).")
    text: str = Field(..., description="The content text.")
    page: Optional[int] = Field(None, description="Page number.")
    section: Optional[str] = Field(None, description="Document section.")
    content_type: Optional[str] = Field("text", description="text, table, or image.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Full metadata payload.")

class SearchResponse(ApiResponse):
    """Response for /search."""
    results: List[SearchResult]