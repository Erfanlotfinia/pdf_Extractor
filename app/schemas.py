# app/schemas.py
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from uuid import uuid4


class VectorizeRequest(BaseModel):
    pdf_url: Optional[HttpUrl] = None
    pdf_base64: Optional[str] = None
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    collection_name: str = "pdf_documents"
    language: str = "fa"  # default to Persian

    class Config:
        json_schema_extra = {
            "example": {
                "pdf_url": "https://example.com/sample.pdf",
                "document_id": "my-document-id",
                "collection_name": "pdf_documents",
                "language": "fa"
            }
        }

    def validate_source(self):
        if not self.pdf_url and not self.pdf_base64:
            raise ValueError("Either pdf_url or pdf_base64 must be provided.")


class ParsedChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content_type: Literal["text", "image", "table"]
    text: str
    metadata: Dict[str, Any]


class ChunkWithVector(ParsedChunk):
    vector: List[float]


class VectorizeResponse(BaseModel):
    status: Literal["success", "error"]
    document_id: str
    collection_name: str
    total_chunks: int
    detail: Optional[str] = None
    chunk_ids: Optional[List[str]] = None