# app/main.py
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config import get_settings
from schemas import (
    VectorizeRequest,
    VectorizeResponse,
    ParsedChunk,
    ChunkWithVector,
)
from pdf_parser import (
    load_pdf_from_url,
    load_pdf_from_base64,
    parse_pdf,
)
from embeddings import get_embedding
from vector_store import upsert_chunks

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
)


@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_pdf(req: VectorizeRequest):
    try:
        req.validate_source()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- 1. Load PDF bytes ---
    try:
        if req.pdf_url:
            pdf_bytes = load_pdf_from_url(str(req.pdf_url))
        else:
            pdf_bytes = load_pdf_from_base64(req.pdf_base64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load PDF: {e}",
        )

    # --- 2. Parse PDF into chunks ---
    try:
        parsed_chunks: List[ParsedChunk] = parse_pdf(
            pdf_bytes=pdf_bytes,
            document_id=req.document_id,
            language=req.language,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PDF: {e}",
        )

    if not parsed_chunks:
        return VectorizeResponse(
            status="success",
            document_id=req.document_id,
            collection_name=req.collection_name,
            total_chunks=0,
            detail="No content extracted from PDF.",
        )

    # --- 3. Generate embeddings ---
    chunks_with_vectors: List[ChunkWithVector] = []
    for chunk in parsed_chunks:
        try:
            vector = get_embedding(chunk.text)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to embed chunk {chunk.id}: {e}",
            )

        chunks_with_vectors.append(
            ChunkWithVector(
                **chunk.model_dump(),
                vector=vector,
            )
        )

    # --- 4. Store in Qdrant ---
    try:
        ids = upsert_chunks(
            collection_name=req.collection_name,
            chunks=chunks_with_vectors,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store vectors in Qdrant: {e}",
        )

    return VectorizeResponse(
        status="success",
        document_id=req.document_id,
        collection_name=req.collection_name,
        total_chunks=len(chunks_with_vectors),
        chunk_ids=ids,
    )


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})