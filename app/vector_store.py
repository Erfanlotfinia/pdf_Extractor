# app/vector_store.py
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from config import get_settings
from schemas import ChunkWithVector

settings = get_settings()

_qdrant_client = QdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)


def ensure_collection(collection_name: str):
    collections = _qdrant_client.get_collections().collections
    existing_names = {c.name for c in collections}
    if collection_name not in existing_names:
        _qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(
    collection_name: str,
    chunks: List[ChunkWithVector]
) -> List[str]:
    ensure_collection(collection_name)

    points = []
    ids = []

    for chunk in chunks:
        ids.append(chunk.id)
        payload = {
            "id": chunk.id,
            "content_type": chunk.content_type,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        points.append(
            PointStruct(
                id=chunk.id,
                vector=chunk.vector,
                payload=payload,
            )
        )

    _qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

    return ids