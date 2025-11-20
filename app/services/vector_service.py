from typing import List
from qdrant_client import AsyncQdrantClient, models
from openai import OpenAI
import anyio
import logging

from core.config import settings
from models.schemas import ProcessedContent


class VectorService:
    """
    Vector database manager using Avalai embeddings + Qdrant.
    """

    def __init__(self):
        # ---- QDRANT ---------------------------------------------------------
        try:
            self.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
        except Exception as e:
            logging.exception("Failed to initialize Qdrant client: %s", e)
            raise RuntimeError(f"Failed to initialize vector DB client: {e}") from e

        # ---- AVALAI OPENAI-COMPATIBLE CLIENT --------------------------------
        try:
            self.embedding_client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,   # <-- IMPORTANT
            )
        except Exception as e:
            logging.exception("Failed to initialize Avalai client: %s", e)
            raise RuntimeError(f"Failed to initialize embedding provider: {e}") from e

        self.model_name = settings.EMBEDDING_MODEL_NAME  # e.g. "text-embedding-3-small"
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    # -------------------------------------------------------------------------
    async def initialize_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            await self.qdrant_client.get_collection(self.collection_name)
        except Exception:
            try:
                await self.qdrant_client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,       # Avalai text-embedding-3-small dimension
                        distance=models.Distance.COSINE,
                    ),
                )
            except Exception as e:
                logging.exception("Failed to create/recreate Qdrant collection '%s': %s",
                                  self.collection_name, e)
                raise RuntimeError(f"Failed to ensure vector collection: {e}") from e

    # -------------------------------------------------------------------------
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Avalai embeddings endpoint is synchronous,
        so we wrap it in a thread → async compatible.
        """

        def sync_embed():
            return self.embedding_client.embeddings.create(
                model=self.model_name,
                input=texts
            )

        try:
            response = await anyio.to_thread.run_sync(sync_embed)
        except Exception as e:
            logging.exception("Embedding generation failed: %s", e)
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

        return [item.embedding for item in response.data]

    # -------------------------------------------------------------------------
    async def check_document_exists(self, file_hash: str) -> List[str]:
        """Check if a document with file_hash exists in Qdrant."""
        try:
            response = await self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_hash",
                            match=models.MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=10000,
                with_payload=["id"],
            )

            records = response[0] if isinstance(response, (list, tuple)) and response else []
            return [record.id for record in records]

        except Exception as e:
            logging.exception("Document existence check failed: %s", e)
            raise RuntimeError(f"Failed to check document existence: {e}") from e

    # -------------------------------------------------------------------------
    async def vectorize_and_upsert(self, contents: List[ProcessedContent]):
        """Generate Avalai embeddings + upsert to Qdrant."""
        if not contents:
            logging.info("No contents provided to vectorize_and_upsert.")
            return

        await self.initialize_collection()

        texts = [c.text_content for c in contents]

        # ---- EMBEDDINGS -----------------------------------------------------
        vectors = await self.generate_embeddings(texts)

        # ---- UPSERT ---------------------------------------------------------
        try:
            points = [
                models.PointStruct(
                    id=content.id,
                    vector=vector,
                    payload={
                        "content_type": content.content_type,
                        "text": content.text_content,
                        "metadata": content.metadata,
                    },
                )
                for content, vector in zip(contents, vectors)
            ]

            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        except Exception as e:
            logging.exception("Failed to upsert vectors to Qdrant: %s", e)
            raise RuntimeError(f"Failed to upsert vectors: {e}") from e
