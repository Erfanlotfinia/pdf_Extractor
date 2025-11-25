import logging
import asyncio
from typing import List, Optional
from uuid import UUID

# Retry logic library (Best practice for external APIs)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.models.schemas import ProcessedContent

logger = logging.getLogger(__name__)

class VectorService:
    """
    Service for managing vector operations with Qdrant and OpenAI embeddings.
    Optimized for batch processing and high resilience.
    """

    # Batch size to prevent hitting OpenAI rate limits or Qdrant payload limits
    BATCH_SIZE = 50 

    def __init__(self):
        try:
            self.qdrant_client = AsyncQdrantClient(
                url=settings.QDRANT_URL, 
                timeout=60 # Increased timeout for heavier loads
            )
            
            self.embedding_model = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model=settings.EMBEDDING_MODEL_NAME,
                # LangChain handles async automatically
            )
            self.collection_name = settings.QDRANT_COLLECTION_NAME
        except Exception as e:
            logger.exception("Critical: Failed to initialize VectorService components.")
            raise RuntimeError(f"VectorService init failed: {e}") from e

    async def initialize(self):
        """
        Idempotent initialization of the Qdrant collection.
        Checks existence before attempting creation to prevent data loss.
        """
        try:
            # Check if collection exists
            exists = await self.qdrant_client.collection_exists(collection_name=self.collection_name)
            
            if exists:
                logger.info("Qdrant collection '%s' is ready.", self.collection_name)
            else:
                logger.info("Collection '%s' not found. Creating...", self.collection_name)
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("Successfully created Qdrant collection '%s'.", self.collection_name)
                
        except Exception as e:
            logger.exception("Failed to initialize Qdrant collection state.")
            raise RuntimeError(f"Qdrant initialization failed: {e}") from e

    async def check_document_exists(self, file_hash: str) -> List[str]:
        """
        Checks if a document exists using the file_hash.
        Optimized to return only IDs, saving bandwidth.
        """
        try:
            # We filter by file_hash
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.file_hash",
                        match=models.MatchValue(value=file_hash),
                    )
                ]
            )

            # Use scroll to get existing points. 
            # Optimized: with_payload=False, with_vectors=False 
            response, _ = await self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1000, 
                with_payload=False,
                with_vectors=False,
            )
            
            # Return list of UUID strings
            return [str(record.id) for record in response]

        except Exception as e:
            logger.error("Error checking document existence: %s", e)
            # Return empty list on error instead of crashing, or re-raise depending on preference
            raise RuntimeError(f"Qdrant existence check failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, UnexpectedResponse))
    )
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Internal helper to generate embeddings with retry logic.
        """
        return await self.embedding_model.aembed_documents(texts)

    async def vectorize_and_upsert(self, contents: List[ProcessedContent]):
        """
        Main processing pipeline.
        1. Batches content to respect API limits.
        2. Generates embeddings.
        3. Upserts to Qdrant.
        """
        if not contents:
            logger.warning("Empty content list provided to vectorizer.")
            return

        total_items = len(contents)
        logger.info("Starting vectorization for %d items...", total_items)

        # Process in batches
        for i in range(0, total_items, self.BATCH_SIZE):
            batch = contents[i : i + self.BATCH_SIZE]
            batch_texts = [c.text_content for c in batch]
            
            try:
                # 1. Generate Embeddings (with retry)
                vectors = await self._generate_embeddings_batch(batch_texts)

                # 2. Prepare Qdrant Points
                points = []
                for content, vector in zip(batch, vectors):
                    # Use Pydantic v2 model_dump() for cleaner serialization
                    payload = {
                        "content_type": content.content_type,
                        "text": content.text_content,
                        "metadata": content.metadata.model_dump(mode='json'), 
                    }
                    
                    points.append(models.PointStruct(
                        id=str(content.id),
                        vector=vector,
                        payload=payload
                    ))

                # 3. Upsert Batch
                await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True # Ensure consistency
                )
                
                logger.info("Processed batch %d to %d of %d", i, i + len(batch), total_items)

            except Exception as e:
                logger.error("Failed to process batch starting at index %d: %s", i, e)
                # In production, you might want to implement a 'Dead Letter Queue' logic here
                # For now, we raise to stop the process and alert the user
                raise RuntimeError(f"Batch processing failed: {e}") from e

        logger.info("Vectorization and upsert completed successfully.")

    async def close(self):
        """Graceful shutdown."""
        await self.qdrant_client.close()
        logger.info("Qdrant client closed.")