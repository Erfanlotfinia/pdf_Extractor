import logging
import asyncio
from typing import List, Dict, Optional

# Resilience
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# AI & Vector DB
from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Local
from app.core.config import settings
from app.models.schemas import ProcessedContent

logger = logging.getLogger(__name__)

class VectorService:
    """
    Service for managing vector operations with Qdrant and OpenAI.
    Optimized for concurrent batch processing and RAG retrieval.
    """

    def __init__(self):
        self._batch_size = 50
        # Limits concurrent embedding requests to OpenAI to avoid RateLimitErrors
        self._concurrency_limit = 5 
        
        try:
            self.qdrant_client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY, 
                timeout=60, # Global timeout for requests
            )
            
            self.embedding_model = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model=settings.EMBEDDING_MODEL_NAME,
            )
            self.collection_name = settings.QDRANT_COLLECTION_NAME
            
        except Exception as e:
            logger.exception("Critical: Failed to initialize VectorService components.")
            raise RuntimeError(f"VectorService init failed: {e}") from e

    async def initialize(self):
        """
        Idempotent initialization of the Qdrant collection.
        """
        try:
            exists = await self.qdrant_client.collection_exists(collection_name=self.collection_name)
            
            if not exists:
                logger.info("Collection '%s' not found. Creating...", self.collection_name)
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=models.Distance.COSINE,
                    ),
                    # Optimize for search speed (HNSW)
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2, 
                        memmap_threshold=20000
                    )
                )
                logger.info("Successfully created Qdrant collection.")
            else:
                logger.info("Qdrant collection is ready.")
                
        except Exception as e:
            logger.exception("Failed to initialize Qdrant.")
            raise

    async def check_document_exists(self, file_hash: str) -> List[str]:
        """
        Checks if a document exists using the file_hash.
        Returns a list of IDs if found, else empty list.
        """
        try:
            # Efficient scroll to check for existence of any point with this hash
            response, _ = await self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_hash",
                            match=models.MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=100, # Just get enough to prove existence
                with_payload=False,
                with_vectors=False,
            )
            
            return [str(record.id) for record in response]

        except Exception as e:
            logger.error("Error checking document existence: %s", e)
            return []

    async def clean_file_data(self, file_hash: str):
        """
        Removes all vectors associated with a specific file hash.
        Essential for re-processing documents without duplicates.
        """
        try:
            logger.info(f"Cleaning existing data for hash: {file_hash}")
            await self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.file_hash",
                                match=models.MatchValue(value=file_hash),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.error(f"Error cleaning file data: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, UnexpectedResponse))
    )
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Internal helper with retry logic."""
        # Replace newlines to improve embedding quality for some models
        cleaned_texts = [t.replace("\n", " ") for t in texts]
        return await self.embedding_model.aembed_documents(cleaned_texts)

    async def _process_batch(self, batch: List[ProcessedContent], semaphore: asyncio.Semaphore):
        """
        Worker function to process a single batch inside a Semaphore context.
        """
        async with semaphore:
            try:
                # 1. Extract Text
                texts = [c.text_content for c in batch]
                
                # 2. Embed (IO Bound - Await)
                vectors = await self._generate_embeddings(texts)

                # 3. Map to Qdrant Points
                points = []
                for content, vector in zip(batch, vectors):
                    # Flatten metadata for easier filtering in Qdrant
                    # e.g., instead of payload.metadata.page, use payload.page
                    base_payload = {
                        "content_type": content.content_type,
                        "text": content.text_content,
                        "image_data": content.image_data if hasattr(content, 'image_data') else None
                    }
                    
                    # Merge flattened metadata
                    meta_dict = content.metadata.model_dump(mode='json')
                    full_payload = {**base_payload, **meta_dict}

                    points.append(models.PointStruct(
                        id=str(content.id),
                        vector=vector,
                        payload=full_payload
                    ))

                # 4. Upsert (IO Bound - Await)
                await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                raise e

    async def vectorize_and_upsert(self, contents: List[ProcessedContent], file_hash: str, force_reload: bool = False):
        """
        Main pipeline:
        1. Cleans old data for this file if force_reload is True.
        2. Chunks data into batches.
        3. Processes batches concurrently.
        """
        if not contents:
            return

        # 1. Clean old data to enforce consistency
        if force_reload:
            await self.clean_file_data(file_hash)

        total_items = len(contents)
        logger.info(f"Starting concurrent vectorization for {total_items} items...")

        # 2. Create Batches
        batches = [
            contents[i : i + self._batch_size] 
            for i in range(0, total_items, self._batch_size)
        ]

        # 3. Process Concurrently
        semaphore = asyncio.Semaphore(self._concurrency_limit)
        tasks = [self._process_batch(batch, semaphore) for batch in batches]

        # Gather results (raises exception if any batch fails)
        await asyncio.gather(*tasks)

        logger.info(f"Successfully upserted {total_items} chunks for hash {file_hash}.")

    async def search(self, query: str, limit: int = 5, file_hash: Optional[str] = None) -> List[Dict]:
        """
        Semantic search functionality for the RAG pipeline.
        """
        try:
            # 1. Convert query to vector
            query_vector = await self.embedding_model.aembed_query(query)

            # 2. Build Filters (Optional)
            query_filter = None
            if file_hash:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_hash", 
                            match=models.MatchValue(value=file_hash)
                        )
                    ]
                )

            # 3. Search Qdrant
            results = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )

            # 4. Format Results
            return [
                {
                    "score": hit.score,
                    "text": hit.payload.get("text"),
                    "page": hit.payload.get("page"),
                    "section": hit.payload.get("section"),
                    "content_type": hit.payload.get("content_type"),
                    "metadata": hit.payload
                }
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def close(self):
        await self.qdrant_client.close()