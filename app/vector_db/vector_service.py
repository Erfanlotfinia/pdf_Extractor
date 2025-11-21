import logging
from typing import List

from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.models.schemas import ProcessedContent

logger = logging.getLogger(__name__)

class VectorService:
    """A service for managing vector operations with Qdrant and OpenAI embeddings."""

    def __init__(self):
        try:
            # Initialize the async Qdrant client
            self.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)

            # Initialize the LangChain OpenAI embeddings model, configured for async usage
            self.embedding_model = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model=settings.EMBEDDING_MODEL_NAME,
                # The underlying client will be async by default with recent langchain versions
            )
        except Exception as e:
            logger.exception("Failed to initialize VectorService components: %s", e)
            raise RuntimeError(f"Failed to initialize VectorService: {e}") from e

        self.collection_name = settings.QDRANT_COLLECTION_NAME

    async def initialize(self):
        """Ensures the Qdrant collection exists with the correct configuration."""
        try:
            await self.qdrant_client.get_collection(collection_name=self.collection_name)
            logger.info("Qdrant collection '%s' already exists.", self.collection_name)
        except Exception:
            logger.info("Qdrant collection '%s' not found. Creating it.", self.collection_name)
            try:
                await self.qdrant_client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("Successfully created Qdrant collection '%s'.", self.collection_name)
            except Exception as e:
                logger.exception("Failed to create Qdrant collection '%s': %s", self.collection_name, e)
                raise RuntimeError(f"Failed to create vector collection: {e}") from e

    async def check_document_exists(self, file_hash: str) -> List[str]:
        """
        Checks if any vectors with the given file_hash already exist in the collection.

        Returns:
            A list of existing document IDs associated with the file hash.
        """
        try:
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
                limit=1000, # A reasonable limit to find all chunks of a single doc
                with_payload=False,
                with_vectors=False,
            )
            return [record.id for record in response]
        except Exception as e:
            logger.exception("Document existence check failed in Qdrant: %s", e)
            raise RuntimeError(f"Failed to check document existence: {e}") from e

    async def vectorize_and_upsert(self, contents: List[ProcessedContent]):
        """
        Generates embeddings for the processed content and upserts them into Qdrant.
        """
        if not contents:
            logger.warning("No content provided to vectorize_and_upsert.")
            return

        texts_to_embed = [c.text_content for c in contents]

        try:
            # Generate embeddings asynchronously using the LangChain model
            vectors = await self.embedding_model.aembed_documents(texts_to_embed)
        except Exception as e:
            logger.exception("Failed to generate embeddings: %s", e)
            raise RuntimeError(f"Embedding generation failed: {e}") from e

        # Construct the points to be upserted into Qdrant
        points = [
            models.PointStruct(
                id=str(content.id),  # Ensure ID is a string
                vector=vector,
                payload={
                    "content_type": content.content_type,
                    "text": content.text_content,
                    "metadata": content.metadata.dict(),  # Convert Pydantic model to dict
                },
            )
            for content, vector in zip(contents, vectors)
        ]

        try:
            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,  # Wait for the operation to complete
            )
            logger.info("Successfully upserted %d vectors to Qdrant.", len(points))
        except Exception as e:
            logger.exception("Failed to upsert vectors to Qdrant: %s", e)
            raise RuntimeError(f"Failed to upsert vectors: {e}") from e

    async def close(self):
        """Closes the Qdrant client connection."""
        await self.qdrant_client.close()
        logger.info("Qdrant client connection closed.")
