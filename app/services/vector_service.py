from typing import List
from qdrant_client import AsyncQdrantClient, models
from langchain_openai import OpenAIEmbeddings
import logging

from core.config import settings
from models.schemas import ProcessedContent

class VectorService:
    """
    Service for handling vector operations, including embedding generation
    and interaction with a Qdrant vector database.
    """
    def __init__(self):
        try:
            self.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
        except Exception as e:
            logging.exception("Failed to initialize Qdrant client: %s", e)
            raise RuntimeError(f"Failed to initialize vector database client: {e}") from e

        try:
            # May raise if API key or configuration invalid
            self.embedding_model = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                api_key=settings.OPENAI_API_KEY,
            )
        except Exception as e:
            logging.exception("Failed to initialize embedding model: %s", e)
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

        self.collection_name = settings.QDRANT_COLLECTION_NAME

    async def initialize_collection(self):
        """
        Ensures the Qdrant collection exists and is configured correctly.
        """
        try:
            await self.qdrant_client.get_collection(collection_name=self.collection_name)
        except Exception:
            try:
                await self.qdrant_client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Dimension for text-embedding-3-small
                        distance=models.Distance.COSINE,
                    ),
                )
            except Exception as e:
                logging.exception("Failed to create/recreate Qdrant collection '%s': %s", self.collection_name, e)
                raise RuntimeError(f"Failed to ensure vector collection: {e}") from e

    async def check_document_exists(self, file_hash: str) -> List[str]:
        """
        Checks if a document with the given hash already exists in Qdrant.
        """
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
            # response may be a tuple (results, next_page)
            records = response[0] if isinstance(response, (list, tuple)) and response else []
            return [record.id for record in records]
        except Exception as e:
            logging.exception("Error checking document existence in Qdrant: %s", e)
            raise RuntimeError(f"Failed to check document existence: {e}") from e

    async def vectorize_and_upsert(self, contents: List[ProcessedContent]):
        """
        Generates embeddings for the processed content and upserts them to Qdrant.
        """
        if not contents:
            logging.info("No contents provided to vectorize_and_upsert; skipping.")
            return

        await self.initialize_collection()

        try:
            texts_to_embed = [content.text_content for content in contents]
            # aembed_documents is async in this usage
            vectors = await self.embedding_model.aembed_documents(texts_to_embed)
        except Exception as e:
            logging.exception("Embedding generation failed: %s", e)
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

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
