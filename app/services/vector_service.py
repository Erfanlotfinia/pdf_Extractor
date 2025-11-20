from typing import List
from qdrant_client import AsyncQdrantClient, models
from langchain_openai import OpenAIEmbeddings

from core.config import settings
from models.schemas import ProcessedContent

class VectorService:
    """
    Service for handling vector operations, including embedding generation
    and interaction with a Qdrant vector database.
    """
    def __init__(self):
        self.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
        # The OpenAI embedding model is chosen for its performance and support
        # for various languages, including Persian (RTL). It correctly handles
        # UTF-8 encoded text.
        self.embedding_model = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    async def initialize_collection(self):
        """
        Ensures the Qdrant collection exists and is configured correctly.
        """
        try:
            await self.qdrant_client.get_collection(collection_name=self.collection_name)
        except Exception: # A more specific exception would be better in production
            await self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Dimension for text-embedding-3-small
                    distance=models.Distance.COSINE,
                ),
            )

    async def check_document_exists(self, file_hash: str) -> List[str]:
        """
        Checks if a document with the given hash already exists in Qdrant.
        
        Returns:
            A list of existing document IDs if found, otherwise an empty list.
        """
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
            limit=10000, # Assuming a PDF won't be chunked into more than 10k pieces
            with_payload=["id"],
        )
        return [record.id for record in response[0]]

    async def vectorize_and_upsert(self, contents: List[ProcessedContent]):
        """
        Generates embeddings for the processed content and upserts them to Qdrant.
        """
        if not contents:
            return

        await self.initialize_collection()

        texts_to_embed = [content.text_content for content in contents]
        vectors = await self.embedding_model.aembed_documents(texts_to_embed)

        points = [
            models.PointStruct(
                id=content.id,
                vector=vector,
                payload={
                    "content_type": content.content_type,
                    "text": content.text_content,
                    "metadata": content.metadata,  # includes page, section, related_images, file_hash
                },
            )
            for content, vector in zip(contents, vectors)
        ]

        await self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
