# app/embeddings.py
from typing import List
from openai import OpenAI
from config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)


def get_embedding(text: str, model: str = settings.embedding_model) -> List[float]:
    # Recommended: normalize newlines
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding