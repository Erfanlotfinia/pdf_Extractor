from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="PDF Vectorization Microservice",
    description="An API to extract content from PDFs, create embeddings, and store them in Qdrant.",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1", tags=["Vectorization"])

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    A simple health check endpoint.
    """
    return {"status": "ok"}
