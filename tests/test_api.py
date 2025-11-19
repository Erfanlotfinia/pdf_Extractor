import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import respx

from app.main import app
from app.api.endpoints import get_pdf_processor, get_vector_service
from app.models.schemas import ProcessedContent

client = TestClient(app)

# --- Mocks ---

@pytest.fixture
def mock_pdf_processor():
    mock = AsyncMock()
    mock.process_pdf.return_value = ("test_hash", [ProcessedContent(
        id="test-uuid",
        content_type="text",
        text_content="This is a test.",
        metadata={"page": 1, "file_hash": "test_hash"}
    )])
    return mock

@pytest.fixture
def mock_vector_service():
    mock = AsyncMock()
    mock.check_document_exists.return_value = [] # Assume document does not exist
    mock.vectorize_and_upsert.return_value = None
    return mock


# --- Tests ---

@pytest.mark.asyncio
@respx.mock
async def test_vectorize_pdf_success(mock_pdf_processor, mock_vector_service):
    # Mock the external PDF download, which is part of the real PDFProcessor.
    # Even though we mock the service, it's good practice for clarity.
    respx.get("http://example.com/test.pdf").respond(
        200, content=b"%PDF-1.4...", headers={"Content-Type": "application/pdf"}
    )

    # Correctly override the dependency functions
    app.dependency_overrides[get_pdf_processor] = lambda: mock_pdf_processor
    app.dependency_overrides[get_vector_service] = lambda: mock_vector_service

    response = client.post(
        "/api/v1/vectorize",
        json={"source_url": "http://example.com/test.pdf"}
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "PDF processed and vectorized successfully" in json_response["message"]
    assert json_response["document_ids"] == ["test-uuid"]

    # Clean up dependency overrides
    app.dependency_overrides = {}


@pytest.mark.asyncio
@respx.mock  # Added the missing respx mock decorator
async def test_vectorize_pdf_already_processed(mock_pdf_processor, mock_vector_service):
    # Mock the PDF download
    respx.get("http://example.com/test.pdf").respond(
        200, content=b"%PDF-1.4...", headers={"Content-Type": "application/pdf"}
    )

    # Setup mock to simulate that the document already exists
    mock_vector_service.check_document_exists.return_value = ["existing_id_1", "existing_id_2"]

    app.dependency_overrides[get_pdf_processor] = lambda: mock_pdf_processor
    app.dependency_overrides[get_vector_service] = lambda: mock_vector_service

    response = client.post(
        "/api/v1/vectorize",
        json={"source_url": "http://example.com/test.pdf"}
    )

    assert response.status_code == 200
    json_response = response.json()
    assert "Document already processed" in json_response["message"]
    assert json_response["document_ids"] == ["existing_id_1", "existing_id_2"]

    app.dependency_overrides = {}

@pytest.mark.asyncio
@respx.mock
async def test_vectorize_pdf_invalid_url(mock_pdf_processor, mock_vector_service):
    # This test will fail if we don't override the dependencies, as the real services would be called.
    app.dependency_overrides[get_pdf_processor] = lambda: mock_pdf_processor
    app.dependency_overrides[get_vector_service] = lambda: mock_vector_service

    # Mock a failed download
    respx.get("http://example.com/notfound.pdf").respond(404)

    # The real PDFProcessor will be called here, which is what we want to test.
    # We need to remove the override for get_pdf_processor.
    app.dependency_overrides = {} # Clear all overrides for this specific test case

    response = client.post(
        "/api/v1/vectorize",
        json={"source_url": "http://example.com/notfound.pdf"}
    )

    assert response.status_code == 400
    assert "Failed to download PDF" in response.json()["detail"]
