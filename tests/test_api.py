import pytest
from unittest.mock import MagicMock, AsyncMock

from fastapi.testclient import TestClient

# Mock the service classes BEFORE they are imported by the main application
mock_storage_cls = MagicMock()
mock_vector_cls = MagicMock()
mock_processor_cls = MagicMock()

@pytest.fixture(autouse=True)
def patch_services(monkeypatch):
    """Patch the service classes to return async-compatible mocks."""
    # Configure the mock classes to return AsyncMock instances
    storage_instance = AsyncMock()
    vector_instance = AsyncMock()
    processor_instance = AsyncMock()

    # CRITICAL: Ensure the default check for existing documents returns an empty list (falsy)
    vector_instance.check_document_exists.return_value = []

    mock_storage_cls.return_value = storage_instance
    mock_vector_cls.return_value = vector_instance
    mock_processor_cls.return_value = processor_instance

    monkeypatch.setattr("app.main.MinioStorageService", mock_storage_cls)
    monkeypatch.setattr("app.main.VectorService", mock_vector_cls)
    monkeypatch.setattr("app.main.PDFProcessorService", mock_processor_cls)

    yield storage_instance, vector_instance, processor_instance

# Now, we can safely import the app
from app.main import app
from app.models.schemas import DocumentMetadata, ProcessedContent

@pytest.fixture
def client():
    """Provides a TestClient instance for the tests."""
    with TestClient(app) as c:
        yield c

# --- Test Cases ---

def test_health_check(client):
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_upload_pdf_success(client, patch_services):
    """Test successful PDF upload."""
    storage_mock, _, _ = patch_services
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "file_key" in json_response
    storage_mock.upload_file.assert_awaited_once()

def test_upload_pdf_invalid_file_type(client):
    """Test upload with a non-PDF file type."""
    files = {"file": ("test.txt", b"not a pdf", "text/plain")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are allowed" in response.json()["detail"]

def test_vectorize_with_file_key_success(client, patch_services):
    """Test successful vectorization using a file_key."""
    storage_mock, vector_mock, processor_mock = patch_services
    processor_mock.process_pdf.return_value = ("mock_hash", [ProcessedContent(
        content_type="text", text_content="Sample", metadata=DocumentMetadata(
            page=1, section="s1", file_hash="mock_hash"
        )
    )])
    
    response = client.post("/api/v1/vectorize", json={"file_key": "key.pdf"})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    storage_mock.download_file.assert_awaited_with("key.pdf")
    processor_mock.process_pdf.assert_awaited_once()
    vector_mock.vectorize_and_upsert.assert_awaited_once()

def test_vectorize_document_already_processed(client, patch_services):
    """Test the case where the document has already been vectorized."""
    _, vector_mock, processor_mock = patch_services
    processor_mock.process_pdf.return_value = ("mock_hash", [])
    vector_mock.check_document_exists.return_value = ["existing_uuid"]

    response = client.post("/api/v1/vectorize", json={"file_key": "processed.pdf"})
    assert response.status_code == 200
    assert "already been processed" in response.json()["message"]
    vector_mock.vectorize_and_upsert.assert_not_awaited()
