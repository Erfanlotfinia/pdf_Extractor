import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from fastapi.testclient import TestClient

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create mock classes BEFORE importing the app
mock_storage_cls = MagicMock()
mock_vector_cls = MagicMock()
mock_processor_cls = MagicMock()

@pytest.fixture
def patch_services(monkeypatch):
    """Patch service classes and return async-mocked instances."""

    # Create async-compatible service instances
    storage_instance = AsyncMock()
    vector_instance = AsyncMock()
    processor_instance = AsyncMock()

    # Default: document does NOT exist
    vector_instance.check_document_exists.return_value = []

    # Make the mock classes return these instances
    mock_storage_cls.return_value = storage_instance
    mock_vector_cls.return_value = vector_instance
    mock_processor_cls.return_value = processor_instance

    # Patch the concrete service classes instantiated in the app's lifespan
    monkeypatch.setattr("app.main.MinioStorageService", mock_storage_cls)
    monkeypatch.setattr("app.main.VectorService", mock_vector_cls)
    monkeypatch.setattr("app.main.PDFProcessorService", mock_processor_cls)

    return storage_instance, vector_instance, processor_instance


# Import the app AFTER mocks exist
from app.main import app
from app.models.schemas import DocumentMetadata, ProcessedContent


@pytest.fixture
def client(patch_services):
    """A TestClient that uses the patched services."""
    with TestClient(app) as c:
        yield c


# ------------------ TESTS ------------------ #

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_pdf_success(client, patch_services):
    storage_mock, _, _ = patch_services
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}

    response = client.post("/api/v1/upload", files=files)

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "file_key" in json_response

    storage_mock.upload_file.assert_awaited_once()


def test_upload_pdf_invalid_file_type(client):
    files = {"file": ("test.txt", b"not a pdf", "text/plain")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are allowed" in response.json()["detail"]


def test_vectorize_with_file_key_success(client, patch_services):
    storage_mock, vector_mock, processor_mock = patch_services

    processor_mock.process_pdf.return_value = (
        "mock_hash",
        [
            ProcessedContent(
                content_type="text",
                text_content="Sample",
                metadata=DocumentMetadata(
                    page=1,
                    section="s1",
                    file_hash="mock_hash"
                )
            )
        ]
    )

    # Configure the mock to return bytes, simulating a downloaded file
    storage_mock.download_file.return_value = b"dummy pdf content"

    response = client.post("/api/v1/vectorize", json={"file_key": "key.pdf"})
    assert response.status_code == 200

    storage_mock.download_file.assert_awaited_with("key.pdf")
    processor_mock.process_pdf.assert_awaited_once()
    vector_mock.vectorize_and_upsert.assert_awaited_once()


def test_vectorize_document_already_processed(client, patch_services):
    storage_mock, vector_mock, processor_mock = patch_services
    processor_mock.process_pdf.return_value = ("mock_hash", [])
    vector_mock.check_document_exists.return_value = ["existing_uuid"]

    storage_mock.download_file.return_value = b"dummy pdf content"

    response = client.post("/api/v1/vectorize", json={"file_key": "processed.pdf"})
    assert response.status_code == 200
    assert "already been processed" in response.json()["message"]
    vector_mock.vectorize_and_upsert.assert_not_awaited()


def test_upload_storage_failure(client, patch_services):
    storage_mock, _, _ = patch_services
    storage_mock.upload_file.side_effect = RuntimeError("MinIO is down")

    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    response = client.post("/api/v1/upload", files=files)

    assert response.status_code == 503
    assert "MinIO is down" in response.json()["detail"]


def test_vectorize_with_source_url_success(client, patch_services):
    storage_mock, _, processor_mock = patch_services
    processor_mock.process_pdf.return_value = (
        "mock_hash",
        [ProcessedContent(
            content_type="text",
            text_content="Sample",
            metadata=DocumentMetadata(
                page=1, section="s1", file_hash="mock_hash"
            )
        )]
    )

    storage_mock.download_file.return_value = b"dummy pdf content"

    response = client.post("/api/v1/vectorize", json={"source_url": "http://example.com/doc.pdf"})
    assert response.status_code == 200
    storage_mock.download_file.assert_awaited_with("http://example.com/doc.pdf")


def test_vectorize_no_source_provided(client):
    response = client.post("/api/v1/vectorize", json={})
    assert response.status_code == 422


def test_vectorize_file_not_found(client, patch_services):
    storage_mock, _, _ = patch_services
    storage_mock.download_file.side_effect = RuntimeError("File not found")

    response = client.post("/api/v1/vectorize", json={"file_key": "not_found.pdf"})
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]


def test_vectorize_storage_download_error(client, patch_services):
    storage_mock, _, _ = patch_services
    storage_mock.download_file.side_effect = RuntimeError("S3 connection failed")

    response = client.post("/api/v1/vectorize", json={"file_key": "any.pdf"})
    assert response.status_code == 503
    assert "S3 connection failed" in response.json()["detail"]


def test_vectorize_pdf_processing_failure(client, patch_services):
    storage_mock, _, processor_mock = patch_services
    storage_mock.download_file.return_value = b"dummy pdf content"
    processor_mock.process_pdf.side_effect = ValueError("Corrupt PDF")

    response = client.post("/api/v1/vectorize", json={"file_key": "corrupt.pdf"})
    assert response.status_code == 422
    assert "Failed to process PDF: Corrupt PDF" in response.json()["detail"]


def test_vectorize_no_content_found(client, patch_services):
    storage_mock, vector_mock, processor_mock = patch_services
    storage_mock.download_file.return_value = b"dummy pdf content"
    processor_mock.process_pdf.return_value = ("mock_hash", [])
    vector_mock.check_document_exists.return_value = []

    response = client.post("/api/v1/vectorize", json={"file_key": "empty.pdf"})
    assert response.status_code == 200

    json_response = response.json()
    assert "no extractable text content was found" in json_response["message"]
    assert json_response["document_ids"] == []

    vector_mock.vectorize_and_upsert.assert_not_awaited()
