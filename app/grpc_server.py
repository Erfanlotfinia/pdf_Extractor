import grpc
import logging
from typing import AsyncIterator

from app.grpc.generated.vectorizer_pb2 import DocumentRequest, ProcessingResponse
from app.grpc.generated import vectorizer_pb2_grpc
from app.vector_db.vector_service import VectorService
from app.processing.pdf_processor import PDFProcessorService
from app.storage.storage_service import StorageService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorizeServiceServicer(vectorizer_pb2_grpc.VectorizeServiceServicer):
    """
    gRPC servicer for the VectorizeService.
    """

    def __init__(
        self,
        storage_service: StorageService,
        pdf_processor_service: PDFProcessorService,
        vector_service: VectorService,
    ):
        self._storage_service = storage_service
        self._pdf_processor_service = pdf_processor_service
        self._vector_service = vector_service
        logger.info("VectorizeServiceServicer initialized with all services.")

    async def ProcessDocument(
        self, request: DocumentRequest, context: grpc.aio.ServicerContext
    ) -> ProcessingResponse:
        """
        Handles the ProcessDocument RPC call.
        """
        source_type = request.WhichOneof("source")
        logger.info(f"[gRPC] Received ProcessDocument request with source type: {source_type}")

        if source_type == "source_url":
            logger.warning("[gRPC] source_url is not implemented.")
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "Processing from source_url is not implemented."
            )

        if not source_type or not request.minio_key:
            logger.error("[gRPC] Request is missing minio_key.")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "minio_key is required.")

        minio_key = request.minio_key

        try:
            # Step 1: Download file from storage
            logger.info(f"[gRPC] Downloading file from MinIO with key: {minio_key}")
            temp_file_path = await self._storage_service.get_file(minio_key)
            if not temp_file_path:
                 logger.error(f"[gRPC] File not found in MinIO for key: {minio_key}")
                 await context.abort(grpc.StatusCode.NOT_FOUND, f"File not found for key: {minio_key}")


            # Step 2: Process the PDF to extract chunks
            logger.info(f"[gRPC] Processing PDF: {temp_file_path}")
            chunks = await self._pdf_processor_service.process_pdf(temp_file_path)
            if not chunks:
                 logger.error(f"[gRPC] PDF processing failed for key: {minio_key}. The document might be corrupt or empty.")
                 await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Failed to process PDF. It may be empty or corrupt.")

            # Step 3: Vectorize and store the chunks
            logger.info(f"[gRPC] Vectorizing {len(chunks)} chunks for document key: {minio_key}")
            document_id, num_chunks = await self._vector_service.vectorize_and_store(chunks, minio_key)
            logger.info(f"[gRPC] Successfully vectorized and stored document '{document_id}' with {num_chunks} chunks.")

            # Step 4: Clean up the temporary file
            await self._storage_service.delete_temp_file(temp_file_path)

            return ProcessingResponse(
                success=True,
                document_id=document_id,
                message=f"Successfully processed and vectorized document '{minio_key}'.",
                chunk_count=num_chunks,
            )

        except Exception as e:
            logger.exception(f"[gRPC] An internal error occurred while processing key '{minio_key}': {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"An unexpected error occurred: {e}")
