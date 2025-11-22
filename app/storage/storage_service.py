import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)

class StorageService(ABC):
    """Abstract base class for a storage service."""

    @abstractmethod
    async def initialize(self):
        """Perform any async setup required, like ensuring a bucket exists."""
        pass

    @abstractmethod
    async def upload_file(self, file_iterator: AsyncIterator[bytes], object_name: str, file_size: int) -> None:
        """Uploads a file from an async iterator."""
        pass

    @abstractmethod
    async def download_file(self, object_name: str) -> bytes:
        """Downloads a file and returns its content as bytes."""
        pass
    
    @abstractmethod
    async def close(self):
        """Perform any async teardown required."""
        pass

class MinioStorageService(StorageService):
    """An asynchronous storage service for MinIO."""

    def __init__(self):
        self.bucket_name = settings.MINIO_BUCKET
        self.endpoint_url = f"http://{settings.MINIO_HOST}:{settings.MINIO_PORT}"
        self.access_key = settings.MINIO_ROOT_USER
        self.secret_key = settings.MINIO_ROOT_PASSWORD
        self._s3_client = None
        self._session = get_session()

    async def _get_client(self):
        """Lazily creates and returns the S3 client."""
        if not self._s3_client:
            # Note: __aenter__ is used to correctly initialize the client context
            self._s3_client = await self._session.create_client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            ).__aenter__()
        return self._s3_client

    async def initialize(self):
        """Initializes the S3 client and ensures the bucket exists."""
        client = await self._get_client()
        try:
            await client.head_bucket(Bucket=self.bucket_name)
            logger.info("MinIO bucket '%s' already exists.", self.bucket_name)
        except ClientError as e:
            # If the bucket does not exist, a 404 error is returned
            if e.response["Error"]["Code"] == "404":
                logger.info("MinIO bucket '%s' not found. Creating it.", self.bucket_name)
                try:
                    await client.create_bucket(Bucket=self.bucket_name)
                    logger.info("Successfully created bucket '%s'.", self.bucket_name)
                except ClientError as ce:
                    logger.exception("Failed to create MinIO bucket '%s': %s", self.bucket_name, ce)
                    raise RuntimeError(f"Could not create MinIO bucket: {ce}") from ce
            else:
                # Handle other potential errors like connection issues
                logger.exception("Failed to check for MinIO bucket '%s': %s", self.bucket_name, e)
                raise RuntimeError(f"Could not connect to MinIO: {e}") from e

    async def upload_file(self, file_iterator: AsyncIterator[bytes], object_name: str, file_size: int) -> None:
        """Uploads a file to MinIO by streaming from an async iterator."""
        client = await self._get_client()
        try:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=file_iterator,
                ContentLength=file_size,
                ContentType="application/pdf",
            )
            logger.info("Successfully uploaded '%s' to MinIO bucket '%s'.", object_name, self.bucket_name)
        except ClientError as e:
            logger.exception("MinIO upload failed for object '%s': %s", object_name, e)
            raise RuntimeError(f"Storage service upload error: {e}") from e

    async def download_file(self, object_name: str) -> bytes:
        """
        Downloads a file from MinIO or a public URL.
        Returns the file content as bytes.
        """
        if object_name.startswith(("http://", "https://")):
            return await self._download_from_url(object_name)
        return await self._download_from_minio(object_name)

    async def _download_from_url(self, url: str) -> bytes:
        """Downloads a file from a URL using an async HTTP client."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                logger.info("Successfully downloaded file from URL: %s", url)
                return response.content
            except httpx.HTTPError as e:
                logger.exception("Failed to download file from URL '%s': %s", url, e)
                raise RuntimeError(f"Failed to download from URL: {e}") from e

    async def _download_from_minio(self, object_name: str) -> bytes:
        """Downloads a file from the MinIO bucket."""
        client = await self._get_client()
        try:
            response = await client.get_object(Bucket=self.bucket_name, Key=object_name)
            # The body is a stream that needs to be read
            content = await response["Body"].read()
            logger.info("Successfully downloaded '%s' from MinIO.", object_name)
            return content
        except ClientError as e:
            logger.exception("Failed to download '%s' from MinIO: %s", object_name, e)
            raise RuntimeError(f"Storage service download error: {e}") from e

    async def close(self):
        """Closes the S3 client connection."""
        if self._s3_client:
            # __aexit__ is used to correctly clean up the client context
            await self._s3_client.__aexit__(None, None, None)
            self._s3_client = None
            logger.info("MinIO client connection closed.")
