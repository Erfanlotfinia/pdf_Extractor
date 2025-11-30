import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import AsyncIterator, Union, BinaryIO

import httpx
import aiofiles
from aiobotocore.session import get_session
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings

logger = logging.getLogger(__name__)

class StorageService(ABC):
    """Abstract base class for a storage service."""

    @abstractmethod
    async def initialize(self):
        """Perform any async setup required."""
        pass

    @abstractmethod
    async def upload_file(self, file_obj: Union[BinaryIO, AsyncIterator[bytes]], object_name: str, file_size: int) -> None:
        """Uploads a file-like object or async iterator."""
        pass

    @abstractmethod
    async def download_file(self, object_name: str) -> bytes:
        """Downloads a file and returns its content as bytes (Use with caution for large files)."""
        pass

    @abstractmethod
    async def download_to_path(self, object_name: str, destination_path: str) -> None:
        """Downloads a file and streams it directly to a local path (Memory efficient)."""
        pass
    
    @abstractmethod
    async def close(self):
        """Perform any async teardown required."""
        pass


class MinioStorageService(StorageService):
    """
    An asynchronous storage service for MinIO/S3.
    Optimized for streaming and resilience.
    """

    def __init__(self):
        self.bucket_name = settings.MINIO_BUCKET
        self.endpoint_url = f"http://{settings.MINIO_HOST}:{settings.MINIO_PORT}"
        self.access_key = settings.MINIO_ROOT_USER
        self.secret_key = settings.MINIO_ROOT_PASSWORD
        
        self._session = get_session()
        self._exit_stack = AsyncExitStack()
        self._s3_client = None

    async def _get_client(self):
        """
        Lazily initializes the S3 client using AsyncExitStack for proper cleanup.
        """
        if not self._s3_client:
            # Create the configuration object directly using botocore.config.Config
            client_config = Config(
                max_pool_connections=20,
                connect_timeout=10,
                read_timeout=60,
                retries={'max_attempts': 3}
            )

            self._s3_client = await self._exit_stack.enter_async_context(
                self._session.create_client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=client_config
                )
            )
        return self._s3_client

    async def initialize(self):
        """Checks for bucket existence and creates it if missing."""
        client = await self._get_client()
        try:
            await client.head_bucket(Bucket=self.bucket_name)
            logger.info("MinIO bucket '%s' exists.", self.bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                logger.info("Bucket '%s' not found. Creating...", self.bucket_name)
                try:
                    await client.create_bucket(Bucket=self.bucket_name)
                    logger.info("Created bucket '%s'.", self.bucket_name)
                except ClientError as ce:
                    logger.error(f"Failed to create bucket: {ce}")
                    raise RuntimeError(f"Bucket creation failed: {ce}")
            else:
                logger.error(f"Failed to connect to MinIO: {e}")
                raise RuntimeError(f"MinIO connection failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((BotoCoreError, ClientError))
    )
    async def upload_file(self, file_obj: Union[BinaryIO, AsyncIterator[bytes]], object_name: str, file_size: int) -> None:
        """
        Uploads a file to MinIO. 
        Handles both standard file objects (spooled) and async iterators.
        """
        client = await self._get_client()
        try:
            # put_object handles file-like objects (read()) automatically
            await client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=file_obj,
                ContentLength=file_size,
                ContentType="application/pdf",
            )
            logger.info(f"Uploaded '{object_name}' ({file_size} bytes).")
        except Exception as e:
            logger.exception(f"Upload failed for '{object_name}': {e}")
            raise RuntimeError(f"Upload failed: {e}")

    async def download_file(self, object_name: str) -> bytes:
        """Downloads a file into memory (bytes)."""
        if object_name.startswith(("http://", "https://")):
            return await self._download_from_url(object_name)
        
        client = await self._get_client()
        try:
            response = await client.get_object(Bucket=self.bucket_name, Key=object_name)
            async with response["Body"] as stream:
                return await stream.read()
        except ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                raise FileNotFoundError(f"File {object_name} not found in storage.")
            raise RuntimeError(f"Download failed: {e}")

    async def download_to_path(self, object_name: str, destination_path: str) -> None:
        """
        Streams a file from S3/URL directly to disk.
        This prevents loading the entire file into RAM.
        """
        if object_name.startswith(("http://", "https://")):
            await self._stream_url_to_file(object_name, destination_path)
            return

        client = await self._get_client()
        try:
            response = await client.get_object(Bucket=self.bucket_name, Key=object_name)
            
            # aiofiles allows non-blocking file writes
            async with aiofiles.open(destination_path, 'wb') as f:
                # Read in chunks (aiobotocore body is an async iterator)
                async for chunk in response["Body"]:
                    await f.write(chunk)
            
            logger.info(f"Streamed '{object_name}' to {destination_path}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                raise FileNotFoundError(f"File {object_name} not found.")
            logger.exception(f"S3 streaming failed: {e}")
            raise RuntimeError(f"S3 streaming failed: {e}")

    async def _download_from_url(self, url: str) -> bytes:
        """Downloads bytes from a URL."""
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
            except httpx.HTTPError as e:
                raise RuntimeError(f"External download failed: {e}")

    async def _stream_url_to_file(self, url: str, destination_path: str):
        """Streams from a URL to a file path."""
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(destination_path, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            await f.write(chunk)
            except httpx.HTTPError as e:
                raise RuntimeError(f"External stream failed: {e}")

    async def close(self):
        """Gracefully closes the client session."""
        await self._exit_stack.aclose()
        self._s3_client = None
        logger.info("Storage service connection closed.")
