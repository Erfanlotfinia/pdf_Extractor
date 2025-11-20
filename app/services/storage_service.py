import io
import logging
from abc import ABC, abstractmethod
import sys
from typing import IO
import os
import httpx
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
class StorageService(ABC):
    @abstractmethod
    def upload_file(self, file_obj: IO, object_name: str):
        pass

    @abstractmethod
    def download_file(self, object_name: str) -> IO:
        pass


class MinioStorageService(StorageService):
    def __init__(self):
        self.bucket_name = os.getenv("MINIO_BUCKET")
        host = os.getenv("MINIO_HOST", "localhost")
        port = os.getenv("MINIO_PORT", "9000")

        endpoint = f"{host}:{port}"
        access_key = os.getenv("MINIO_ROOT_USER")
        secret_key = os.getenv("MINIO_ROOT_PASSWORD")

        try:
            if not (host and access_key and secret_key and self.bucket_name):
                for i in (host, access_key, secret_key, self.bucket_name):
                    if not i:
                        logging.error("MinIO configuration missing: %s", i)
                raise RuntimeError("Missing MinIO configuration.")

            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False
            )

            # Create bucket if not exists
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)

        except Exception as e:
            logging.exception("Failed to initialize MinIO client: %s", e)
            self.client = None


    def upload_file(self, file_obj: IO, object_name: str):
        if not self.client:
            raise RuntimeError("MinIO client not initialized.")

        try:
            # Read all bytes from file-like object
            data = file_obj.read()
            file_obj.seek(0)

            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(data),
                length=len(data),
                content_type="application/octet-stream",
            )

        except S3Error as e:
            logging.exception("MinIO upload failed: %s", e)
            raise
        except Exception as e:
            logging.exception("Unexpected upload error: %s", e)
            raise


    def download_file(self, object_name: str) -> IO:
        # If object_name is URL → download via HTTP
        if object_name.startswith("http://") or object_name.startswith("https://"):
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(object_name)
                    resp.raise_for_status()
                    buf = io.BytesIO(resp.content)
                    buf.seek(0)
                    return buf
            except Exception as e:
                logging.exception("HTTP download error: %s", e)
                raise RuntimeError(f"Failed downloading URL: {e}")

        # Download from MinIO
        if not self.client:
            raise RuntimeError("MinIO client not initialized.")

        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()

            buf = io.BytesIO(data)
            buf.seek(0)
            return buf

        except S3Error as e:
            logging.exception("MinIO download failed: %s", e)
            raise
        except Exception as e:
            logging.exception("Unexpected download error: %s", e)
            raise RuntimeError(f"Error downloading object: {e}")


def get_storage_service() -> StorageService:
    """Factory: return MinIO storage service instance."""
    return MinioStorageService()
