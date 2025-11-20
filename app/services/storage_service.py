from abc import ABC, abstractmethod
from typing import IO
import boto3
import os

class StorageService(ABC):
    @abstractmethod
    def upload_file(self, file_obj: IO, object_name: str):
        pass

    @abstractmethod
    def download_file(self, object_name: str) -> IO:
        pass

class MinioStorageService(StorageService):
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{os.getenv('MINIO_HOST')}:{os.getenv('MINIO_PORT')}",
            aws_access_key_id=os.getenv('MINIO_ROOT_USER'),
            aws_secret_access_key=os.getenv('MINIO_ROOT_PASSWORD'),
        )
        self.bucket_name = os.getenv('MINIO_BUCKET')

    def upload_file(self, file_obj: IO, object_name: str):
        self.s3_client.upload_fileobj(file_obj, self.bucket_name, object_name)

    def download_file(self, object_name: str) -> IO:
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_name)
        return response['Body']
    
def get_storage_service() -> StorageService:
    """
    Factory function to get the storage service instance.
    Currently returns a MinioStorageService instance.
    """
    return MinioStorageService()