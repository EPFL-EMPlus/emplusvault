from rts.db_settings import S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT
from minio import Minio


def get_storage_client():
    return StorageClient()


class StorageClient:
    def __init__(self):
        self.client = Minio(
            S3_ENDPOINT,
            access_key=S3_ACCESS_KEY,
            secret_key=S3_SECRET_KEY,
            secure=False
        )

    def upload(self, bucket_name, object_name, file_path):
        self.client.fput_object(bucket_name, object_name, file_path)

    def download(self, bucket_name, object_name):
        return self.client.get_object(bucket_name, object_name)

    def list_objects(self, bucket_name, prefix=None):
        return self.client.list_objects(bucket_name, prefix=prefix)
