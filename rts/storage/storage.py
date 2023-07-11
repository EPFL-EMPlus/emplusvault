from rts.db_settings import S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT
import boto3
import os


def get_storage_client():
    return StorageClient()


class StorageClient:
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )

    def upload(self, bucket_name, object_name, file_path):
        normalized_object_name = os.path.normpath(object_name).replace(os.sep, '/')
        self.client.upload_file(file_path, bucket_name, normalized_object_name)

    def upload_binary(self, bucket_name, object_name, binary_data):
        self.client.put_object(
            Bucket=bucket_name, Key=object_name, Body=binary_data)

    def download(self, bucket_name, object_name):
        response = self.client.get_object(
            Bucket=bucket_name, Key=object_name)
        return response['Body'].read()

    def delete(self, bucket_name, object_name):
        self.client.delete_object(Bucket=bucket_name, Key=object_name)
        