import os
import boto3
import threading
import emv.utils

from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from botocore.exceptions import ClientError, BotoCoreError
from botocore.response import StreamingBody
from emv.settings import S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT


LOG = emv.utils.get_logger()


def get_storage_client():
    return StorageClient()

# class ProgressPercentage(object):
#     def __init__(self, filename):
#         self._filename = filename
#         self._size = float(os.path.getsize(filename))
#         self._seen_so_far = 0
#         self._lock = threading.Lock()

#     def __call__(self, bytes_amount):
#         # To simplify, assume this is hooked up to a single filename
#         with self._lock:
#             self._seen_so_far += bytes_amount
#             percentage = (self._seen_so_far / self._size) * 100
#             LOG.debug(
#                 "\r%s  %s / %s  (%.2f%%)" % (
#                     self._filename, self._seen_so_far, self._size,
#                     percentage))
#             # sys.stdout.flush()


class StorageClient:
    def __init__(self):
        self.client = self._initialize_s3_client()

    def _initialize_s3_client(self):
        try:
            client = boto3.client(
                's3',
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
            )
        except (BotoCoreError, ClientError) as e:
            LOG.error(f"Failed to initialize S3 client: {e}")
            raise e
        return client

    def upload(self, bucket_name: str, object_name: str, file_data: Any) -> bool:
        """
        Uploads a file to the specified S3 bucket.

        The file_data parameter can be one of three types:
        - str or Path: interpreted as the file path to an existing file, which will be opened and read.
        - bytes: raw binary data, which will be converted to a BytesIO object and uploaded.
        - file-like object: an object that supports the read method, which will be uploaded as is.

        :param bucket_name: The name of the S3 bucket to upload to.
        :param object_name: The key name to use for the uploaded object in the S3 bucket.
        :param file_data: The file data to upload. Can be a file path (str or Path), raw binary data (bytes), or a file-like object.
        :return: True if the upload was successful, False otherwise.
        """
        normalized_object_name = os.path.normpath(
            object_name).replace(os.sep, '/')
        ok = True
        try:
            if isinstance(file_data, (str, Path)):
                # If file_data is a file path, open the file in binary mode
                with open(file_data, 'rb') as f:
                    self.client.upload_fileobj(
                        f, bucket_name, normalized_object_name)
            elif isinstance(file_data, bytes):
                # If file_data is raw binary data, convert to a BytesIO object
                with BytesIO(file_data) as f:
                    self.client.upload_fileobj(
                        f, bucket_name, normalized_object_name)
            else:
                # If file_data is a file-like object, upload it directly
                self.client.upload_fileobj(
                    file_data, bucket_name, normalized_object_name)
            LOG.debug(
                f"Upload successful: {bucket_name}/{normalized_object_name}")
        except (BotoCoreError, ClientError) as e:
            LOG.error(
                f"Failed to upload: {bucket_name}/{normalized_object_name} due to {e}")
            ok = False
        return ok

    def download(self, bucket_name: str, object_name: str, file_or_buf: Any) -> bool:
        """
        Downloads an object from the specified S3 bucket.

        The file_or_buf parameter can be one of two types:
        - str or Path: interpreted as the file path to a file, where the downloaded data will be written.
        - file-like object: an object that supports the write method, where the downloaded data will be written.

        :param bucket_name: The name of the S3 bucket to download from.
        :param object_name: The key name of the object to download from the S3 bucket.
        :param file_or_buf: Where to write the downloaded data. Can be a file path (str or Path) or a file-like object.
        :return: True if the download was successful, False otherwise.
        """
        ok = True
        try:
            if isinstance(file_or_buf, (str, Path)):
                # If file_or_buf is a file path, open the file in binary mode for writing
                with open(file_or_buf, 'wb') as f:
                    self.client.download_fileobj(bucket_name, object_name, f)
            else:
                # If file_or_buf is a file-like object, write to it directly
                self.client.download_fileobj(
                    bucket_name, object_name, file_or_buf)
            LOG.debug(f"Download successful: {bucket_name}/{object_name}")
        except (BotoCoreError, ClientError, IOError) as e:
            LOG.error(
                f"Failed to download: {bucket_name}/{object_name} due to {e}")
            ok = False
        return ok

    def get_stream(self, bucket_name: str, object_name: str, start: Optional[int] = None, end: Optional[int] = None) -> Optional[StreamingBody]:
        """
        Gets a streaming body for an object from the specified S3 bucket.

        :param bucket_name: The name of the S3 bucket to get from.
        :param object_name: The key name of the object to get from the S3 bucket.
        :param start: Optional; The start byte of the range to download. If not specified, the download starts from the beginning of the object.
        :param end: Optional; The end byte of the range to download. If not specified, the download goes until the end of the object.
        :return: A StreamingBody object for the object data, or None if the get fails. Compatible with FastAPI's StreamingResponse.
        """
        try:
            range_header = None
            if start is not None or end is not None:
                range_header = f"bytes={start or '0'}-{end or ''}"
            response = self.client.get_object(
                Bucket=bucket_name, Key=object_name, Range=range_header)
            LOG.debug(f"Got streaming body: {bucket_name}/{object_name}")
            return response['Body']
        except (BotoCoreError, ClientError, IOError) as e:
            LOG.error(f"Failed to get: {bucket_name}/{object_name} due to {e}")
            return None

    def get_bytes(self, bucket_name: str, object_name: str) -> Optional[bytes]:
        """
        Downloads an object from the specified S3 bucket into memory and returns the data as bytes.

        :param bucket_name: The name of the S3 bucket to download from.
        :param object_name: The key name of the object to download from the S3 bucket.
        :return: The downloaded data as bytes, or None if the download fails.
        """
        try:
            response = self.client.get_object(
                Bucket=bucket_name, Key=object_name)
            LOG.debug(f"Download successful: {bucket_name}/{object_name}")
            return response['Body'].read()
        except (BotoCoreError, ClientError, IOError) as e:
            LOG.error(
                f"Failed to download: {bucket_name}/{object_name} due to {e}")
            return None

    def delete(self, bucket_name: str, object_name: str) -> bool:
        ok = False
        try:
            ok = self.client.delete_object(Bucket=bucket_name, Key=object_name)
            LOG.debug(f"Deleted: {bucket_name}/{object_name}")
        except (BotoCoreError, ClientError) as e:
            LOG.error(
                f"Failed to delete: {bucket_name}/{object_name} due to {e}")
        return ok

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """
        Checks if an object exists in the specified S3 bucket.

        :param bucket_name: The name of the S3 bucket.
        :param object_name: The key name of the object in the S3 bucket.
        :return: True if the object exists, False otherwise.
        """
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError as e:
            # If the error code is 404 (Not Found), then the object does not exist.
            if e.response['Error']['Code'] == '404':
                return False
            else:
                # If the error code is anything else, then something else has gone wrong.
                raise
