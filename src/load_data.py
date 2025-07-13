"""
module to download data from GCP buckets 
"""

from typing import NoReturn
from google.cloud import storage


def download_blobs(bucket_name: str, prefix: str, local_dir: str) -> NoReturn:
    """
    Download all blobs (files) from a specific prefix in a Google Cloud Storage bucket
    and save them to a local directory.

    Args:
        bucket_name (str): Name of the GCS bucket (e.g., 'my-bucket').
        prefix (str): Folder-like prefix inside the bucket (e.g., 'docs/').
        local_dir (str): Local directory path where the files will be saved.

    Returns:
        None: Files are saved to disk. Prints a message for each downloaded file.
    """
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        blob.download_to_filename(f"{local_dir}/{filename}")
        print(f"Downloaded {filename}")
