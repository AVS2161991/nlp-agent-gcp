"""
module to download data from GCP buckets 
"""

from google.cloud import storage


def download_blobs(bucket_name, prefix, local_dir):
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        blob.download_to_filename(f"{local_dir}/{filename}")
        print(f"Downloaded {filename}")
