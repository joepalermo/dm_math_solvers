import os
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    directory = os.path.dirname(destination_file_name)
    if os.path.dirname(destination_file_name) and not os.path.exists(directory):
        os.makedirs(directory)
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}")


if __name__ == '__main__':
    download_blob('jpalermo', 'mathematics_dataset-v1.0.tar.gz', 'mathematics_dataset-v1.0.tar.gz')