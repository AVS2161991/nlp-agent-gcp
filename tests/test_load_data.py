"""
tests for load_data.py module
"""

from unittest.mock import patch, MagicMock
from src.load_data import download_blobs


@patch("src.load_data.storage.Client")
def test_download_blobs(mock_storage_client_class, tmp_path):
    mock_client = MagicMock()
    mock_storage_client_class.return_value = mock_client

    mock_blob1 = MagicMock()
    mock_blob1.name = "prefix/file1.txt"
    mock_blob2 = MagicMock()
    mock_blob2.name = "prefix/dir/file2.csv"

    mock_client.list_blobs.return_value = [mock_blob1, mock_blob2]

    download_blobs("test-bucket", "prefix", str(tmp_path))

    expected_path1 = tmp_path / "file1.txt"
    expected_path2 = tmp_path / "file2.csv"

    mock_blob1.download_to_filename.assert_called_once_with(str(expected_path1))
    mock_blob2.download_to_filename.assert_called_once_with(str(expected_path2))

    mock_client.list_blobs.assert_called_once_with("test-bucket", prefix="prefix")
