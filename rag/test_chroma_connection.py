import os
import unittest
from unittest.mock import patch

from rag.chroma_connection import create_chroma_client, get_collection_name


class ChromaConnectionTests(unittest.TestCase):
    def test_cloud_configuration_is_required(self):
        empty_cloud_environment = {
            "CHROMA_API_KEY": "",
            "CHROMA_TENANT": "",
            "CHROMA_DATABASE": "",
        }

        with patch.dict(os.environ, empty_cloud_environment):
            with self.assertRaisesRegex(RuntimeError, "configuration is required"):
                create_chroma_client()

    def test_creates_cloud_client_with_required_configuration(self):
        cloud_environment = {
            "CHROMA_API_KEY": "test-key",
            "CHROMA_TENANT": "test-tenant",
            "CHROMA_DATABASE": "test-database",
            "CHROMA_HOST": "",
        }

        with patch.dict(os.environ, cloud_environment):
            with patch("rag.chroma_connection.chromadb.CloudClient") as cloud_client:
                create_chroma_client()

        cloud_client.assert_called_once_with(
            tenant="test-tenant",
            database="test-database",
            api_key="test-key",
        )

    def test_collection_name_has_cloud_default(self):
        with patch.dict(os.environ, {"CHROMA_COLLECTION": ""}):
            self.assertEqual(get_collection_name(), "constitution_english")


if __name__ == "__main__":
    unittest.main()
