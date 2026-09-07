import os
import unittest
from unittest.mock import patch

from api.chat_repository import get_database_url


class DatabaseUrlTests(unittest.TestCase):
    def test_canonical_database_url_has_priority(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://canonical",
                "INTERNAL_DB_URL": "postgresql://internal",
                "EXTERNAL_DB_URL": "postgresql://external",
            },
            clear=True,
        ):
            self.assertEqual(get_database_url(), "postgresql://canonical")

    def test_render_prefers_internal_url(self):
        with patch.dict(
            os.environ,
            {
                "RENDER": "true",
                "INTERNAL_DB_URL": "postgresql://internal",
                "EXTERNAL_DB_URL": "postgresql://external",
            },
            clear=True,
        ):
            self.assertEqual(get_database_url(), "postgresql://internal")

    def test_local_development_prefers_external_url(self):
        with patch.dict(
            os.environ,
            {
                "INTERNAL_DB_URL": "postgresql://internal",
                "EXTERNAL_DB": "postgresql://legacy-external",
            },
            clear=True,
        ):
            self.assertEqual(get_database_url(), "postgresql://legacy-external")

    def test_missing_url_fails_without_disclosing_secrets(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "connection URL is required"):
                get_database_url()


if __name__ == "__main__":
    unittest.main()
