from __future__ import annotations

import os
import unittest
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("LOREGUARD_TEST_POSTGRES_URL"),
    "requires an explicitly disposable PostgreSQL/pgvector test database",
)
class PgvectorIntegrationTests(unittest.TestCase):
    def test_real_vector_column_and_cosine_distance(self):
        url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = url
        command.upgrade(config, "head")
        engine = create_engine(url)
        with engine.begin() as connection:
            data_type = connection.exec_driver_sql(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'evidence_embeddings' AND column_name = 'vector'"
            ).scalar_one()
            self.assertEqual(data_type, "vector")
            distance = connection.exec_driver_sql(
                "SELECT '[1,0]'::vector <=> '[0,1]'::vector"
            ).scalar_one()
            self.assertAlmostEqual(float(distance), 1.0)
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
