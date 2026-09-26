"""Isolated SQLite coverage for the project-name sort-key migration."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, Table, create_engine, inspect, text

from app.project_sort import project_name_sort_key_v1


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "0018_character_axis_direction"
HEAD = "0019_project_name_sort_key"
INDEX = "ix_projects_workspace_name_sort_id"


def _config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    return config


class ProjectNameSortMigrationTests(unittest.TestCase):
    def test_backfills_old_projects_as_blob_and_creates_composite_index(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{(Path(directory) / 'projects.db').as_posix()}"
            config = _config(url)
            command.upgrade(config, PREVIOUS)

            engine = create_engine(url)
            projects = Table("projects", MetaData(), autoload_with=engine)
            names = ("张三", "Ａlpha", "安然", "alice")
            with engine.begin() as connection:
                workspace_id = connection.execute(
                    text("SELECT id FROM workspaces ORDER BY id LIMIT 1")
                ).scalar_one()
                connection.execute(
                    projects.insert(),
                    [
                        {
                            "id": f"project-{number:04d}",
                            "workspace_id": workspace_id,
                            "name": names[number % len(names)],
                            "description": "preserve this",
                            "created_at": datetime(2026, 9, 1),
                        }
                        for number in range(503)
                    ],
                )
            engine.dispose()

            command.upgrade(config, HEAD)
            engine = create_engine(url)
            try:
                with engine.connect() as connection:
                    rows = connection.execute(text(
                        "SELECT id, name, description, name_sort_key, "
                        "typeof(name_sort_key) AS storage_type FROM projects "
                        "ORDER BY id"
                    )).all()
                    revision = connection.execute(text(
                        "SELECT version_num FROM alembic_version"
                    )).scalar_one()
                    ordered_ids = connection.execute(text(
                        "SELECT id FROM projects WHERE workspace_id = :workspace_id "
                        "ORDER BY name_sort_key, id"
                    ), {"workspace_id": workspace_id}).scalars().all()

                self.assertEqual(revision, HEAD)
                self.assertEqual(len(rows), 503)
                self.assertTrue(all(row.description == "preserve this" for row in rows))
                self.assertTrue(all(row.storage_type == "blob" for row in rows))
                self.assertTrue(all(
                    row.name_sort_key == project_name_sort_key_v1(row.name)
                    for row in rows
                ))
                self.assertEqual(
                    ordered_ids,
                    [row.id for row in sorted(
                        rows, key=lambda row: (row.name_sort_key, row.id)
                    )],
                )
                indexes = {
                    index["name"]: index["column_names"]
                    for index in inspect(engine).get_indexes("projects")
                }
                self.assertEqual(
                    indexes[INDEX], ["workspace_id", "name_sort_key", "id"]
                )
            finally:
                engine.dispose()

            command.downgrade(config, PREVIOUS)
            engine = create_engine(url)
            try:
                inspector = inspect(engine)
                self.assertNotIn(
                    "name_sort_key",
                    {column["name"] for column in inspector.get_columns("projects")},
                )
                self.assertNotIn(
                    INDEX, {index["name"] for index in inspector.get_indexes("projects")}
                )
                with engine.connect() as connection:
                    self.assertEqual(connection.execute(text(
                        "SELECT count(*) FROM projects WHERE description = 'preserve this'"
                    )).scalar_one(), 503)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
