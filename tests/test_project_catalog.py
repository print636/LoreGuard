from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.dialects import postgresql

from app.auth import AuthContext, get_auth_context
from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRunRow,
    DocumentRow,
    LOCAL_USER_ID,
    ProjectRow,
    SessionLocal,
    WorkspaceRow,
    engine,
)
from app.main import _project_catalog_diagnostic_projection, app
from app.project_sort import project_name_sort_key


@contextmanager
def catalog_workspace(workspace_id: str, *, anonymous: bool = False):
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        user_id=LOCAL_USER_ID, workspace_id=workspace_id, role="owner", anonymous=anonymous
    )
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_auth_context, None)


class ProjectCatalogTests(unittest.TestCase):
    def test_all_api_project_creation_paths_write_sort_key(self):
        workspace_id = str(uuid4())
        with TestClient(app) as client:
            with SessionLocal() as db:
                db.add(WorkspaceRow(id=workspace_id, name="project creation"))
                db.commit()

            with catalog_workspace(workspace_id, anonymous=True):
                created = [
                    client.post("/api/v1/projects", json={"name": "李四"}),
                    client.post("/api/v1/demo"),
                    client.post("/api/v1/demo/advanced"),
                ]
            for response in created:
                self.assertEqual(201, response.status_code, response.text)
            with SessionLocal() as db:
                for response in created:
                    project = db.get(ProjectRow, response.json()["id"])
                    self.assertEqual(project_name_sort_key(project.name),
                                     project.name_sort_key)

    def test_postgresql_diagnostic_projection_compiles_to_json_paths(self):
        statement = _project_catalog_diagnostic_projection(
            [str(uuid4())], "postgresql"
        )
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertEqual(5, sql.count(" #> "))
        self.assertNotIn("json_type", sql)
        self.assertNotIn("analysis_diagnostics.payload AS", sql)

    def test_name_sort_uses_pinyin_across_pages_and_stable_id_ties(self):
        workspace_id = str(uuid4())
        rows = [
            ("10000000-0000-0000-0000-000000000101", "张三"),
            ("20000000-0000-0000-0000-000000000102", "常安"),
            ("30000000-0000-0000-0000-000000000103", "长安"),
            ("40000000-0000-0000-0000-000000000104", "李四"),
            ("50000000-0000-0000-0000-000000000105", "ＡLICE"),
            ("60000000-0000-0000-0000-000000000106", "Alice"),
        ]
        self.assertLess(project_name_sort_key("李四"), project_name_sort_key("张三"))
        self.assertEqual(project_name_sort_key("长安"), project_name_sort_key("常安"))
        self.assertEqual(project_name_sort_key("ＡLICE"), project_name_sort_key("Alice"))
        with TestClient(app) as client:
            with SessionLocal() as db:
                db.add(WorkspaceRow(id=workspace_id, name="pinyin sort"))
                db.flush()
                db.add_all(ProjectRow(id=project_id, workspace_id=workspace_id,
                                      name=name, description="")
                           for project_id, name in rows)
                db.commit()

            with catalog_workspace(workspace_id):
                ordered_ids = []
                statements = []

                def record_sql(_connection, _cursor, statement, parameters,
                               _context, _executemany):
                    if statement.lstrip().lower().startswith("select"):
                        statements.append((statement, parameters))

                event.listen(engine, "before_cursor_execute", record_sql)
                try:
                    for page in range(1, 4):
                        response = client.get("/api/v1/project-catalog", params={
                            "sort": "name", "page": page, "page_size": 2,
                        })
                        self.assertEqual(200, response.status_code, response.text)
                        self.assertEqual(6, response.json()["total"])
                        ordered_ids.extend(item["id"] for item in response.json()["items"])
                finally:
                    event.remove(engine, "before_cursor_execute", record_sql)
                self.assertEqual([
                    rows[4][0], rows[5][0], rows[1][0], rows[2][0],
                    rows[3][0], rows[0][0],
                ], ordered_ids)
                page_sql, page_params = next(
                    (sql, params) for sql, params in statements
                    if "ORDER BY projects.name_sort_key" in sql
                )
                with engine.connect() as connection:
                    plan = connection.exec_driver_sql(
                        "EXPLAIN QUERY PLAN " + page_sql, page_params
                    ).all()
                self.assertIn(
                    "ix_projects_workspace_name_sort_id",
                    " ".join(str(row[3]) for row in plan),
                )

    def test_cards_search_sort_pagination_and_safe_latest_run(self):
        timestamp = datetime(2026, 1, 1, 12, 0, 0)
        workspace_id = str(uuid4())
        other_workspace_id = str(uuid4())
        first_id = "10000000-0000-0000-0000-000000000001"
        second_id = "20000000-0000-0000-0000-000000000002"
        third_id = "30000000-0000-0000-0000-000000000003"
        other_id = str(uuid4())
        with TestClient(app) as client:
            with SessionLocal() as db:
                db.add_all([
                    WorkspaceRow(id=workspace_id, name="catalog owner"),
                    WorkspaceRow(id=other_workspace_id, name="catalog other"),
                ])
                db.flush()
                db.add_all([
                    ProjectRow(id=first_id, workspace_id=workspace_id, name="Zeta",
                               description="Contains 10% extra", created_at=timestamp),
                    ProjectRow(id=second_id, workspace_id=workspace_id, name="alpha",
                               description="Second", created_at=timestamp),
                    ProjectRow(id=third_id, workspace_id=workspace_id, name="Alpha",
                               description="Third", created_at=timestamp),
                    ProjectRow(id=other_id, workspace_id=other_workspace_id,
                               name="private project", description="Private", created_at=timestamp),
                ])
                db.flush()
                db.add_all([
                    DocumentRow(project_id=third_id, name="active-a", content="a", active=True),
                    DocumentRow(project_id=third_id, name="active-b", content="b", active=True),
                    DocumentRow(project_id=third_id, name="old", content="old", active=False),
                ])
                old_run_id = "10000000-0000-0000-0000-000000000010"
                latest_run_id = "20000000-0000-0000-0000-000000000020"
                db.add_all([
                    AnalysisRunRow(id=old_run_id, project_id=third_id,
                                   status="completed", created_at=timestamp),
                    AnalysisRunRow(id=latest_run_id, project_id=third_id,
                                   status="failed", created_at=timestamp,
                                   provider_identity={"source": "account_byok",
                                                      "configured": True}),
                ])
                db.flush()
                db.add(AnalysisDiagnosticRow(run_id=latest_run_id, payload={
                    "model": {"provider_calls": [{"status": "failure",
                                                 "raw_response": "PRIVATE PROVIDER DATA"}]},
                    "raw_prompt": "PRIVATE PROMPT DATA",
                }))
                db.commit()

            with catalog_workspace(workspace_id):
                response = client.get("/api/v1/project-catalog", params={"page_size": 2})
                self.assertEqual(200, response.status_code, response.text)
                body = response.json()
                self.assertEqual((1, 2, 3),
                                 (body["page"], body["page_size"], body["total"]))
                self.assertEqual([third_id, second_id],
                                 [item["id"] for item in body["items"]])
                card = body["items"][0]
                self.assertEqual({"id", "name", "description", "created_at",
                                  "active_document_count", "latest_run"}, set(card))
                self.assertEqual(2, card["active_document_count"])
                self.assertEqual({"id", "status", "created_at", "model_execution"},
                                 set(card["latest_run"]))
                self.assertEqual(latest_run_id, card["latest_run"]["id"])
                self.assertEqual("provider_failed",
                                 card["latest_run"]["model_execution"]["status"])
                self.assertNotIn("PRIVATE", json.dumps(body))

                second_page = client.get("/api/v1/project-catalog",
                                         params={"page": 2, "page_size": 2}).json()
                self.assertEqual([first_id], [item["id"] for item in second_page["items"]])
                self.assertIsNone(second_page["items"][0]["latest_run"])
                names = client.get("/api/v1/project-catalog",
                                   params={"sort": "name"}).json()
                self.assertEqual([second_id, third_id, first_id],
                                 [item["id"] for item in names["items"]])
                searched = client.get("/api/v1/project-catalog",
                                      params={"query": " alpha "}).json()
                self.assertEqual([third_id, second_id],
                                 [item["id"] for item in searched["items"]])
                percent = client.get("/api/v1/project-catalog",
                                     params={"query": "%"}).json()
                self.assertEqual([first_id], [item["id"] for item in percent["items"]])
                selected = client.get("/api/v1/project-catalog",
                                      params={"project_id": first_id}).json()
                self.assertEqual(1, selected["total"])
                self.assertEqual([first_id], [item["id"] for item in selected["items"]])
                excluded = client.get("/api/v1/project-catalog",
                                      params={"project_id": first_id, "query": "alpha"}).json()
                self.assertEqual({"page": 1, "page_size": 40, "total": 0, "items": []},
                                 excluded)
                self.assertEqual([], client.get("/api/v1/project-catalog",
                                                params={"page": 3, "page_size": 2})
                                 .json()["items"])
                # The older full-detail route remains available to legacy consumers.
                legacy = client.get("/api/v1/projects")
                self.assertEqual(200, legacy.status_code, legacy.text)
                self.assertEqual(3, len(legacy.json()))
                legacy_card = next(item for item in legacy.json() if item["id"] == third_id)
                self.assertIn("input_documents", legacy_card["latest_run"])

            with catalog_workspace(other_workspace_id):
                response = client.get("/api/v1/project-catalog")
                self.assertEqual([other_id],
                                 [item["id"] for item in response.json()["items"]])
                self.assertEqual(0, client.get("/api/v1/project-catalog",
                                               params={"project_id": third_id})
                                 .json()["total"])

    def test_validation_bounds(self):
        with TestClient(app) as client:
            invalid = (
                {"page": 0}, {"page": 100001}, {"page": "oops"},
                {"page_size": 0}, {"page_size": 101}, {"sort": "unknown"},
                {"query": "a" * 161}, {"project_id": "a" * 37},
            )
            for params in invalid:
                with self.subTest(params=params):
                    response = client.get("/api/v1/project-catalog", params=params)
                    self.assertEqual(422, response.status_code, response.text)

    def test_diagnostics_are_skipped_when_run_status_is_already_known(self):
        workspace_id = str(uuid4())
        with TestClient(app) as client:
            with SessionLocal() as db:
                db.add(WorkspaceRow(id=workspace_id, name="known execution"))
                db.flush()
                used_project = ProjectRow(workspace_id=workspace_id, name="used")
                planned_project = ProjectRow(workspace_id=workspace_id, name="planned")
                db.add_all([used_project, planned_project])
                db.flush()
                used_run = AnalysisRunRow(project_id=used_project.id, status="completed",
                                          prompt_tokens=2)
                planned_run = AnalysisRunRow(project_id=planned_project.id, status="queued")
                db.add_all([used_run, planned_run])
                db.flush()
                db.add_all([
                    AnalysisDiagnosticRow(run_id=used_run.id,
                                          payload={"raw_prompt": "PRIVATE DATA"}),
                    AnalysisDiagnosticRow(run_id=planned_run.id,
                                          payload={"raw_prompt": "PRIVATE DATA"}),
                ])
                db.commit()

            statements = []

            def record_sql(_connection, _cursor, statement, _parameters, _context, _executemany):
                if statement.lstrip().lower().startswith("select"):
                    statements.append(statement)

            with catalog_workspace(workspace_id):
                event.listen(engine, "before_cursor_execute", record_sql)
                try:
                    response = client.get("/api/v1/project-catalog")
                finally:
                    event.remove(engine, "before_cursor_execute", record_sql)

            self.assertEqual(200, response.status_code, response.text)
            cards = {item["id"]: item for item in response.json()["items"]}
            self.assertEqual("used", cards[used_project.id]["latest_run"]
                             ["model_execution"]["status"])
            self.assertEqual("planned", cards[planned_project.id]["latest_run"]
                             ["model_execution"]["status"])
            self.assertEqual(4, len(statements))
            self.assertFalse(any("FROM analysis_diagnostics" in sql for sql in statements))
            self.assertNotIn("PRIVATE DATA", response.text)

    def test_diagnostic_projection_preserves_status_without_loading_story(self):
        workspace_id = str(uuid4())
        cases = [
            ("failed call", {"model": {"provider_calls": [
                {"status": "failure", "raw_response": "PRIVATE RESPONSE"}
            ]}}, "provider_failed"),
            ("successful counter", {"model": {"attempted_chunks": 1,
                                                "succeeded_chunks": 1,
                                                "failed_chunks": 0}}, "used"),
            ("successful usage", {"usage_accounting": {"provider_calls": [
                {"status": "success"}
            ]}}, "used"),
            ("boolean counters", {"model": {"attempted_chunks": True,
                                             "succeeded_chunks": False,
                                             "failed_chunks": True}}, "not_used"),
        ]
        with TestClient(app) as client:
            expected = {}
            with SessionLocal() as db:
                db.add(WorkspaceRow(id=workspace_id, name="diagnostic projection"))
                db.flush()
                for name, diagnostic, status in cases:
                    project = ProjectRow(workspace_id=workspace_id, name=name)
                    db.add(project)
                    db.flush()
                    run = AnalysisRunRow(project_id=project.id, status="completed")
                    db.add(run)
                    db.flush()
                    db.add(AnalysisDiagnosticRow(run_id=run.id, payload={
                        **diagnostic,
                        "story_body": "PRIVATE STORY BODY" * 10_000,
                        "raw_prompt": "PRIVATE PROMPT",
                    }))
                    expected[project.id] = status
                db.commit()

            statements = []

            def record_sql(_connection, _cursor, statement, _parameters, _context, _executemany):
                if statement.lstrip().lower().startswith("select"):
                    statements.append(statement)

            with catalog_workspace(workspace_id):
                event.listen(engine, "before_cursor_execute", record_sql)
                try:
                    response = client.get("/api/v1/project-catalog")
                finally:
                    event.remove(engine, "before_cursor_execute", record_sql)

            self.assertEqual(200, response.status_code, response.text)
            actual = {item["id"]: item["latest_run"]["model_execution"]["status"]
                      for item in response.json()["items"]}
            self.assertEqual(expected, actual)
            self.assertEqual(5, len(statements))
            diagnostic_sql = next(sql for sql in statements
                                  if "FROM analysis_diagnostics" in sql)
            self.assertIn("JSON_EXTRACT", diagnostic_sql.upper())
            self.assertNotIn("SELECT analysis_diagnostics.run_id, analysis_diagnostics.payload",
                             diagnostic_sql)
            self.assertNotIn("PRIVATE", response.text)

    def test_sql_query_count_stays_constant_for_large_workspace(self):
        workspace_id = str(uuid4())
        timestamp = datetime(2026, 2, 1)
        with TestClient(app) as client:
            with SessionLocal() as db:
                db.add(WorkspaceRow(id=workspace_id, name="large catalog"))
                db.flush()
                projects = [ProjectRow(workspace_id=workspace_id,
                                       name=f"Project {index:03d}",
                                       created_at=(timestamp + timedelta(days=1)
                                                   if index == 0 else timestamp))
                            for index in range(240)]
                db.add_all(projects)
                db.flush()
                runs = [AnalysisRunRow(project_id=project.id, status="completed",
                                       created_at=timestamp)
                        for project in projects]
                runs.extend(
                    AnalysisRunRow(project_id=projects[0].id, status="completed",
                                   created_at=timestamp + timedelta(seconds=index + 1))
                    for index in range(1_200)
                )
                db.add_all(runs)
                db.flush()
                heavy_project_latest_run_id = runs[-1].id
                db.add_all(AnalysisDiagnosticRow(run_id=run.id, payload={})
                           for run in runs)
                db.commit()

            statements = []

            def record_sql(_connection, _cursor, statement, parameters, _context, _executemany):
                if statement.lstrip().lower().startswith("select"):
                    statements.append((statement, parameters))

            with catalog_workspace(workspace_id):
                event.listen(engine, "before_cursor_execute", record_sql)
                try:
                    small = client.get("/api/v1/project-catalog", params={"page_size": 1})
                    small_queries = len(statements)
                    statements.clear()
                    large = client.get("/api/v1/project-catalog", params={"page_size": 80})
                    large_queries = len(statements)
                    latest_run_sql, latest_run_params = statements[3]
                finally:
                    event.remove(engine, "before_cursor_execute", record_sql)

            self.assertEqual(200, small.status_code, small.text)
            self.assertEqual(200, large.status_code, large.text)
            self.assertEqual(240, large.json()["total"])
            self.assertEqual(80, len(large.json()["items"]))
            self.assertEqual((5, 5), (small_queries, large_queries))
            self.assertEqual(projects[0].id, large.json()["items"][0]["id"])
            self.assertEqual(heavy_project_latest_run_id,
                             large.json()["items"][0]["latest_run"]["id"])
            with engine.connect() as connection:
                plan = connection.exec_driver_sql(
                    "EXPLAIN QUERY PLAN " + latest_run_sql, latest_run_params
                ).all()
            details = " ".join(str(row[3]) for row in plan).lower()
            self.assertIn("correlated scalar subquery", details)
            self.assertIn("ix_analysis_runs_project_created_id", details)
