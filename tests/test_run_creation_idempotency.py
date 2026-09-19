from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import func, select

# Run-creation reliability tests must not inherit a developer model credential.
os.environ["OPENAI_API_KEY"] = ""
os.environ["ENABLE_MODEL_EXTRACTION"] = "false"

from app.db import (  # noqa: E402
    AnalysisRunInputRow,
    AnalysisRunRow,
    DocumentRow,
    ProjectRow,
    RunEventRow,
    SessionLocal,
)
from app.main import (  # noqa: E402
    _create_run_or_load_winner,
    app,
    settings,
    write_limiter,
)
from app.service import DISPATCH_FAILED_ERROR, capture_run_inputs  # noqa: E402


settings.enable_model_extraction = False
settings.openai_api_key = ""


class RunCreationIdempotencyTests(unittest.TestCase):
    def setUp(self) -> None:
        write_limiter.events.clear()

    def tearDown(self) -> None:
        write_limiter.events.clear()

    def _project_with_document(self, client: TestClient, content: str = "第一版正文") -> dict:
        project = client.post(
            "/api/v1/projects", json={"name": f"幂等测试-{uuid4().hex}"}
        ).json()
        response = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={"name": "chapter.md", "content": content},
        )
        self.assertEqual(201, response.status_code, response.text)
        return project

    def test_sequential_duplicate_key_returns_same_run_and_dispatches_once(self):
        with TestClient(app) as client, patch("app.main.dispatch_analysis") as dispatch:
            project = self._project_with_document(client)
            path = f"/api/v1/projects/{project['id']}/analysis-runs"
            headers = {"Idempotency-Key": "start-20260919.001"}

            first = client.post(path, headers=headers)
            second = client.post(path, headers=headers)

        self.assertEqual(202, first.status_code, first.text)
        self.assertEqual(202, second.status_code, second.text)
        self.assertEqual(first.json()["id"], second.json()["id"])
        self.assertFalse(first.json()["deduplicated"])
        self.assertTrue(second.json()["deduplicated"])
        dispatch.assert_called_once_with(first.json()["id"])
        self.assertNotIn("start-20260919.001", str(first.json()))
        with SessionLocal() as db:
            runs = db.scalars(
                select(AnalysisRunRow).where(
                    AnalysisRunRow.project_id == project["id"],
                    AnalysisRunRow.idempotency_key == "start-20260919.001",
                )
            ).all()
            self.assertEqual(1, len(runs))
            input_count = db.scalar(
                select(func.count())
                .select_from(AnalysisRunInputRow)
                .where(AnalysisRunInputRow.run_id == runs[0].id)
            )
            self.assertEqual(1, input_count)

    def test_concurrent_requests_create_one_snapshot_set_and_dispatch_once(self):
        with TestClient(app) as client:
            project = self._project_with_document(client)
            path = f"/api/v1/projects/{project['id']}/analysis-runs"
            headers = {"Idempotency-Key": "parallel-001"}
            with patch("app.main.dispatch_analysis") as dispatch:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    responses = list(
                        pool.map(lambda _: client.post(path, headers=headers), range(6))
                    )

        self.assertTrue(all(item.status_code == 202 for item in responses))
        run_ids = {item.json()["id"] for item in responses}
        self.assertEqual(1, len(run_ids))
        run_id = run_ids.pop()
        dispatch.assert_called_once_with(run_id)
        with SessionLocal() as db:
            self.assertEqual(
                1,
                db.scalar(
                    select(func.count())
                    .select_from(AnalysisRunRow)
                    .where(
                        AnalysisRunRow.project_id == project["id"],
                        AnalysisRunRow.idempotency_key == "parallel-001",
                    )
                ),
            )
            self.assertEqual(
                1,
                db.scalar(
                    select(func.count())
                    .select_from(AnalysisRunInputRow)
                    .where(AnalysisRunInputRow.run_id == run_id)
                ),
            )

    def test_unique_constraint_race_path_loads_committed_winner(self):
        with TestClient(app) as client:
            project = self._project_with_document(client)
        with SessionLocal() as db:
            document = db.scalar(
                select(DocumentRow).where(DocumentRow.project_id == project["id"])
            )
            winner = AnalysisRunRow(
                project_id=project["id"], idempotency_key="race-winner"
            )
            db.add(winner)
            db.flush()
            capture_run_inputs(db, winner, [document])
            db.commit()
            winner_id = winner.id

        prepare_called = False
        with SessionLocal() as db:
            loser = AnalysisRunRow(
                project_id=project["id"], idempotency_key="race-winner"
            )

            def prepare(_run):
                nonlocal prepare_called
                prepare_called = True

            resolved, created = _create_run_or_load_winner(
                db, loser, "race-winner", prepare
            )
            self.assertEqual(winner_id, resolved.id)
            self.assertFalse(created)
        self.assertFalse(prepare_called)

    def test_different_keys_and_missing_header_remain_distinct(self):
        with TestClient(app) as client, patch("app.main.dispatch_analysis") as dispatch:
            first_project = self._project_with_document(client)
            second_project = self._project_with_document(client, "另一个项目")
            first_path = f"/api/v1/projects/{first_project['id']}/analysis-runs"
            second_path = f"/api/v1/projects/{second_project['id']}/analysis-runs"
            responses = [
                client.post(first_path, headers={"Idempotency-Key": "key-a"}),
                client.post(first_path, headers={"Idempotency-Key": "key-b"}),
                client.post(first_path),
                client.post(first_path),
                # Project scope permits the same opaque key in another project.
                client.post(second_path, headers={"Idempotency-Key": "key-a"}),
            ]
        self.assertTrue(all(item.status_code == 202 for item in responses))
        self.assertEqual(5, len({item.json()["id"] for item in responses}))
        self.assertEqual(5, dispatch.call_count)

    def test_retry_is_idempotent_and_keeps_original_frozen_input(self):
        with TestClient(app) as client:
            project = self._project_with_document(client)
            with patch("app.main.dispatch_analysis"):
                original = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                ).json()
            with SessionLocal() as db:
                row = db.get(AnalysisRunRow, original["id"])
                row.status = "failed"
                db.commit()
            updated = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={"name": "chapter.md", "content": "第二版正文"},
            )
            self.assertEqual(201, updated.status_code, updated.text)

            headers = {"Idempotency-Key": "retry-original-v1"}
            retry_path = f"/api/v1/analysis-runs/{original['id']}/retry"
            with patch("app.main.dispatch_analysis") as dispatch:
                first = client.post(retry_path, headers=headers)
                second = client.post(retry_path, headers=headers)

            self.assertEqual(first.json()["id"], second.json()["id"])
            self.assertTrue(second.json()["deduplicated"])
            self.assertEqual(original["id"], first.json()["retried_from"])
            dispatch.assert_called_once_with(first.json()["id"])
            status = client.get(
                f"/api/v1/analysis-runs/{first.json()['id']}"
            ).json()
            self.assertEqual(1, status["input_documents"][0]["document_version"])
            self.assertEqual("第一版正文", self._snapshot_content(first.json()["id"]))

    def test_key_reuse_for_different_operation_or_retry_source_conflicts(self):
        with TestClient(app) as client, patch("app.main.dispatch_analysis"):
            project = self._project_with_document(client)
            start_path = f"/api/v1/projects/{project['id']}/analysis-runs"
            original = client.post(
                start_path,
                headers={"Idempotency-Key": "operation-start-key"},
            ).json()
            second = client.post(start_path).json()
            with SessionLocal() as db:
                db.get(AnalysisRunRow, original["id"]).status = "failed"
                db.get(AnalysisRunRow, second["id"]).status = "failed"
                db.commit()

            start_key_on_retry = client.post(
                f"/api/v1/analysis-runs/{original['id']}/retry",
                headers={"Idempotency-Key": "operation-start-key"},
            )
            self.assertEqual(409, start_key_on_retry.status_code)
            self.assertEqual(
                "idempotency_key_conflict",
                start_key_on_retry.json()["detail"]["code"],
            )

            retry_headers = {"Idempotency-Key": "retry-source-key"}
            first_retry = client.post(
                f"/api/v1/analysis-runs/{original['id']}/retry",
                headers=retry_headers,
            )
            self.assertEqual(202, first_retry.status_code, first_retry.text)
            wrong_source = client.post(
                f"/api/v1/analysis-runs/{second['id']}/retry",
                headers=retry_headers,
            )
            self.assertEqual(409, wrong_source.status_code)
            self.assertEqual(
                "idempotency_key_conflict",
                wrong_source.json()["detail"]["code"],
            )
            retry_key_on_start = client.post(start_path, headers=retry_headers)
            self.assertEqual(409, retry_key_on_start.status_code)

    def _snapshot_content(self, run_id: str) -> str:
        with SessionLocal() as db:
            return db.scalar(
                select(AnalysisRunInputRow.content).where(
                    AnalysisRunInputRow.run_id == run_id
                )
            )

    def test_dispatch_failure_is_terminal_recoverable_and_secret_safe(self):
        secret = "private-dispatch-secret-marker"
        endpoint = "https://private-provider.invalid/v1"
        with TestClient(app) as client:
            project = self._project_with_document(client)
            path = f"/api/v1/projects/{project['id']}/analysis-runs"
            headers = {"Idempotency-Key": "dispatch-failure-001"}
            with patch(
                "app.main.dispatch_analysis",
                side_effect=RuntimeError(f"{secret} at {endpoint}"),
            ) as dispatch:
                failed = client.post(path, headers=headers)
                repeated = client.post(path, headers=headers)

            self.assertEqual(503, failed.status_code, failed.text)
            self.assertEqual("1", failed.headers["Retry-After"])
            run_id = failed.json()["detail"]["run_id"]
            self.assertTrue(failed.json()["detail"]["retryable"])
            self.assertEqual(202, repeated.status_code, repeated.text)
            self.assertEqual(run_id, repeated.json()["id"])
            self.assertEqual("failed", repeated.json()["status"])
            self.assertTrue(repeated.json()["deduplicated"])
            dispatch.assert_called_once_with(run_id)

            status = client.get(f"/api/v1/analysis-runs/{run_id}")
            self.assertEqual("failed", status.json()["status"])
            self.assertEqual(DISPATCH_FAILED_ERROR, status.json()["error"])
            serialized = failed.text + repeated.text + status.text
            self.assertNotIn(secret, serialized)
            self.assertNotIn(endpoint, serialized)
            self.assertNotIn("dispatch-failure-001", serialized)
            with SessionLocal() as db:
                events = db.scalars(
                    select(RunEventRow).where(RunEventRow.run_id == run_id)
                ).all()
            self.assertEqual(1, len(events))
            self.assertEqual(("failed", 100), (events[0].stage, events[0].progress))
            self.assertNotIn(secret, events[0].message)

    def test_key_validation_rejects_blank_unsafe_and_oversized_values(self):
        with TestClient(app) as client, patch("app.main.dispatch_analysis"):
            project = self._project_with_document(client)
            path = f"/api/v1/projects/{project['id']}/analysis-runs"
            blank = client.post(path, headers={"Idempotency-Key": "   "})
            unsafe = client.post(path, headers={"Idempotency-Key": "bad/key"})
            oversized = client.post(path, headers={"Idempotency-Key": "a" * 129})
        self.assertEqual(400, blank.status_code)
        self.assertEqual(400, unsafe.status_code)
        self.assertEqual(422, oversized.status_code)


if __name__ == "__main__":
    unittest.main()
