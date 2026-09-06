import unittest
from pathlib import Path
from unittest.mock import patch

from app.embeddings import EmbeddingProfile
from scripts.run_compose_smoke import PGVECTOR_SMOKE_SQL, verify_pgvector_compose, wait_until_ready


class ComposeSmokeTests(unittest.TestCase):
    def test_backend_docker_context_excludes_local_secrets_and_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        rules = {
            line.strip()
            for line in (root / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue(
            {
                ".env",
                ".env.*",
                ".git",
                ".venv",
                "artifacts",
                "*.db",
                "frontend/node_modules",
                "frontend/dist",
                "**/__pycache__",
                ".pytest_cache",
                "*.log",
            }
            <= rules
        )
        self.assertIn("!.env.example", rules)

        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        for required_source in (
            "requirements.txt",
            "alembic.ini",
            "migrations",
            "app",
            "data",
        ):
            self.assertNotIn(required_source, rules)
            self.assertIn(required_source, dockerfile)

    def test_frontend_docker_context_excludes_host_build_artifacts(self):
        frontend_root = Path(__file__).resolve().parents[1] / "frontend"
        rules = {
            line.strip()
            for line in (frontend_root / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue(
            {"node_modules", "dist", ".pnpm-store", ".vite", ".cache"} <= rules
        )
        dockerfile = (frontend_root / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("RUN corepack enable && pnpm install --frozen-lockfile", dockerfile)
        self.assertIn("COPY . .", dockerfile)

    def test_readiness_retries_connection_reset_during_container_startup(self):
        with (
            patch(
                "scripts.run_compose_smoke.request_json",
                side_effect=[ConnectionResetError("starting"), {"status": "ok"}],
            ) as request,
            patch("scripts.run_compose_smoke.time.sleep"),
        ):
            wait_until_ready("http://127.0.0.1:8000", 5)
        self.assertEqual(2, request.call_count)

    @patch("scripts.run_compose_smoke.subprocess.run")
    def test_pgvector_check_is_non_model_transactional_and_fail_closed(self, execute):
        execute.return_value.returncode = 0
        verify_pgvector_compose()
        command = execute.call_args.args[0]
        self.assertEqual(command[:4], ["docker", "compose", "exec", "-T"])
        self.assertIn("BEGIN;", PGVECTOR_SMOKE_SQL)
        self.assertIn("ROLLBACK;", PGVECTOR_SMOKE_SQL)
        self.assertIn("<=>", PGVECTOR_SMOKE_SQL)
        self.assertIn("document_version = 1", PGVECTOR_SMOKE_SQL)
        self.assertIn("content_sha256", PGVECTOR_SMOKE_SQL)
        candidates = PGVECTOR_SMOKE_SQL.split(
            "WITH candidates AS MATERIALIZED (", 1
        )[1].split(")\n    SELECT chunk_id", 1)[0]
        for required_filter in (
            "c.project_id = 'ci-rag-project'",
            "c.document_id = 'ci-rag-document'",
            "c.document_version = 1",
            "c.content_sha256 = repeat('1', 64)",
            "e.profile_id =",
        ):
            self.assertIn(required_filter, candidates)
        # Each legal decoy changes one snapshot key only and is closer than
        # the target. Omitting any one predicate would therefore change the
        # winner instead of merely broadening an otherwise harmless result.
        self.assertIn("'ci-rag-other-document', 1", PGVECTOR_SMOKE_SQL)
        self.assertIn("'ci-rag-document', 2", PGVECTOR_SMOKE_SQL)
        self.assertIn("repeat('2', 64), 'ci-v1', 0, 'wrong content hash'", PGVECTOR_SMOKE_SQL)
        self.assertIn("wrong document", PGVECTOR_SMOKE_SQL)
        self.assertIn("wrong version", PGVECTOR_SMOKE_SQL)
        self.assertIn("2, '[0.8,0.2]'::vector", PGVECTOR_SMOKE_SQL)
        self.assertGreaterEqual(PGVECTOR_SMOKE_SQL.count("2, '[1,0]'::vector"), 4)
        self.assertIn("same snapshot", PGVECTOR_SMOKE_SQL)
        self.assertIn("3, '[1,0,0]'::vector", PGVECTOR_SMOKE_SQL)
        self.assertIn("repeat('b', 64)", PGVECTOR_SMOKE_SQL)
        self.assertIn("global PK", PGVECTOR_SMOKE_SQL)
        self.assertIn("composite owner FK", PGVECTOR_SMOKE_SQL)
        for cte_name, omitted_predicate in (
            ("document_filter_omitted", "c.document_id ="),
            ("version_filter_omitted", "c.document_version ="),
            ("content_hash_filter_omitted", "c.content_sha256 ="),
        ):
            counterfactual = PGVECTOR_SMOKE_SQL.split(
                f"WITH {cte_name} AS MATERIALIZED (", 1
            )[1].split(
                f")\n    SELECT chunk_id INTO decoy_chunk FROM {cte_name}", 1
            )[0]
            self.assertNotIn(omitted_predicate, counterfactual)
            self.assertIn("e.profile_id =", counterfactual)
        same_dimension_profile_counterfactual = PGVECTOR_SMOKE_SQL.split(
            "WITH profile_filter_omitted_same_dim AS MATERIALIZED (", 1
        )[1].split(
            ")\n    SELECT chunk_id INTO decoy_chunk "
            "FROM profile_filter_omitted_same_dim",
            1,
        )[0]
        self.assertNotIn("e.profile_id =", same_dimension_profile_counterfactual)
        self.assertIn("e.dimensions = 2", same_dimension_profile_counterfactual)
        for expected_decoy in (
            "document isolation decoy is ineffective",
            "version isolation decoy is ineffective",
            "content hash isolation decoy is ineffective",
            "same-dimension profile isolation decoy is ineffective",
            "different-dimension profile safety decoy is missing",
        ):
            self.assertIn(expected_decoy, PGVECTOR_SMOKE_SQL)
        profile_ids = {}
        for namespace, dimensions in (("ci-a", 2), ("ci-b", 3), ("ci-c", 2)):
            profile_id = EmbeddingProfile.openai_compatible(
                provider_namespace=namespace,
                model_identifier="ci-model",
                model_revision="r1",
                dimensions=dimensions,
            ).profile_id
            profile_ids[namespace] = profile_id
            self.assertIn(profile_id, PGVECTOR_SMOKE_SQL)
        self.assertIn(profile_ids["ci-a"], candidates)
        self.assertNotIn(profile_ids["ci-b"], candidates)
        self.assertNotIn(profile_ids["ci-c"], candidates)
        self.assertNotIn("embedding_api", PGVECTOR_SMOKE_SQL.lower())

        execute.return_value.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "pgvector"):
            verify_pgvector_compose()


if __name__ == "__main__":
    unittest.main()
