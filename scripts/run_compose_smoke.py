from __future__ import annotations

import argparse
import json
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


EXPECTED_CATEGORIES = {
    "fact_conflict",
    "location_collision",
    "knowledge_without_acquisition",
    "item_ownership",
    "world_rule_conflict",
}

PGVECTOR_SMOKE_SQL = r"""
\set ON_ERROR_STOP on
BEGIN;
DO $$
DECLARE
    current_revision text;
    vector_extension_count integer;
    vector_column_count integer;
BEGIN
    SELECT version_num INTO current_revision FROM alembic_version;
    IF current_revision <> '0003_embedding_identity' THEN
        RAISE EXCEPTION 'unexpected Alembic revision';
    END IF;
    SELECT count(*) INTO vector_extension_count FROM pg_extension WHERE extname = 'vector';
    IF vector_extension_count <> 1 THEN
        RAISE EXCEPTION 'pgvector extension is unavailable';
    END IF;
    SELECT count(*) INTO vector_column_count
      FROM information_schema.columns
     WHERE table_schema = current_schema()
       AND table_name = 'evidence_embeddings'
       AND column_name = 'vector'
       AND udt_name = 'vector';
    IF vector_column_count <> 1 THEN
        RAISE EXCEPTION 'evidence_embeddings.vector is not a pgvector column';
    END IF;
END $$;

INSERT INTO projects (id, name, description, created_at)
VALUES
    ('ci-rag-project', 'CI RAG smoke', '', now()),
    ('ci-rag-other', 'CI RAG other', '', now());
INSERT INTO documents (id, project_id, name, content, version, active, created_at)
VALUES
    ('ci-rag-document', 'ci-rag-project', 'ci.md', 'test', 2, true, now()),
    ('ci-rag-other-document', 'ci-rag-project', 'other.md', 'decoy', 1, true, now());
INSERT INTO embedding_profiles
    (id, provider_kind, provider_namespace, model_identifier, model_revision,
     deployment_fingerprint, document_transform_identity,
     query_transform_identity, dimensions, normalized, created_at)
VALUES
    ('emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e',
     'openai-compatible', 'ci-a', 'ci-model', 'r1', 'ci-runtime-v1',
     'raw-v1', 'raw-v1', 2, true, now()),
    ('emb-5547170f104692d0ca66e38375d4afaca58902e07528fc7a45fcafdb386aa930',
     'openai-compatible', 'ci-b', 'ci-model', 'r1', 'ci-runtime-v1',
     'raw-v1', 'raw-v1', 3, true, now()),
    ('emb-ed00e1e5883ed2c99440d69fe7c3421e19275f008683e161e8b118ee441a63da',
     'openai-compatible', 'ci-c', 'ci-model', 'r1', 'ci-runtime-v1',
     'raw-v1', 'raw-v1', 2, true, now());
INSERT INTO evidence_chunks
    (id, project_id, document_id, document_version, content_sha256,
     chunker_version, ordinal, text, text_sha256, char_start, char_end,
     line_start, line_end, created_at)
VALUES
    ('chk-' || repeat('a', 64), 'ci-rag-project', 'ci-rag-other-document', 1,
     repeat('1', 64), 'ci-v1', 0, 'wrong document', repeat('3', 64),
     0, 14, 1, 1, now()),
    ('chk-' || repeat('b', 64), 'ci-rag-project', 'ci-rag-document', 1,
     repeat('1', 64), 'ci-v1', 0, 'target', repeat('4', 64), 0, 6, 1, 1, now()),
    ('chk-' || repeat('c', 64), 'ci-rag-project', 'ci-rag-document', 1,
     repeat('1', 64), 'ci-v1', 1, 'same snapshot', repeat('5', 64),
     0, 13, 1, 1, now()),
    ('chk-' || repeat('d', 64), 'ci-rag-project', 'ci-rag-document', 2,
     repeat('1', 64), 'ci-v1', 0, 'wrong version', repeat('6', 64),
     0, 13, 1, 1, now()),
    ('chk-' || repeat('e', 64), 'ci-rag-project', 'ci-rag-document', 1,
     repeat('2', 64), 'ci-v1', 0, 'wrong content hash', repeat('7', 64),
     0, 18, 1, 1, now());
INSERT INTO evidence_embeddings (chunk_id, profile_id, dimensions, vector, created_at)
VALUES
    ('chk-' || repeat('a', 64),
     'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e',
     2, '[1,0]'::vector, now()),
    ('chk-' || repeat('b', 64),
     'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e',
     2, '[0.8,0.2]'::vector, now()),
    ('chk-' || repeat('c', 64),
     'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e',
     2, '[0,1]'::vector, now()),
    ('chk-' || repeat('d', 64),
     'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e',
     2, '[1,0]'::vector, now()),
    ('chk-' || repeat('e', 64),
     'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e',
     2, '[1,0]'::vector, now()),
    ('chk-' || repeat('c', 64),
     'emb-5547170f104692d0ca66e38375d4afaca58902e07528fc7a45fcafdb386aa930',
     3, '[1,0,0]'::vector, now()),
    ('chk-' || repeat('c', 64),
     'emb-ed00e1e5883ed2c99440d69fe7c3421e19275f008683e161e8b118ee441a63da',
     2, '[1,0]'::vector, now());

DO $$
DECLARE
    nearest_chunk text;
    decoy_chunk text;
    different_dimension_count integer;
BEGIN
    WITH candidates AS MATERIALIZED (
        SELECT c.id AS chunk_id, e.vector
          FROM evidence_chunks c
          JOIN evidence_embeddings e ON e.chunk_id = c.id
         -- documents.id is a global PK and the composite owner FK binds its
         -- project_id; this project predicate is explicit defense-in-depth.
         WHERE c.project_id = 'ci-rag-project'
           AND c.document_id = 'ci-rag-document'
           AND c.document_version = 1
           AND c.content_sha256 = repeat('1', 64)
           AND e.profile_id =
               'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e'
    )
    SELECT chunk_id INTO nearest_chunk
      FROM candidates
     ORDER BY vector <=> '[1,0]'::vector, chunk_id
     LIMIT 1;
    IF nearest_chunk <> ('chk-' || repeat('b', 64)) THEN
        RAISE EXCEPTION 'pgvector ordering or snapshot/profile isolation is incorrect';
    END IF;

    -- Counterfactual checks prove that each individual snapshot predicate is
    -- necessary: every omitted field admits a closer, otherwise-valid row.
    WITH document_filter_omitted AS MATERIALIZED (
        SELECT c.id AS chunk_id, e.vector
          FROM evidence_chunks c
          JOIN evidence_embeddings e ON e.chunk_id = c.id
         WHERE c.project_id = 'ci-rag-project'
           AND c.document_version = 1
           AND c.content_sha256 = repeat('1', 64)
           AND e.profile_id =
               'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e'
    )
    SELECT chunk_id INTO decoy_chunk FROM document_filter_omitted
     ORDER BY vector <=> '[1,0]'::vector, chunk_id LIMIT 1;
    IF decoy_chunk <> ('chk-' || repeat('a', 64)) THEN
        RAISE EXCEPTION 'document isolation decoy is ineffective';
    END IF;

    WITH version_filter_omitted AS MATERIALIZED (
        SELECT c.id AS chunk_id, e.vector
          FROM evidence_chunks c
          JOIN evidence_embeddings e ON e.chunk_id = c.id
         WHERE c.project_id = 'ci-rag-project'
           AND c.document_id = 'ci-rag-document'
           AND c.content_sha256 = repeat('1', 64)
           AND e.profile_id =
               'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e'
    )
    SELECT chunk_id INTO decoy_chunk FROM version_filter_omitted
     ORDER BY vector <=> '[1,0]'::vector, chunk_id LIMIT 1;
    IF decoy_chunk <> ('chk-' || repeat('d', 64)) THEN
        RAISE EXCEPTION 'version isolation decoy is ineffective';
    END IF;

    WITH content_hash_filter_omitted AS MATERIALIZED (
        SELECT c.id AS chunk_id, e.vector
          FROM evidence_chunks c
          JOIN evidence_embeddings e ON e.chunk_id = c.id
         WHERE c.project_id = 'ci-rag-project'
           AND c.document_id = 'ci-rag-document'
           AND c.document_version = 1
           AND e.profile_id =
               'emb-a98ebdc29966c3972c656fe1d405df5e53303aa229c9536b76e943922c769d5e'
    )
    SELECT chunk_id INTO decoy_chunk FROM content_hash_filter_omitted
     ORDER BY vector <=> '[1,0]'::vector, chunk_id LIMIT 1;
    IF decoy_chunk <> ('chk-' || repeat('e', 64)) THEN
        RAISE EXCEPTION 'content hash isolation decoy is ineffective';
    END IF;

    WITH profile_filter_omitted_same_dim AS MATERIALIZED (
        SELECT c.id AS chunk_id, e.vector
          FROM evidence_chunks c
          JOIN evidence_embeddings e ON e.chunk_id = c.id
         WHERE c.project_id = 'ci-rag-project'
           AND c.document_id = 'ci-rag-document'
           AND c.document_version = 1
           AND c.content_sha256 = repeat('1', 64)
           AND e.dimensions = 2
    )
    SELECT chunk_id INTO decoy_chunk FROM profile_filter_omitted_same_dim
     ORDER BY vector <=> '[1,0]'::vector, chunk_id LIMIT 1;
    IF decoy_chunk <> ('chk-' || repeat('c', 64)) THEN
        RAISE EXCEPTION 'same-dimension profile isolation decoy is ineffective';
    END IF;

    SELECT count(*) INTO different_dimension_count
      FROM evidence_chunks c
      JOIN evidence_embeddings e ON e.chunk_id = c.id
     WHERE c.project_id = 'ci-rag-project'
       AND c.document_id = 'ci-rag-document'
       AND c.document_version = 1
       AND c.content_sha256 = repeat('1', 64)
       AND e.profile_id =
           'emb-5547170f104692d0ca66e38375d4afaca58902e07528fc7a45fcafdb386aa930'
       AND e.dimensions = 3;
    IF different_dimension_count <> 1 THEN
        RAISE EXCEPTION 'different-dimension profile safety decoy is missing';
    END IF;

    BEGIN
        INSERT INTO evidence_chunks
            (id, project_id, document_id, document_version, content_sha256,
             chunker_version, ordinal, text, text_sha256, char_start, char_end,
             line_start, line_end, created_at)
        VALUES
            ('chk-' || repeat('f', 64), 'ci-rag-other', 'ci-rag-document', 1,
             repeat('3', 64), 'ci-v1', 0, 'wrong owner', repeat('8', 64),
             0, 11, 1, 1, now());
        RAISE EXCEPTION 'cross-project document ownership was accepted';
    EXCEPTION WHEN foreign_key_violation THEN
        NULL;
    END;
END $$;
ROLLBACK;
"""


def request_json(base_url: str, path: str, method: str = "GET"):
    request = Request(f"{base_url}{path}", method=method)
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if request_json(base_url, "/health").get("status") == "ok":
                return
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"API did not become ready: {type(last_error).__name__}")


def verify_pgvector_compose() -> None:
    """Verify the migrated Compose database without downloading any model."""

    completed = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "loreguard",
            "-d",
            "loreguard",
            "-X",
            "-q",
            "-f",
            "-",
        ],
        input=PGVECTOR_SMOKE_SQL,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("PostgreSQL/pgvector Compose verification failed")


def run(base_url: str, timeout_seconds: float) -> dict:
    wait_until_ready(base_url, timeout_seconds)
    verify_pgvector_compose()
    project = request_json(base_url, "/api/v1/demo/advanced", "POST")
    run_row = request_json(
        base_url, f"/api/v1/projects/{project['id']}/analysis-runs", "POST"
    )
    deadline = time.monotonic() + timeout_seconds
    status = "queued"
    while time.monotonic() < deadline:
        run_row = request_json(base_url, f"/api/v1/analysis-runs/{run_row['id']}")
        status = run_row["status"]
        if status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(1)
    if status != "completed":
        raise RuntimeError(f"Celery-backed analysis ended as {status}")

    run_id = run_row["id"]
    issues = request_json(base_url, f"/api/v1/analysis-runs/{run_id}/issues")
    categories = {issue["category"] for issue in issues}
    if categories != EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"Unexpected issue categories: {sorted(categories)}"
        )
    graph = request_json(base_url, f"/api/v1/analysis-runs/{run_id}/graph")
    timeline = request_json(base_url, f"/api/v1/analysis-runs/{run_id}/timeline")
    timeline_entry_count = sum(len(group["entries"]) for group in timeline["groups"])
    timeline_entry_count += len(timeline["unscheduled"])
    with urlopen(f"{base_url}/api/v1/analysis-runs/{run_id}/events", timeout=10) as response:
        events = response.read().decode("utf-8")
    if "event: terminal" not in events or '"status": "completed"' not in events:
        raise RuntimeError("SSE stream did not expose the completed terminal state")
    if not graph["nodes"] or not timeline_entry_count:
        raise RuntimeError("Completed run projections are unexpectedly empty")
    return {
        "status": status,
        "document_count": project["document_count"],
        "issue_count": len(issues),
        "categories": sorted(categories),
        "graph_nodes": len(graph["nodes"]),
        "timeline_entries": timeline_entry_count,
        "sse_terminal": True,
        "pgvector_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.base_url.rstrip("/"), args.timeout_seconds),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
