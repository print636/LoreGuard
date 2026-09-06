# LoreGuard

LoreGuard is an evidence-first narrative consistency review platform for game writers and narrative designers. It extracts versioned facts and events from authorized story material, detects deterministic continuity conflicts, and optionally augments its baseline extractor with a validated OpenAI-compatible provider. Provider failures fall back to the baseline; model and baseline records are deduplicated and bound to source lines.

See [README.zh-CN.md](README.zh-CN.md) for the full guide. Credentials are read only from server-side environment variables and must never be committed.

Quick start:

```bash
docker compose up --build
```

The default stack keeps embeddings disabled and does not pull or start an
embedding model. An optional CPU-only local TEI overlay is pinned by image
digest and model revision:

```bash
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.rag.yml up --build
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.rag.yml --profile rag-smoke run --rm tei-smoke
```

The overlay serves `BAAI/bge-small-zh-v1.5` privately inside the Compose
network with float32 CLS pooling, a fixed revision, no silent truncation and a
persistent named model cache. It exposes no host port and requires no external
API key. The smoke verifies the reported model SHA, 512-dimensional finite
L2-normalized vectors, batch/single agreement, a frozen Chinese relevance
ordering and rejection of over-limit input. Initial startup needs network access
to download the pinned model revision; CPU latency, memory and disk consumption
depend on the host and are not presented as production throughput. The
[TEI project](https://github.com/huggingface/text-embeddings-inference) is
Apache-2.0 software and the pinned [BGE model card](https://huggingface.co/BAAI/bge-small-zh-v1.5)
identifies the weights as MIT;
deployers must still review both licenses for their use. This is local embedding
infrastructure only: the main analysis path still does not consume these vectors,
so the overlay is not evidence that product Evidence RAG is complete.

The repository includes a FastAPI backend, a React/Vite local project workbench, Markdown/TXT/JSON/DOCX imports, versioned document and run history, evidence-linked Cytoscape graph and conservative timeline projections, model chunking with global evidence lines, explicit alias normalization, deterministic local hybrid-retrieval diagnostics, Redis/Celery integration, resumable SSE progress, audited feedback, Docker Compose and CI. Graph/timeline views read persisted completed-run records and never trigger a provider call. The active analysis path still uses a stable SHA-256 character n-gram score rather than embedding retrieval. A separate opt-in Evidence RAG substrate now implements an independently configured OpenAI-compatible embedding client, deterministic Chinese line-aware chunks, versioned embedding profiles, exact project/document/version/content-hash snapshot storage, Alembic migrations, and a real PostgreSQL `vector` column. Local smoke runs have verified both pgvector snapshot/profile isolation and real pinned TEI embeddings, but the main analysis path still does not consume vector results and no hybrid-retrieval benefit evaluation exists yet, so Evidence RAG remains unfinished and is not a resume claim. Evaluation includes an 80-case directive regression, a 100-case synthetic natural-Chinese dev/test dataset, a 50-case developer-visible challenge-v2, a 14-case original multi-document complex acceptance suite, and a generated 24k-character long-text smoke. None is presented as a human blind test or production accuracy. See [the resume-readiness gate](docs/resume-readiness.md) for the deliberately conservative publication boundary.

An experimental first-stage bounded evidence-repair Agent is wired behind a feature flag that defaults to off. It uses LangGraph 1.2.11 `StateGraph` and an application-level JSON tool protocol—not native provider `tool_calls`—for `READ_SPAN`, `PATCH_RECORDS`, and `ABSTAIN`. Its safety boundaries have automated and Mock coverage. A three-task real-provider development pilot exposed benchmark-label errors, and all three tasks are permanently classified as tuned. The corrected v2 `full × 3` evaluation used the real Provider only for the Agent stage; the main-extraction candidate was synthetically injected from the frozen manifest, so this was not an end-to-end real-model extraction evaluation. It recorded all 90 required executions, including 81 holdout executions, but only 59/90 were strictly correct and the full gate failed: holdout runtime success was 74/81 with seven `read_timeout` failures, recovery was 26/51, and semantic abstention was 24/30. Twelve patches accepted by production validation were wrong against the evaluation oracle; no accepted safety violation was observed. No successful direct-`ABSTAIN` path was observed, so the required three-path coverage gate also failed. These results are failure diagnostics, not an Agent quality, product-benefit, extraction-quality, or resume-readiness claim. There is no multi-agent implementation, and real Evidence RAG remains unfinished. See the [first-stage boundary](docs/review-agent-phase1.md), [pilot record](docs/review-agent-pilot-20260906.md), [sanitized v2 full checkpoint](docs/review-agent-v2-full-checkpoint-20260906.md), [v2 frozen manifest](data/agent-acceptance-v2/manifest.json), and [offline runner](scripts/run_agent_acceptance.py).

Both JSON-text and multipart document APIs accept an explicit `document_role`
(`canon`, `character_profile`, `chapter`, or `reference`) and a constrained
`story_scope`. Omitted fields inherit from the previous same-name version; a new
document safely defaults to `chapter/global`. LoreGuard never guesses these values
from filenames. Completed runs expose questions and evidence gaps separately at
`GET /api/v1/analysis-runs/{id}/clarifications`; they are not counted as confirmed
consistency issues.

Multipart upload accepts ordinary `.docx` Office Open XML documents in addition
to UTF-8 Markdown, TXT and JSON. DOCX import turns main-body paragraphs, explicit
line breaks and table rows into stable plain-text lines so later evidence line
numbers refer to the imported representation. It applies bounded ZIP/XML parsing
and rejects encrypted, macro-enabled, malformed, ambiguous-path or oversized
packages. It never follows external relationships or imports macros, embedded
objects, images, comments, headers or footers. The JSON-text endpoint remains a
plain-text API and does not accept binary DOCX content.
