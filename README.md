# LoreGuard

LoreGuard is an evidence-first narrative consistency review platform for game writers and narrative designers. It extracts versioned facts and events from authorized story material, detects deterministic continuity conflicts, and optionally augments its baseline extractor with a validated OpenAI-compatible provider. Provider failures fall back to the baseline; model and baseline records are deduplicated and bound to source lines.

See [README.zh-CN.md](README.zh-CN.md) for the full guide. Credentials are read only from server-side environment variables and must never be committed.

Quick start:

```bash
docker compose up --build
```

The repository includes a FastAPI backend, a React/Vite local project workbench, versioned document and run history, evidence-linked Cytoscape graph and conservative timeline projections, model chunking with global evidence lines, explicit alias normalization, deterministic local hybrid-retrieval diagnostics, Redis/Celery integration, resumable SSE progress, audited feedback, Docker Compose and CI. Graph/timeline views read persisted completed-run records and never trigger a provider call. The retrieval vector-like score uses stable SHA-256 character n-grams, not embeddings or pgvector. Evaluation includes an 80-case directive regression, a 100-case synthetic natural-Chinese dev/test dataset, a 50-case developer-visible challenge-v2, a 14-case original multi-document complex acceptance suite, and a generated 24k-character long-text smoke. None is presented as a human blind test or production accuracy. See [the resume-readiness gate](docs/resume-readiness.md) for the deliberately conservative publication boundary.

An experimental first-stage bounded evidence-repair Agent is wired behind a feature flag that defaults to off. It uses LangGraph 1.2.11 `StateGraph` and an application-level JSON tool protocol—not native provider `tool_calls`—for `READ_SPAN`, `PATCH_RECORDS`, and `ABSTAIN`. Its safety boundaries and dynamic paths have automated and Mock coverage; the developer-visible frozen suite contains 30 tasks and 3 repetitions each, but all 90 scored executions currently come only from the Mock scorer. No real-model Agent acceptance has been completed, so this is not evidence of Agent quality, product benefit, or resume readiness. The fixed semantic-label repair pass is a separate non-Agent path, and real Evidence RAG is still unfinished. See the [first-stage boundary](docs/review-agent-phase1.md), [frozen manifest](data/agent-acceptance-v1/manifest.json), and [offline runner](scripts/run_agent_acceptance.py).

Both JSON-text and multipart document APIs accept an explicit `document_role`
(`canon`, `character_profile`, `chapter`, or `reference`) and a constrained
`story_scope`. Omitted fields inherit from the previous same-name version; a new
document safely defaults to `chapter/global`. LoreGuard never guesses these values
from filenames. Completed runs expose questions and evidence gaps separately at
`GET /api/v1/analysis-runs/{id}/clarifications`; they are not counted as confirmed
consistency issues.
