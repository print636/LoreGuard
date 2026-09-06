# LoreGuard

LoreGuard is an evidence-first narrative consistency review platform for game writers and narrative designers. It extracts versioned facts and events from authorized story material, detects deterministic continuity conflicts, and optionally augments its baseline extractor with a validated OpenAI-compatible provider. Provider failures fall back to the baseline; model and baseline records are deduplicated and bound to source lines.

See [README.zh-CN.md](README.zh-CN.md) for the full guide. Credentials are read only from server-side environment variables and must never be committed.

Quick start:

```bash
docker compose up --build
```

The repository includes a FastAPI backend, a React/Vite local project workbench, versioned document and run history, evidence-linked Cytoscape graph and conservative timeline projections, model chunking with global evidence lines, explicit alias normalization, deterministic local hybrid-retrieval diagnostics, Redis/Celery integration, resumable SSE progress, audited feedback, Docker Compose and CI. Graph/timeline views read persisted completed-run records and never trigger a provider call. The retrieval vector-like score uses stable SHA-256 character n-grams, not embeddings or pgvector. Evaluation includes an 80-case directive regression, a 100-case synthetic natural-Chinese dev/test dataset, a 50-case developer-visible challenge-v2, a 14-case original multi-document complex acceptance suite, and a generated 24k-character long-text smoke. None is presented as a human blind test or production accuracy. See [the resume-readiness gate](docs/resume-readiness.md) for the deliberately conservative publication boundary.

An experimental first-stage bounded evidence-repair Agent is wired behind a feature flag that defaults to off. It uses LangGraph 1.2.11 `StateGraph` and an application-level JSON tool protocol—not native provider `tool_calls`—for `READ_SPAN`, `PATCH_RECORDS`, and `ABSTAIN`. Its safety boundaries have automated and Mock coverage. A three-task real-provider development pilot exposed benchmark-label errors, and all three tasks are permanently classified as tuned. The corrected v2 `full × 3` evaluation used the real Provider only for the Agent stage; the main-extraction candidate was synthetically injected from the frozen manifest, so this was not an end-to-end real-model extraction evaluation. It recorded all 90 required executions, including 81 holdout executions, but only 59/90 were strictly correct and the full gate failed: holdout runtime success was 74/81 with seven `read_timeout` failures, recovery was 26/51, and semantic abstention was 24/30. Twelve patches accepted by production validation were wrong against the evaluation oracle; no accepted safety violation was observed. No successful direct-`ABSTAIN` path was observed, so the required three-path coverage gate also failed. These results are failure diagnostics, not an Agent quality, product-benefit, extraction-quality, or resume-readiness claim. There is no multi-agent implementation, and real Evidence RAG remains unfinished. See the [first-stage boundary](docs/review-agent-phase1.md), [pilot record](docs/review-agent-pilot-20260906.md), [sanitized v2 full checkpoint](docs/review-agent-v2-full-checkpoint-20260906.md), [v2 frozen manifest](data/agent-acceptance-v2/manifest.json), and [offline runner](scripts/run_agent_acceptance.py).

Both JSON-text and multipart document APIs accept an explicit `document_role`
(`canon`, `character_profile`, `chapter`, or `reference`) and a constrained
`story_scope`. Omitted fields inherit from the previous same-name version; a new
document safely defaults to `chapter/global`. LoreGuard never guesses these values
from filenames. Completed runs expose questions and evidence gaps separately at
`GET /api/v1/analysis-runs/{id}/clarifications`; they are not counted as confirmed
consistency issues.
