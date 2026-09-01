# LoreGuard

LoreGuard is an evidence-first narrative consistency review platform for game writers and narrative designers. It extracts versioned facts and events from authorized story material, detects deterministic continuity conflicts, and optionally augments its baseline extractor with a validated OpenAI-compatible provider. Provider failures fall back to the baseline; model and baseline records are deduplicated and bound to source lines.

See [README.zh-CN.md](README.zh-CN.md) for the full guide. Credentials are read only from server-side environment variables and must never be committed.

Quick start:

```bash
docker compose up --build
```

The repository includes a FastAPI backend, a React/Vite local project workbench, versioned document and run history, evidence-linked Cytoscape graph and conservative timeline projections, model chunking with global evidence lines, explicit alias normalization, deterministic local hybrid-retrieval diagnostics, Redis/Celery integration, resumable SSE progress, audited feedback, Docker Compose and CI. Graph/timeline views read persisted completed-run records and never trigger a provider call. The retrieval vector-like score uses stable SHA-256 character n-grams, not embeddings or pgvector. Evaluation includes an 80-case directive regression, a 100-case synthetic natural-Chinese dev/test dataset, a 50-case developer-visible challenge-v2, and a generated 24k-character long-text smoke. None is presented as a human blind test or production accuracy.
