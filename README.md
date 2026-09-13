# LoreGuard

LoreGuard is an evidence-first narrative consistency review platform for game writers and narrative designers. It extracts versioned facts and events from authorized story material, detects deterministic continuity conflicts, and optionally augments its baseline extractor with a validated OpenAI-compatible provider. Provider failures fall back to the baseline; model and baseline records are deduplicated and bound to source lines. A separate default-off Evidence Investigator uses provider-native function calls to search and read authorized snapshots before submitting one untrusted candidate or abstaining; deterministic promotion remains the only path from that candidate to an issue.

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

Those commands do not enable chat-model calls. To exercise the downstream AI
evidence review in the UI, copy `.env.example` to an uncommitted `.env`, add the
server-side model configuration, set `ENABLE_ISSUE_EVIDENCE_REVIEW=true`, and
start the overlay with `--env-file .env`. Completed issue cards then show the AI
verdict, hybrid retrieval marker, and authorized citations without replacing the
deterministic issue.

Every remote model path is independently opt-in. `ENABLE_MODEL_EXTRACTION`
enables chat-based record extraction; `ENABLE_REVIEW_AGENT` additionally enables
the older LangGraph repair stage inside that extraction path and does not work as
a standalone caller; `ENABLE_ISSUE_EVIDENCE_REVIEW` enables the post-rule chat
annotation; and `ENABLE_EVIDENCE_INVESTIGATOR` enables the separate
provider-native function-calling loop. `ENABLE_EMBEDDINGS` enables only the
embedding client and does not enable chat by itself. The Reviewer and Investigator
also require their documented embedding/PostgreSQL configuration, and all five
switches default to false. The UI connection test is a separate explicit user
action that makes one minimal chat request; merely loading the page never calls a
model.

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
deployers must still review both licenses for their use. The same pinned runtime
has now been exercised through the version-scoped pgvector retrieval path and the
optional downstream Evidence Reviewer; this remains a small local evaluation,
not a production-throughput claim.

The repository includes a FastAPI backend, a React/Vite local project workbench, Markdown/TXT/JSON/DOCX imports, versioned document and run history, evidence-linked Cytoscape graph and conservative timeline projections, model chunking with global evidence lines, explicit alias normalization, Redis/Celery integration, resumable SSE progress, audited feedback, Docker Compose and CI. Graph/timeline views read persisted completed-run records and never trigger a provider call. The deterministic issue engine still uses its reproducible local candidate path. Separately, an opt-in Evidence RAG path now performs real BGE embeddings, exact PostgreSQL/pgvector cosine search, strict project/document/version/content-hash isolation, and keyword+dense reciprocal-rank fusion. On the first frozen 28-query holdout, hybrid Recall@5 was 81.82%, all-evidence@5 was 71.43%, and low-lexical Recall@5 was 79.31%; the last metric missed the preset gate by one evidence item, so the overall retrieval gate and any claim of hybrid superiority remain false. See the [sanitized holdout report](docs/evidence-retrieval-v1-holdout.md).

An optional downstream `IssueEvidenceReviewer` sends only authorized retrieved evidence to the same validated chat contract and stores a separate AI evidence annotation. It never deletes or rewrites the deterministic issue. It is disabled by default and requires the explicit `ENABLE_ISSUE_EVIDENCE_REVIEW=true` switch together with embedding, PostgreSQL and model configuration. In the frozen 12-case, three-repeat A/B, local context scored 4/12 and RAG evidence scored 7/12; contextual exceptions improved from 1/4 to 3/4 and evidence coverage from 0/12 to 8/12. All 106 RAG citations were allowlisted, no excluded-source leak or provider degradation occurred, but the absolute quality gate failed and `insufficient_evidence` remained 0/4. These are bounded diagnostic results, not an open-text accuracy or model-quality claim. See the [sanitized A/B result](docs/issue-review-v1-result-20260907.md).

Evaluation also includes an 80-case directive regression, a 100-case synthetic natural-Chinese dev/test dataset, a 50-case developer-visible challenge-v2, a 14-case original multi-document complex acceptance suite, and a generated 24k-character long-text smoke. None is presented as a human blind test or production accuracy. See [the resume-readiness boundary](docs/resume-readiness.md).

An experimental first-stage bounded evidence-repair Agent is wired behind a feature flag that defaults to off. It uses LangGraph 1.2.11 `StateGraph` and an application-level JSON tool protocol—not native provider `tool_calls`—for `READ_SPAN`, `PATCH_RECORDS`, and `ABSTAIN`. Its safety boundaries have automated and Mock coverage. A three-task real-provider development pilot exposed benchmark-label errors, and all three tasks are permanently classified as tuned. The corrected v2 `full × 3` evaluation used the real Provider only for the Agent stage; the main-extraction candidate was synthetically injected from the frozen manifest, so this was not an end-to-end real-model extraction evaluation. It recorded all 90 required executions, including 81 holdout executions, but only 59/90 were strictly correct and the full gate failed: holdout runtime success was 74/81 with seven `read_timeout` failures, recovery was 26/51, and semantic abstention was 24/30. Twelve patches accepted by production validation were wrong against the evaluation oracle; no accepted safety violation was observed. No successful direct-`ABSTAIN` path was observed, so the required three-path coverage gate also failed. These results are failure diagnostics, not an Agent quality, product-benefit, extraction-quality, or resume-readiness claim. There is no multi-agent implementation. The newer Evidence Reviewer is a separate bounded RAG consumer and must not be described as this repair Agent or as multi-agent orchestration. See the [first-stage boundary](docs/review-agent-phase1.md), [pilot record](docs/review-agent-pilot-20260906.md), [sanitized v2 full checkpoint](docs/review-agent-v2-full-checkpoint-20260906.md), [v2 frozen manifest](data/agent-acceptance-v2/manifest.json), and [offline runner](scripts/run_agent_acceptance.py).

A separate default-off Evidence Investigator uses provider-native function calls to search and read authorized frozen snapshots before submitting one candidate or abstaining; deterministic promotion remains the only path from an untrusted candidate to an issue. At commit `618ca991`, two consecutive DEV runs scored 8/8 with the same reproducibility fingerprint. The first frozen 10-case holdout scored 8/10 (TP 4, FN 1, TN 4, FP 1; 80% precision and recall), with all cases terminating normally. Promotion accepted no wrong Agent candidate, while the deterministic main path still produced one false positive. This small fixture is sealed against further tuning and is not an open-text, production-quality, or multi-agent claim. Entry points are the [frozen fixture and protocol](data/evaluation/evidence_investigator_live/README.md), [read-only fixture validator](data/evaluation/evidence_investigator_live/validate_fixture.py), [live HTTP runner](scripts/run_evidence_investigator_live.py), [two-run DEV checker](scripts/check_evidence_investigator_dev_pair.py), and [sanitized evaluation summary](docs/evidence-investigator-live-evaluation.md). Local run artifacts remain Git-ignored; the repository does not commit credentials, service addresses, prompts, provider payloads, or story bodies.

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
