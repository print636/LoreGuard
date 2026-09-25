# LoreGuard

LoreGuard is an evidence-first narrative consistency review platform for game writers and narrative designers. It extracts versioned facts and events from authorized story material, detects deterministic continuity conflicts, and optionally augments its baseline extractor with a validated OpenAI-compatible provider. Provider failures fall back to the baseline; model and baseline records are deduplicated and bound to source lines. A separate default-off Evidence Investigator uses provider-native function calls to search and read authorized snapshots before submitting one untrusted candidate or abstaining; deterministic promotion remains the only path from that candidate to an issue.

See [README.zh-CN.md](README.zh-CN.md) for the full guide. Deployment
credentials are read only from server-side environment variables or
mounted secret files and must never be committed.

Authenticated accounts may instead save one OpenAI-compatible chat Provider.
Those API keys are AES-256-GCM encrypted, never echoed, and analysis runs freeze
a non-secret provider revision before Celery dispatch. See the
[BYOK V1 security and rotation contract](docs/account-model-provider.md).

## Guided review workflow

The recommended path is `baseline_build` -> human confirmation of character-profile candidates -> `draft_review`:

1. Import documents and confirm each document's role, publication status and narrative scope. An explicit AI inference request may propose these fields with source evidence, but the server always stores it as `origin=model_inferred`, `resolution_state=inferred`, and unresolved authority. It never auto-confirms or promotes model output.
2. Run `baseline_build` to freeze only confirmed canon, character profiles and published history as background. Draft, unconfirmed and retired inputs are excluded with visible reasons. If the default-off character-consistency stage is enabled, it may produce candidates; a human must confirm or reject them.
3. Run `draft_review` with confirmed draft or in-review chapters as targets. The server—not the client—derives compatible confirmed background, and freezes `target`/`background` roles with document versions and context snapshots. Background-only findings are not presented as draft issues.

For new `core_personality` candidates, the character workbench now lets the author
choose or create a project-scoped, immutable v1 comparison axis before confirming
the candidate. Axis creation and candidate confirmation are separate actions;
the binding affects future runs only. The model's raw `trait_key` and source
evidence remain visible. A clean, single-target draft extraction can match a
different raw label to the approved axis, while reuse of one source line for two
axes is treated as ambiguous. Legacy unbound candidates retain their old path.
See the [approved-axis v1 contract](docs/character-approved-axis-rfc.md) for
the exact API and limits. This has not passed a separate cross-story real-model
quality gate.

Context inference is single-document and limited to 30,000 characters and 2,000 lines; larger inputs require manual context assignment. It is not a story rewrite, bulk classifier, authority decision, or production-accuracy claim. See the [V1 workflow and API contract](docs/guided-review-batch-v1.md) and the [full Chinese guide](README.zh-CN.md).

Relevant API entry points are:

- `POST /api/v1/projects/{project_id}/documents/{document_id}/narrative-context/inference` with `expected_revision` for a model suggestion only;
- `POST /api/v1/projects/{project_id}/documents/{document_id}/narrative-context/revisions` for the human-reviewed revision;
- `POST /api/v1/projects/{project_id}/analysis-runs` with `mode=baseline_build|draft_review|full_review`, optional draft targets and a sensitivity level;
- `GET /api/v1/analysis-runs/{run_id}` to inspect `review_batch` coverage and frozen `input_documents[].batch_role`.
- `GET /api/v1/analysis-runs/{run_id}/export.md` to download a completed run's evidence-first Markdown report with current feedback labels. The export uses all issues, regardless of the browser's current filters, and is workspace-scoped.
- `GET` / `POST /api/v1/projects/{project_id}/character-trait-axes` to list or create immutable project axes; candidate confirmation accepts `approved_axis_id` and `expected_axis_version` together for `core_personality`.

An omitted analysis body or `{}` retains the legacy `full_review` behavior for existing clients. Retry preserves the frozen batch; recheck advances only the logical draft targets to their current active versions and re-derives the background.

Quick start:

```bash
docker compose up --build
```

The default stack is an explicit local/demo deployment with one durable
anonymous workspace. Real account registration and personal-workspace
isolation are enabled with `AUTH_MODE=required` and a unique server-only
`AUTH_SECRET_KEY` of at least 32 characters. A public deployment must also set
`DEPLOYMENT_ENVIRONMENT=production`, secure cookies, and an exact HTTPS CORS
origin. Required-auth users can change their password and inspect or revoke
their own active sessions at `/app/settings/account`; a password change keeps
the current session and revokes the others. The development Compose file
exposes ports and development database credentials and must not be published
unchanged. Use the fail-closed production overlay only behind an external HTTPS
reverse proxy; see [`docs/production-deployment.md`](docs/production-deployment.md)
and [`docs/auth-security-contract.md`](docs/auth-security-contract.md).

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
a standalone caller; `ENABLE_CHARACTER_CONSISTENCY` enables frozen-source
character-profile candidate extraction and draft-drift review;
`ENABLE_ISSUE_EVIDENCE_REVIEW` enables the post-rule chat annotation; and
`ENABLE_EVIDENCE_INVESTIGATOR` enables the separate
provider-native function-calling loop. `ENABLE_EMBEDDINGS` enables only the
embedding client and does not enable chat by itself. The Reviewer and Investigator
also require their documented embedding/PostgreSQL configuration, and all six
switches default to false. The UI connection test is a separate explicit user
action that makes one minimal chat request; merely loading the page never calls a
model.

The character-consistency stage also has a bounded, developer-visible v26
real-model checkpoint: three independent HTTP workflows passed all 12 frozen
gates after candidate-confirmation and evidence-kind hardening. This is not a
blind test, an open-text generalization result, or a production-quality claim. See the
[sanitized checkpoint](docs/character-consistency-live-checkpoint-20260922.md).
An additional [OOC challenge](docs/character-ooc-challenge-checkpoint-20260923.md)
uses a different original version-event story. Its first full diagnostic did
not pass: all three formal character candidates were confirmed, but baseline
and draft coverage were partial, one intended conflict was missed, one growth
case remained unverifiable, and the single emergency behavior was not falsely
promoted. It is a developer-visible failure diagnostic, not a quality gain.
The follow-up added evidence-bound adjacent-pronoun handling, directional
trait-key rejection, and a separately labeled diagnostic candidate-review
path. An earlier build passed three strict runs of the known Demo, but later
object-identity and actor-binding changes required fresh validation. The next
build passed only 1/3 strict Demo trials at a locally raised 100k character-stage
and 22k per-signal limit. On the latest diagnostic build, independent strict
three-trial sets passed 3/3 at a 150k stage, 40k signal and 200k per-run limit;
3/3 at a 100k stage and 40k signal limit; and 2/3 at a 100k stage and 22k
signal limit, in that order. These small, developer-visible observations do
not prove that a larger budget resolves every failure. The partial draft in
the latest 22k set had rejected model records; its new, content-free
token-admission events were empty. A different frozen OOC story stopped at
baseline candidate selection on a preceding build. Product defaults remain
100k per run, 100k daily, 60k
for the character stage and 22k per signal; the character stage defaults off.
The latest runs in that checkpoint used a locally raised 10m daily limit.
The [author-approved axis v1](docs/character-approved-axis-rfc.md) was
implemented afterward; it does not retroactively change the frozen transfer
Oracle or those runs. None of these results establishes cross-story or
production OOC quality. See the checkpoint for the full chronology.
The later [fixed DEV v2 prompt A/B checkpoint](docs/character-axis-v2-dev-checkpoint-20260924.md)
records six independent, developer-visible trials. Fewer evidence-excerpt
rejections did not complete the review workflow: all 30 case evaluations
remained unavailable.
The subsequent [fixed DEV v3 core-label prompt A/B checkpoint](docs/character-axis-v3-dev-checkpoint-20260925.md)
records 2/3 baseline admissions with v3 on versus 0/3 off, but both admitted
trials stopped at candidate uniqueness; no draft cases were evaluated and the
frozen transfer suite was not run.
The [V4 DEV comparison](docs/character-axis-v4-dev-checkpoint-20260925.md)
keeps the support-ID protocol disabled by default after it reduced frozen
candidate coverage. A later [anonymous support-trace diagnostic](docs/character-support-trace-dev-checkpoint-20260925.md)
used three real-model DEV trials: three target clauses were not submitted and
two were submitted but failed the actor-support guard in every trial. All three
baselines remained partial, so these are pipeline observations, not OOC
accuracy results. The guard's current direct-actor contract would reject all
five target clauses if submitted; their cross-clause subject and label scope
still need independent author confirmation, not automatic inheritance.

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

A [September 2026 extraction-format pilot](docs/extraction-format-pilot-20260923.md) records two real-model reruns on one developer-visible three-document sample. A separate [cross-task extraction check](docs/extraction-transfer-check-20260923.md) covers 18 original cases: dev 5/5 positive evidence hits and 3/3 clean negatives; previously used holdout 4/5 and 4/5, with one miss and one false positive. Both are developer-visible diagnostics, not open-text accuracy or a production-quality claim.

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
