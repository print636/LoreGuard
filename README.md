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

The document library groups stored versions by filename and supports filters,
on-demand line-numbered reading, literal text search, and comparison of a selected
historical version with the current one. It is read-only: opening a document never
changes its narrative authority or publishes a draft. DOCX previews show the
imported text, not the original Word layout. See the
[document-library V1 contract](docs/document-library-v1.md).

Report evidence also opens a paged, read-only view of its frozen original
context. Formal issues, review clues and unverified model proposals stay
distinct; opening their source does not validate a conclusion. Confirmed
character traits may refer to an older source run. If only the verified frozen
excerpt survives, the reader says that full context is unavailable instead of
substituting the current draft. See the
[report evidence reader V1 contract](docs/report-evidence-reader-v1.md).

An in-product Simplified Chinese [usage guide](docs/product-help-v1.md) opens
in place from the project center, check workspace and report. Six topics explain
first-review paths, document identity, character baselines, review selection,
evidence and execution coverage. Reading it does not navigate away, clear drafts,
call APIs or invoke a model; Escape or the close button returns to the trigger.

In authenticated mode, [tab-local draft recovery](docs/session-draft-recovery-v1.md)
preserves edited quick-text inputs and unsent formal-issue comments in
`sessionStorage`. Only the original verified account can restore them through the
product; explicit logout clears that account's cache. This is unencrypted local
storage, not cloud autosave, and never stores passwords or provider settings.
Anonymous local mode does not enable recovery; storage failures remain visible.

The audit view now uses a [paged run catalog](docs/run-history-v1.md) with status
filters and 20 metadata rows per page. Paging does not change the open run, and
recorded token counts do not establish model participation or complete coverage.
Other baseline-guidance consumers still use the legacy all-run metadata path;
this is not a claim that the whole workspace is free of unbounded list reads.

[Per-file import recovery](docs/import-queue-v2.md) keeps successful uploads,
failed files and uncertain responses distinct. Each file has its own role and
scope; manual replay of the same upload uses a server receipt rather than
creating another version. A created project is reused for remaining files;
an uncertain project-creation response never triggers automatic re-creation.
Inspect the queue, then explicitly enter the created project. File objects live
only in page memory, not in tab-local draft storage. Importing does not confirm
authority, publish chapters or start an analysis.

[Project metadata editing](docs/project-metadata-v1.md) changes only the name
and description. Revision checks prevent silent overwrites; a conflicting edit
can be explicitly merged with the latest metadata before a separate save.
Document content, authority and frozen reports remain unchanged.

For character-OOC coverage, the recommended path is `baseline_build` -> human confirmation of character-profile candidates -> `draft_review`:

1. Import documents and confirm each document's role, publication status and narrative scope. An explicit AI inference request may propose these fields with source evidence, but the server always stores it as `origin=model_inferred`, `resolution_state=inferred`, and unresolved authority. It never auto-confirms or promotes model output.
2. Run `baseline_build` to freeze only confirmed canon, character profiles and published history as background. Draft, unconfirmed and retired inputs are excluded with visible reasons. If the default-off character-consistency stage is enabled, it may produce candidates; a human must confirm or reject them.
3. Run `draft_review` with confirmed draft or in-review chapters as targets. The server—not the client—derives compatible confirmed background, and freezes `target`/`background` roles with document versions and context snapshots. Background-only findings are not presented as draft issues.

For a project that has only draft chapters and no active formal setting or published history, the guided page also offers a chapter-only first review. The author still confirms each selected chapter's draft status; the request explicitly freezes the target IDs and opts into `no_formal_context_expected`. If formal material appears before submission, the server rejects the run instead of silently changing its scope. Model extraction can still run, but without formal background or applicable confirmed character traits, this is **not** a completed character-OOC review. Confirmed traits can remain active after their source document is retired; actual coverage is reported per run.

Publishing a draft chapter as history is a separate, explicit author action in the document-context workbench; finishing an analysis never publishes it. The confirmation explains that a report is not a publication license, and the author may publish despite absent or partial review coverage. Publication affects future runs only; earlier input snapshots and reports stay frozen. After the first chapter is published, a new draft can still receive a limited review against confirmed formal background without a complete character baseline. Such a run must not be described as a completed character-OOC review.

The current OOC foundation covers six major capability families: core
personality, stable preference, speech pattern, value or behaviour boundary,
relationship attitude toward a concrete person or group, and commitment to a
concrete long-term goal. Formal promotion requires source-bound baseline and
draft evidence; repeated-behaviour dimensions require two independently
verified contrary events. Relationship and goal traits bind both a neutral
axis and a concrete object so different attitudes, people, or goals cannot be
collapsed by a shared label. Confirmed formal character profiles may also
authorize explicit aliases; drafts cannot grant themselves aliases, and
ambiguous, cyclic, or main-name-conflicting aliases fail closed.

For new `core_personality` candidates, the character workbench now lets the author
choose or create a project-scoped, immutable v1 comparison axis before confirming
the candidate. Each axis has an author-written positive proposition; the author
must explicitly map the candidate's raw direction as `same` or `opposite` to
that proposition. Uncertain mappings remain pending. Axis creation and
candidate confirmation are separate actions;
the binding affects future runs only. The model's raw `trait_key` and source
evidence remain visible. A clean, single-target draft extraction can match a
different raw label to the approved axis, while reuse of one source line for two
axes is treated as ambiguous. Older confirmed axes without an author-approved
direction stay active as character facts, but new runs mark their same-axis
directional coverage partial until the author reviews each mapping. Ordinary
analysis and legacy unbound candidates retain their old path.
See the [approved-axis v1 contract](docs/character-approved-axis-rfc.md) for
the exact API and limits. This has not passed a separate cross-story real-model
quality gate.

`value` and `behavior_boundary` axes also have an experimental author-approved
baseline binding path: an author supplies an object key, applicability scope,
positive proposition, and explicit scope acknowledgment for a verified formal
candidate. The server checks the frozen source, object identity, axis version,
and hashes. Set `CHARACTER_SCOPED_AXIS_DRIFT_V1=true` to enable this
default-off path for an isolated experiment. New runs can then nominate a draft observation through a clean,
single-target extraction bound to that exact frozen object. A separate model
review must cite the current draft and confirm both object and situation before
any scoped-axis conflict can be reported; a different or unclear situation is
not promoted as an OOC issue, and one action alone cannot prove drift. This is
an **experimental review path**, not a measured cross-story OOC accuracy claim:
actor attribution, model judgments, and incomplete coverage can still miss
valid cases, so authors must inspect both evidence spans and coverage status.

Character-review evidence is now projected from the reviewer's validated
`B/C/G/X/P` handles back to the exact frozen spans used in that call. A completed
review whose handles cannot be resolved to both baseline and current evidence
fails closed and cannot become a formal issue; the report never substitutes
the first available scene. Repeated-behaviour OOC review sends a bounded pool
of source-diverse observations; the model must identify and cite exactly two
current observations and return a structured independent-event judgment. A
second, narrowly prompted model call then receives only those two C spans and
must independently classify their event identity before formal promotion.
Failure, invalid citations, budget/deadline exhaustion, or a non-different
answer stays in review. Same-document nearby spans additionally require an
explicit time/event boundary at the start of the later evidence line, even if
both model calls label them independent.
When the first review is valid but the event-identity pass cannot complete,
the clue uses a separate `first_pass_review_citations_v1` binding and displays
only the exact first-pass B/C selection; it does not publish an event-identity
result or become a formal issue.
A repeated description of one event therefore
stays in review instead of being counted twice. These are evidence-integrity
contracts, not an open-story accuracy result; see the
[v2 development checkpoint](docs/character-ooc-evidence-binding-v2-20260927.md).

An additional default-off `CHARACTER_EXPLANATION_REVIEW_V1` path reviews
bounded candidate windows from the run's exact frozen sources for growth,
disguise/role, temporary state/pressure, and possible foreshadowing. Eligible
source text is exhaustively segmented while it fits the stage bound; lexical,
axis, actor, and locality signals only rank candidates after that bound is
exceeded, and the case then becomes partial. This is **not** embedding RAG.
The current response contract is `character-explanation-review-v2` with
`character-explanation-review-prompt-v2`: every response echoes the schema and
a server-generated digest of the frozen request. An explicit `irrelevant`
result emits no `G`, `X`, or `P`; malformed irrelevant slots fail closed.
The reviewer can emit `G` (growth/causal bridge), `X`
(applicable exception), or `P` (possible explanation) evidence. Explicit
same-draft disguise or temporary-state evidence may become `X`, but a draft
cannot certify its own growth into `G`; ambiguous or foreshadowing material
stays `P`, as does non-explicit bounded inference. Every semantic support row
is bound server-side to the stable current-observation IDs selected from its
strict `C` allowlist. A `P` bound to a `C` used by the final contradiction,
incomplete discovery, provider/validation failure, or a
disagreement between semantic `G/X` support and the final contradiction
verdict keeps the case in review rather than promoting a formal issue. Formal
issues and review clues remain separate in counts, export, and feedback. See
the [explanation-review contract](docs/character-ooc-explanation-review-v1.md).
This path is default-off and has no published open-text accuracy result or
completed sealed real-model holdout. While it is off, any eligible character
drift candidate remains a review clue with `explanation_coverage=not_run`;
the run cannot publish that candidate as a formal character issue or claim
complete character-review coverage.

An author-confirmed character trait remains active even if its source document
is later retired or reclassified as reference material. The character workbench
offers an explicit, audited per-trait withdrawal with confirmation; it excludes
the trait from future runs without changing historical run snapshots. Withdrawn
traits remain visible in the profile archive and cannot be silently restored.

Draft preference review currently auto-binds only unambiguous, unquoted direct
statements or explicitly attributed self-reports. Quoted, conditional, staged,
or retrospectively corrected speech is conservatively marked as incomplete
coverage rather than promoted to a definite contradiction. This intentionally
loses some valid dialogue recall; it is not a general speaker-understanding
claim or a new real-model OOC accuracy result.

For draft actions, a narrow source-bound check rejects obvious non-occurrence,
rumor, hypothetical, and possessive-other-actor claims before they become
character observations. It does not resolve complex cross-sentence meaning.
When the optional character-consistency stage is enabled, its primary signal
extractor now defaults to the V2 complete-source-line echo prompt. Set
`CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2=false` to use the earlier prompt for a
deployment or comparison. The evidence binder is unchanged, and this prompt
default is not a claim that character-OOC review passes on open stories.
An opt-in, default-off source-excerpt repair can restore a single model-quoted
draft action to its exact full source line only when both the excerpt and full
line independently pass the existing binding checks; it never relaxes formal
baseline evidence requirements. Incomplete model packages still yield partial
coverage rather than silently recycling a provisional record into OOC issues.

`CHARACTER_DRAFT_ACTOR_REVIEW_V1` 是另一个默认关闭的实验门，只复核待审新稿中其余绑定检查均已通过、仅主体归属仍待判断的候选。服务端先作结构预筛并绑定冻结原文窗口；每个信号抽取逻辑周期至多使用一次批量调用要求严格 JSON，一次分析含多个分块或 targeted 周期时总调用可多于一次。模型返回 `supported` 后仍须通过服务端的来源复核和 `source_context_veto`。安全、来源完整的 `uncertain`／`source_context_veto` 最多进入 provisional 待复核线索，不进入正式问题、计数、反馈或导出；有效的显式 `rejected` 保持拒收，协议、预算、deadline 或传输失败则 fail closed、保留部分覆盖且不产生 provisional 线索。该门不审查正式基线，也不把一次行为证明为 OOC；真实模型目前只有一次 7-window 的已知 DEV smoke，不是盲测或准确率结果，详见[开发检查点](docs/guided-review-ooc-dev-checkpoint-20260927.md)。

`CHARACTER_TARGET_BOUND_DRAFT_REVIEW_V3` 是与旧 V2 互斥、默认关闭的受限语义发现开关；为兼容现有部署保留了开关名，当前主链使用 request/response/batch V4 与 prompt-v8。定向抽取没有留下可用记录时，服务端只从冻结新稿中白名单化普通显式主语、旁白明确归属和严格相邻的零主语分句，再交给独立模型逐项判断主体、事实性、语义轴、对象关系、目标方向及后续更正；模型不能自由搜索全文或回写事实。V4 会为每个 proposal 投影 `object_binding_mode`：若未规范化的 `target.key_object` 在该事实分句中按原始 Unicode 码点恰好出现一次，`matches_target` 必须返回 null offset，服务端只在主体、事实性、statement、axis、方向、更正、basis 和来源等其他门全部通过后绑定这一个冻结字面量；这只是坐标绑定，不是语义证明。`broader`/`narrower`、零次或多次出现仍必须返回合法的事实分句区间，且显式区间在存在字面量时必须完整覆盖至少一次出现；服务端不会修正错误的非空区间，也不会为 V4 的 object-span 失败重放同一模型请求。作者批准的 value / behavior-boundary scoped axis 仍严格使用 `object_relation=not_applicable` 和 null offset。扁平响应、自造枚举、错误坐标、超时、拒绝服务、`uncertain`、结构错误、来源或摘要不匹配一律失败关闭并保留部分覆盖。该协议扩大常见场景召回，不是开放文本质量或生产可用性声明。

An incomplete, parseable primary draft response may also produce a separate,
read-only list of provisional clues. Each clue is rechecked against the run's
frozen source and is visibly labeled as an unverified model proposal. For the
ordinary package-validation path, a later clean package supersedes that clue;
source-bound actor-review uncertainty follows the stricter rule above. These
clues never enter facts, OOC issues, issue
counts, feedback, Markdown exports, or visualizations; partial coverage remains
partial. `GET /api/v1/analysis-runs/{run_id}/provisional-clues` returns at most
64 workspace-authorized clues with an explicit truncation flag. The ordinary
diagnostics endpoint does not expose their source text.

角色审查结果分为“正式一致性问题”和“待复核线索”。角色漂移只有最终判为冲突、已确认设定与当前新稿均有可核对的原文证据时，才进入正式问题列表、数量统计和报告导出。单次反向行为，或虽定位到双方原文但情境、材料或复核结论不足的情况，进入待复核线索；无法核对原文的提案不展示为结论。`GET /api/v1/analysis-runs/{run_id}/issues` 只返回正式问题，`GET /api/v1/analysis-runs/{run_id}/review-clues` 单独返回角色审查线索；上文的 `provisional-clues` 仍是未通过完整抽取校验的模型提案，不参与问题计数。旧数据升级时，不满足旧版冲突格式的角色条目保守降为线索；旧线索若无法与冻结原文核对，不展示正文，接口通过 `unavailable_count` 提示。历史线索超过安全扫描上限时，`scan_limited` 提示还有未核对条目；这两种情况都建议重新分析。

已发布历史中的价值观与行为边界记录可选择开启默认关闭的 `CHARACTER_HISTORY_SEMANTIC_REVIEW_V1`：仅当候选只因词面极性门失败、其余冻结证据绑定检查均通过时，才由受限 AI 阅读完整原文行，复核行为主体、对象、现实性、语义方向及后文更正。拒绝、不确定、模型或预算失败不会晋升该记录；该能力不能替代作者确认角色档案，也不能修正正式设定抽错比较对象的问题。真实模型 OOC 质量尚未完成验收，详见[角色一致性合同](docs/character-consistency-v1.md)。

Context inference is single-document and limited to 30,000 characters and 2,000 lines; larger inputs require manual context assignment. It is not a story rewrite, bulk classifier, authority decision, or production-accuracy claim. See the [V1 workflow and API contract](docs/guided-review-batch-v1.md) and the [full Chinese guide](README.zh-CN.md).

Relevant API entry points are:

- `POST /api/v1/projects/{project_id}/documents/{document_id}/narrative-context/inference` with `expected_revision` for a model suggestion only;
- `POST /api/v1/projects/{project_id}/documents/{document_id}/narrative-context/revisions` for the human-reviewed revision;
- `POST /api/v1/projects/{project_id}/analysis-runs` with `mode=baseline_build|draft_review|full_review`, optional draft targets and a sensitivity level;
- `GET /api/v1/analysis-runs/{run_id}` to inspect `review_batch` coverage and frozen `input_documents[].batch_role`.
- `GET /api/v1/analysis-runs/{run_id}/review-clues` to inspect source-checked character review leads separately from formal issues; the response includes `items`, `truncated`, `unavailable_count` and `scan_limited`.
- `GET /api/v1/analysis-runs/{run_id}/export.md` to download a completed run's evidence-first Markdown report with current feedback labels. The export uses all issues, regardless of the browser's current filters, and is workspace-scoped.
- `GET` / `POST /api/v1/projects/{project_id}/character-trait-axes` to list or create immutable project axes with a positive proposition; `GET /{axis_id}` reads one exact axis and `POST /{axis_id}/positive-proposition` authors the one-time definition for a legacy axis.
- Core-personality confirmation with an axis additionally requires `axis_alignment=same|opposite` and the expected positive-proposition hash. A legacy confirmed candidate uses `POST /api/v1/projects/{project_id}/characters/{character_key}/profile-candidates/{candidate_id}/alignment` for a separately audited mapping; neither action changes an old run.

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
character-profile candidate extraction and draft-drift review; the nested
`CHARACTER_EXPLANATION_REVIEW_V1` gate additionally enables bounded semantic
explanation discovery inside that stage;
`ENABLE_ISSUE_EVIDENCE_REVIEW` enables the post-rule chat annotation; and
`ENABLE_EVIDENCE_INVESTIGATOR` enables the separate
provider-native function-calling loop. `ENABLE_EMBEDDINGS` enables only the
embedding client and does not enable chat by itself. The Reviewer and Investigator
also require their documented embedding/PostgreSQL configuration. All six top-level
switches and the nested explanation-review gate default to false. The UI connection test is a separate explicit user
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
baseline candidate selection on a preceding build. Product defaults are now
finite, quality-first admission ceilings of 400k per run, 2m daily, 300k for
the character stage and 26k per signal; the character stage defaults off.
Explicitly lower or exhausted budgets remain `partial` rather than being
presented as an absence of problems. These ceilings are not usage targets and
do not guarantee complete processing for arbitrary long stories.
The latest runs in that checkpoint used a locally raised 10m daily limit.
The [author-approved axis v1](docs/character-approved-axis-rfc.md) was
implemented afterward; it does not retroactively change the frozen transfer
Oracle or those runs. None of these results establishes cross-story or
production OOC quality. See the checkpoint for the full chronology.
A separate [`character-ooc-alpha-v1`](data/character-ooc-alpha-v1/README.md)
fixture and [strict acceptance contract](docs/character-ooc-alpha-acceptance.md)
cover the current six capability families and report-layer separation. They
are developer-visible synthetic DEV material that may be used for prompt,
retrieval, extraction, rule, and model-configuration changes—not a human-sealed
test, production-quality result, or evidence of open-text generalization.
The [sanitized Alpha v20 checkpoint](docs/character-ooc-alpha-v20-checkpoint-20260929.md)
records one V4 / Prompt V8 real-model workflow: all eight known DEV cases and
20/20 required gates matched, with no independent repeat trial. It remains a
developer-visible synthetic result, not blind, production, or open-text evidence.
The protocol and offline tools for the next step are documented in
[Character OOC human closed evaluation V1](docs/character-ooc-closed-evaluation-v1.md).
The 36-case public freeze and separate A/B offline annotation pages now exist
outside the repository. Two real humans have not yet completed the independent
annotations or adjudication, and no real-model sealed run, formal score, or
failure-driven repair has been completed. Formal scoring accepts only the
runner-native sealed HTTP report bound to the exact frozen run config, never a
standalone extracted prediction artifact; this is not a completed blind test.
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

An opt-in [semantic scope review](docs/semantic-scope-review-rfc-20260925.md)
now lets a bounded model reviewer assess same-line subject and label carryover
against frozen source clauses. The existing default-off environment flag names
are retained for deployment compatibility, while enabled runs report the
`semantic-scope-v6` / `character-scope-review-v2` /
`character-scope-review-prompt-v4` identity. Its
[precise support binding](docs/character-support-binding-v1.md)
keeps separately supported clauses on one line as independently reviewable
author decisions and distinguishes the target clause from carryover context in
the UI. Invalid new bindings block review; legacy candidates retain the older
whole-line display. Both changes are default-off and do not alter the frozen
evaluation gate. They establish evidence identity and failure handling, **not**
open-text semantic accuracy or production readiness. A [single real-model DEV
checkpoint](docs/character-support-binding-dev-checkpoint-20260926.md) completed
the character baseline but stopped at candidate selection (3/5 frozen
selectors uniquely matched); no draft or OOC case was evaluated.
Two further [same-build DEV repetitions and diagnostic boundaries](docs/character-support-funnel-dev-checkpoint-20260926.md)
also stopped before draft review despite complete baselines (4/5 and 3/5
unique frozen selector matches). The implemented V2 funnel was replayed
offline against three isolated snapshots produced by real-model runs; it was
not emitted in the original online reports or counted in their scores.

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
