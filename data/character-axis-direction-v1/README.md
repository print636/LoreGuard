# Character axis direction v1 (developer-visible)

This fixture reuses the exact story bytes and oracle bytes of
`character-axis-challenge-v2`. Its separate, hash-pinned evaluation plan adds a
proposed positive proposition and expected baseline axis direction. These are
developer-authored test intentions, not an actual user's approval or independent
human annotation. The v2
manifest, review plans, oracle, runner, and historical scores are unchanged.

`dev` and `transfer` have both been visible during development. Neither is a
blind holdout or evidence of production accuracy. The story and author plan
may be read before a model run; the oracle must be parsed only after all model
calls have ended.

An actual human reviewer must inspect each candidate in the product UI against its exact
source clause and the axis proposition. The author chooses `same`, `opposite`,
or `uncertain` relative to the candidate's own raw trait label and polarity.
`uncertain` leaves the candidate pending. The evaluator must never compute an
author choice from the expected axis polarity, the v2 raw-polarity selector, or
oracle labels. An explicit author decision is required for every confirmation.

The first runner increment provides a no-network preflight and isolated
baseline/author-confirmation stages. Its decision-file declaration is a
self-attestation, not proof that a human or independent annotator performed the
review. It does not issue a draft quality score.
All live model calls require an explicit `prepare` command; no provider call is
made by `preflight` or `confirm`.

## Isolated live service (required for `prepare` and `confirm`)

The runner rejects the ordinary port 8000 and non-loopback URLs. A new,
file-backed SQLite database must be explicitly provisioned as a direct child
of `Desktop/dev/artifacts/character-axis-direction-v1` (the path is derived
from this repository's `Desktop/dev/projects/LoreGuard` layout). The provisioner
refuses an existing file and writes a random, persistent evaluation DB ID to
the new database; it does not start a service, migrate product tables, or call
a model. For example, from the repository root in PowerShell:

```powershell
$evalDb = 'C:\Users\dell\Desktop\dev\artifacts\character-axis-direction-v1\dev-eval.sqlite3'
.venv\Scripts\python.exe scripts\provision_character_axis_isolation.py --db-path $evalDb
```

Record the returned `eval_db_id`. Choose and retain a separate random instance
UUID for this service (`[guid]::NewGuid().ToString()` in PowerShell). Configure
the isolated process with an absolute SQLite `DATABASE_URL` pointing to that
exact file and `EVAL_ISOLATION_INSTANCE_ID` set to the retained UUID, then
start it on a dedicated non-8000 loopback port. First commit the intended code
and require a clean worktree: the runner checks its Git HEAD and cleanliness,
and the service must report that same full revision and matching service
artifact hash. Do not attempt a paid `prepare` from the current uncommitted
development tree. For example:

```powershell
if (git status --porcelain) { throw 'A clean committed build is required' }
$env:LOREGUARD_BUILD_REVISION = (git rev-parse HEAD).Trim()
$env:DATABASE_URL = 'sqlite:///' + ($evalDb -replace '\\', '/')
$evalInstanceId = [guid]::NewGuid().ToString()
$env:EVAL_ISOLATION_INSTANCE_ID = $evalInstanceId
$env:ENABLE_CHARACTER_CONSISTENCY = 'true'
$env:CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2 = 'true'
$env:CHARACTER_SIGNAL_CORE_SCOPE_PROMPT_V3 = 'true'
$env:CHARACTER_SIGNAL_SUPPORT_ID_V4 = 'true'
$env:CHARACTER_SIGNAL_SEMANTIC_SCOPE_V5 = 'true'
$env:CHARACTER_SIGNAL_SCOPE_REVIEW_V1 = 'true'
# Configure an authorized model provider securely; never put a key in reports.
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Retain the instance UUID for every later `confirm` or process restart; generating
a new one changes the bound instance identity. Before any paid `prepare`, verify
the isolated process has an authorized model provider, the character consistency
feature and V2/V3/V4/V5/scope-review flags above are effective, and the
service build revision matches the clean runner checkout. The runner enforces
these capabilities and provenance again before creating a project.

The service refuses to start in evaluation mode if its actual SQLite connection
does not use the provisioned file. Its read-only
`/api/v1/evaluation/isolation-identity` endpoint checks the
connected file via SQLite `PRAGMA database_list`, the dedicated directory, and
the database's persisted sentinel. It returns only opaque IDs/hashes and
`fresh_for_prepare`, never a DB path, URL, credential, or story text. The
identity check must open the existing SQLite file before `PRAGMA` can verify
it; for an unmarked daily database, this is a read-only rejection, not
provisioning or migration. The ordinary `/health` contract remains unchanged.

For both live phases, supply `--base-url http://127.0.0.1:8765`,
`--expected-eval-db-id <provisioned UUID>`,
`--expected-eval-instance-id <retained instance UUID>`, and
`--expected-eval-db-path <absolute $evalDb path>`. The runner independently
hashes the expected path, compares all three identities to the service before
any project or model call, and rechecks after writes. The prepare phase also
requires `fresh_for_prepare=true` (no existing projects). `confirm` permits
the same database after its baseline project exists but requires the identity
frozen in its checkpoint. Use a separately provisioned database and instance
for `transfer`.

If `prepare` is interrupted before it emits a checkpoint, do not simply rerun
it against the now non-fresh database: model charges or partial project writes
may already exist, and there is no safe automatic resume. Inspect the isolated
run and provider usage, then deliberately provision a new evaluation database
for a new attempt. No automatic deletion is performed.

Use `python scripts/run_character_axis_direction_live.py --preflight-only` to
check both suites' hashes and contracts without HTTP. The `prepare` phase
requires a clean, pinned build and an isolated service with character
consistency, V4 assertion IDs, V5 scope, and scope review enabled. It also
requires `--allow-provider-call`, `--suite dev|transfer`, and a new
`--output-json artifacts/<name>.json` path, plus the isolation arguments above.
This phase creates a new project,
uploads only the three background documents, runs a real baseline, and emits a
sanitized checkpoint with candidate IDs. It does not upload the draft.

After inspecting the candidates in that project, the reviewer writes a separate
JSON file under `artifacts/` with this shape (all keys and candidate IDs must
come from the checkpoint and the UI):

```json
{
  "schema_version": "character-axis-direction-decisions-v1",
  "project_id": "PROJECT_ID",
  "baseline_run_id": "BASELINE_RUN_ID",
  "reviewer_declaration": "I reviewed each actual candidate, its frozen source clause, and the axis positive proposition without consulting the scoring oracle.",
  "choices": [
    {"candidate_key": "dev_partner_signature", "candidate_id": "ACTUAL_CANDIDATE_ID", "decision": "confirm", "axis_alignment": "opposite"}
  ]
}
```

The real decision file must include one choice for every target. Core-axis
choices use `same` or `opposite` for confirmation; `defer` with `uncertain`
leaves the candidate pending. The preference target confirms with
`axis_alignment: null`. The runner checks the persisted axis polarity against
the frozen expectation only after confirmation; it never generates the human
choice. Only axes used by an explicit `confirm` choice are created, so deferred
targets do not cause unused axes to be written. Run the `confirm` phase with `--checkpoint-json`,
`--author-decisions-json`, the same isolation arguments, and a new `--output-json` path. This phase does not
call the Provider and reports `quality_scored=false` even when its API contract
passes. The report separates `api_decision_integrity` (whether the explicit
choices were safely persisted or deferred) from `plan_agreement` and
`coverage`; a valid defer or author/plan disagreement is not an API failure.
If confirmation is interrupted after some writes, use the same checkpoint and
decision file with a new output path. The runner rechecks the frozen candidate
inventory and each existing review chain, then safely replays matching
confirmations with deterministic idempotency keys. A changed review chain,
candidate, or axis mapping stops the run. Old v2 scores remain untouched.
