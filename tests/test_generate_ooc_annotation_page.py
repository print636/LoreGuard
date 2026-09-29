"""Synthetic-only tests for the offline OOC annotation page generator."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys

import pytest

from scripts import generate_ooc_annotation_page as page
from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_closed_bundle as closed_bundle
from scripts import ooc_eval_contract as contract


DATASET_ID = "00000000-0000-4000-8000-000000000101"
CASE_ID = "00000000-0000-4000-8000-000000000102"
GROUP_ID = "00000000-0000-4000-8000-000000000103"
WORLD_ID = "00000000-0000-4000-8000-000000000104"
BASELINE_DOCUMENT_ID = "00000000-0000-4000-8000-000000000105"
AXIS_ID = "00000000-0000-4000-8000-000000000106"
EVIDENCE_IDS = (
    "00000000-0000-4000-8000-000000000107",
    "00000000-0000-4000-8000-000000000108",
    "00000000-0000-4000-8000-000000000109",
    "00000000-0000-4000-8000-000000000110",
)
TARGET_DOCUMENT_ID = "00000000-0000-4000-8000-000000000111"
BASELINE_STORY = (
    "角色始终守约。\n"
    "</script><script>globalThis.annotationLeak = true</script>\n"
)
TARGET_STORY = (
    "新稿中第一次明确违约。\n"
    "新稿中第二次明确违约。\n"
)


def _fixture(tmp_path):
    case_dir = tmp_path / "bundle" / "cases" / GROUP_ID
    case_dir.mkdir(parents=True)
    baseline_document = case_dir / f"{BASELINE_DOCUMENT_ID}.md"
    target_document = case_dir / f"{TARGET_DOCUMENT_ID}.md"
    baseline_payload = BASELINE_STORY.encode("utf-8")
    target_payload = TARGET_STORY.encode("utf-8")
    baseline_document.write_bytes(baseline_payload)
    target_document.write_bytes(target_payload)
    baseline_digest = hashlib.sha256(baseline_payload).hexdigest()
    target_digest = hashlib.sha256(target_payload).hexdigest()
    selector = {
        "character_key": "沈砚",
        "trait_type": "core_personality",
        "source_document_id": BASELINE_DOCUMENT_ID,
        "source_line_start": 1,
        "source_line_end": 1,
        "origin": "explicit_setting",
        "polarity": "positive",
        "stability": "core",
    }
    author_axis = {
        "display_name": "守约倾向",
        "definition": "角色是否持续履行自己明确作出的承诺。",
        "positive_proposition": "沈砚会履行已明确作出的承诺。",
        "applicability_scope": "仅适用于沈砚本人主动作出的明确承诺。",
        "axis_alignment": "same",
    }
    axis_payload = json.dumps(
        {"candidate_selector": selector, "author_axis": author_axis},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    axis_digest = hashlib.sha256(axis_payload).hexdigest()
    public_payload = {
        "dataset_id": DATASET_ID,
        "cases": [
            {
                "case_id": CASE_ID,
                "split_id": "closed-pilot-v1",
                "group_id": GROUP_ID,
                "world_id": WORLD_ID,
                "documents": [
                    {
                        "document_id": BASELINE_DOCUMENT_ID,
                        "path": f"cases/{GROUP_ID}/{BASELINE_DOCUMENT_ID}.md",
                        "version": 1,
                        "sha256": baseline_digest,
                    },
                    {
                        "document_id": TARGET_DOCUMENT_ID,
                        "path": f"cases/{GROUP_ID}/{TARGET_DOCUMENT_ID}.md",
                        "version": 1,
                        "sha256": target_digest,
                    },
                ],
                "axis": {"axis_id": AXIS_ID, "version": 1, "sha256": axis_digest},
                "evidence_catalog": [
                    {
                        "evidence_id": evidence_id,
                        "document_id": (
                            BASELINE_DOCUMENT_ID if index <= 2 else TARGET_DOCUMENT_ID
                        ),
                        "line_start": index if index <= 2 else index - 2,
                        "line_end": index if index <= 2 else index - 2,
                    }
                    for index, evidence_id in enumerate(EVIDENCE_IDS, start=1)
                ],
            }
        ],
    }
    public = contract.PublicInputBundle.model_validate_json(
        json.dumps(public_payload, ensure_ascii=False)
    )
    public_file = tmp_path / "public-input.json"
    public_file.write_text(
        json.dumps(public_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    setup_payload = {
        "dataset_id": DATASET_ID,
        "public_input_sha256": contract.canonical_sha256(public),
        "cases": [
            {
                "case_id": CASE_ID,
                "group_id": GROUP_ID,
                "target_document_id": TARGET_DOCUMENT_ID,
                "documents": [
                    {
                        "document_id": BASELINE_DOCUMENT_ID,
                        "phase": "baseline",
                        "document_role": "character_profile",
                        "story_scope": "character-bible",
                        "resolution_state": "confirmed",
                        "publication_status": "published",
                        "import_order": 1,
                    },
                    {
                        "document_id": TARGET_DOCUMENT_ID,
                        "phase": "target",
                        "document_role": "chapter",
                        "story_scope": "chapter-12",
                        "resolution_state": "draft",
                        "publication_status": "draft",
                        "import_order": 2,
                    },
                ],
                "candidate_selector": selector,
                "author_axis": author_axis,
            }
        ],
    }
    setup = closed_bundle.PublicAuthorSetupBundle.model_validate_json(
        json.dumps(setup_payload, ensure_ascii=False)
    )
    setup_file = tmp_path / "author-setup.json"
    setup_file.write_text(
        json.dumps(setup.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    freeze_payload = {
        "dataset_id": DATASET_ID,
        "public_input_sha256": contract.canonical_sha256(public),
        "author_setup_sha256": contract.canonical_sha256(setup),
        "case_count": 1,
        "group_count": 1,
    }
    freeze = closed_bundle.ExecutionFreeze.model_validate_json(
        json.dumps(freeze_payload)
    )
    freeze_file = tmp_path / "freeze.json"
    freeze_file.write_text(
        json.dumps(freeze.model_dump(mode="json")),
        encoding="utf-8",
    )
    return (
        public,
        public_file,
        tmp_path / "bundle",
        setup,
        setup_file,
        freeze,
        freeze_file,
    )


def _embedded_data(html: str) -> dict:
    match = re.search(
        r'<script id="annotation-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _inline_javascript(html: str) -> str:
    matches = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, flags=re.DOTALL)
    assert len(matches) == 2
    return matches[1]


def test_page_is_self_contained_and_embeds_only_blinded_public_material(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    data = page.build_page_data(
        public,
        root,
        setup,
        freeze,
        reviewer_id="reviewer-a",
        manual_version="ooc-manual-v1",
    )
    html = page.render_annotation_page(data)
    embedded = _embedded_data(html)

    assert embedded["dataset_id"] == DATASET_ID
    assert embedded["public_input_sha256"] == contract.canonical_sha256(public)
    assert embedded["author_setup_sha256"] == contract.canonical_sha256(setup)
    assert embedded["execution_freeze_sha256"] == contract.canonical_sha256(freeze)
    assert embedded["reviewer_id"] == "reviewer-a"
    assert embedded["manual_version"] == "ooc-manual-v1"
    assert [row["text"] for row in embedded["cases"][0]["documents"][0]["lines"]] == BASELINE_STORY.splitlines()
    assert [row["text"] for row in embedded["cases"][0]["documents"][1]["lines"]] == TARGET_STORY.splitlines()
    assert embedded["cases"][0]["dimension"] == "core_trait"
    assert embedded["cases"][0]["candidate_selector"]["character_key"] == "沈砚"
    assert embedded["cases"][0]["author_axis"]["display_name"] == "守约倾向"
    assert embedded["cases"][0]["documents"][0]["phase"] == "baseline"
    assert embedded["cases"][0]["documents"][1]["phase"] == "target"
    assert "globalThis.annotationLeak" in embedded["cases"][0]["documents"][0]["lines"][1][
        "text"
    ]
    assert "</script><script>globalThis.annotationLeak" not in html
    assert html.count("<script") == 2
    assert "connect-src 'none'" in html
    assert "src=\"http" not in html
    assert "href=\"http" not in html
    assert not any(
        key in embedded
        for key in ("gold", "private_gold", "prediction", "peer_annotation")
    )


def test_generated_javascript_builds_exact_current_reviewer_bundle_fields(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    html = page.render_annotation_page(
        page.build_page_data(
            public,
            root,
            setup,
            freeze,
            reviewer_id="reviewer-a",
            manual_version="ooc-manual-v1",
        )
    )
    javascript = _inline_javascript(html)

    for field in workflow.ReviewerAnnotationBundle.model_fields:
        assert field in javascript
    for field in workflow.ReviewerCaseAnnotation.model_fields:
        assert field in javascript
    assert 'reviewer_kind: state.declarations.humanConfirmed ? "human" : null' in javascript
    assert "blinded_to_peer: state.declarations.peerBlindConfirmed" in javascript
    assert "blinded_to_system_prediction: state.declarations.systemBlindConfirmed" in javascript
    assert "independence_declaration: state.declarations.independenceConfirmed" in javascript
    assert "humanConfirmed: false" in javascript
    assert "dimension: caseRow.dimension" in javascript
    assert "roleAllowed(caseIndex, item.evidence_id, item.role)" in javascript
    assert "DATA.author_setup_sha256" in javascript
    assert "DATA.execution_freeze_sha256" in javascript
    assert "select.disabled = state.locked" in javascript
    assert "deepEqual(row, rebuiltByCase.get(row.case_id))" in javascript
    assert '["G", "X", "P"]' in javascript
    assert 'row.role === "P" ? "ambiguous" : "bounded"' in javascript
    assert "independent_event_groups" in javascript
    assert "explanation_links" in javascript

    valid_bundle = workflow.ReviewerAnnotationBundle.model_validate_json(
        json.dumps({
            "dataset_id": public.dataset_id,
            "public_input_sha256": contract.canonical_sha256(public),
            "reviewer_kind": "human",
            "reviewer_id": "reviewer-a",
            "manual_version": "ooc-manual-v1",
            "blinded_to_peer": True,
            "blinded_to_system_prediction": True,
            "independence_declaration": True,
            "cases": [
                {
                    "case_id": CASE_ID,
                    "dimension": "core_trait",
                    "outcome": "conflict",
                    "surface": "formal_issue",
                    "conflict_level": "L3",
                    "material_coverage": "complete",
                    "evidence": [
                        {"evidence_id": EVIDENCE_IDS[0], "role": "B"},
                        {"evidence_id": EVIDENCE_IDS[2], "role": "C"},
                        {"evidence_id": EVIDENCE_IDS[3], "role": "C"},
                    ],
                    "independent_event_groups": [
                        {"event_group_id": "event-1", "evidence_ids": [EVIDENCE_IDS[2]]},
                        {"event_group_id": "event-2", "evidence_ids": [EVIDENCE_IDS[3]]},
                    ],
                    "explanation_links": [],
                    "phenomena": ["core_trait_reversal"],
                    "reason_codes": ["two_independent_events"],
                    "confidence": 5,
                    "review_note": "两次独立行为均与稳定基线直接相反。",
                }
            ],
        }, ensure_ascii=False)
    )
    workflow.validate_annotation_binding(public, valid_bundle)
    workflow._validate_annotation_setup_binding(public, setup, valid_bundle)


def test_generator_writes_once_and_cli_reports_created(tmp_path, capsys):
    _, public_file, root, _, setup_file, _, freeze_file = _fixture(tmp_path)
    output = tmp_path / "reviewer-a.html"
    created = page.generate_annotation_page(
        public_file,
        root,
        setup_file,
        freeze_file,
        reviewer_id="reviewer-a",
        manual_version="ooc-manual-v1",
        output_path=output,
    )
    assert created == output.resolve()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
    with pytest.raises(page.AnnotationPageError, match="output_already_exists"):
        page.generate_annotation_page(
            public_file,
            root,
            setup_file,
            freeze_file,
            reviewer_id="reviewer-a",
            manual_version="ooc-manual-v1",
            output_path=output,
        )

    cli_output = tmp_path / "reviewer-b.html"
    assert page.main(
        [
            "--public-input",
            str(public_file),
            "--bundle-root",
            str(root),
            "--author-setup",
            str(setup_file),
            "--execution-freeze",
            str(freeze_file),
            "--reviewer-id",
            "reviewer-b",
            "--manual-version",
            "ooc-manual-v1",
            "--output",
            str(cli_output),
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {"status": "created", "output": str(cli_output.resolve())}

    direct_output = tmp_path / "reviewer-c.html"
    result = subprocess.run(
        [
            sys.executable,
            str(page.REPO_ROOT / "scripts" / "generate_ooc_annotation_page.py"),
            "--public-input",
            str(public_file),
            "--bundle-root",
            str(root),
            "--author-setup",
            str(setup_file),
            "--execution-freeze",
            str(freeze_file),
            "--reviewer-id",
            "reviewer-c",
            "--manual-version",
            "ooc-manual-v1",
            "--output",
            str(direct_output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "status": "created",
        "output": str(direct_output.resolve()),
    }


def test_generator_rejects_non_exhaustive_catalog_and_unsafe_identity(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    payload = public.model_dump(mode="json")
    payload["cases"][0]["evidence_catalog"] = payload["cases"][0][
        "evidence_catalog"
    ][:-1]
    incomplete = contract.PublicInputBundle.model_validate_json(
        json.dumps(payload, ensure_ascii=False)
    )
    with pytest.raises(workflow.AnnotationWorkflowError, match="cover_each_nonempty"):
        page.build_page_data(
            incomplete,
            root,
            setup,
            freeze,
            reviewer_id="reviewer-a",
            manual_version="ooc-manual-v1",
        )
    with pytest.raises(page.AnnotationPageError, match="reviewer_id_invalid"):
        page.build_page_data(
            public,
            root,
            setup,
            freeze,
            reviewer_id="reviewer\nother",
            manual_version="ooc-manual-v1",
        )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_inline_javascript_has_valid_syntax(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    html = page.render_annotation_page(
        page.build_page_data(
            public,
            root,
            setup,
            freeze,
            reviewer_id="reviewer-a",
            manual_version="ooc-manual-v1",
        )
    )
    script = tmp_path / "annotation-page.js"
    script.write_text(_inline_javascript(html), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node"), "--check", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


PLAYWRIGHT_MODULE = page.REPO_ROOT / "frontend" / "node_modules" / "@playwright" / "test"


@pytest.mark.skipif(
    shutil.which("node") is None or not PLAYWRIGHT_MODULE.exists(),
    reason="Playwright is unavailable",
)
def test_browser_requires_active_declarations_and_exports_bound_bundle(tmp_path):
    public, public_file, root, setup, setup_file, _, freeze_file = _fixture(tmp_path)
    html_path = tmp_path / "reviewer-browser.html"
    page.generate_annotation_page(
        public_file,
        root,
        setup_file,
        freeze_file,
        reviewer_id="reviewer-browser",
        manual_version="ooc-manual-v1",
        output_path=html_path,
    )
    locked_path = tmp_path / "browser-export.locked.json"
    browser_script = r"""
const { chromium } = require(process.argv[1]);
const { pathToFileURL } = require("url");
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.goto(pathToFileURL(process.argv[2]).href);
  const roles = page.locator("select.role-select");
  await roles.nth(0).selectOption("B");
  await roles.nth(2).selectOption("C");
  await roles.nth(3).selectOption("C");
  await page.locator('input[name="level-0"][value="L3"]').check();
  await page.locator("#event-group-0-0").fill("event-1");
  await page.locator("#event-group-0-1").fill("event-2");
  await page.locator("#phenomena-0").fill("core_trait_reversal");
  await page.locator("#reasons-0").fill("two_independent_events");
  await page.locator('input[name="confidence-0"][value="5"]').check();
  await page.locator("#review-note-0").fill("两次独立行为均与冻结语义轴相反。");
  const blocked = page.waitForEvent("download", { timeout: 600 }).then(() => false).catch(() => true);
  await page.locator("#download-locked").click();
  if (!await blocked) throw new Error("download was not blocked");
  const errorText = await page.locator("#error-summary").innerText();
  if (!errorText.includes("标注者本人") || !errorText.includes("未查看同伴答案")) {
    throw new Error(`declaration errors missing: ${errorText}`);
  }
  const declarations = page.locator('#declarations input[type="checkbox"]');
  if (await declarations.count() !== 4) throw new Error("expected four declarations");
  for (let index = 0; index < 4; index += 1) await declarations.nth(index).check();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#download-locked").click();
  const download = await downloadPromise;
  await download.saveAs(process.argv[3]);
  const evidenceLocked = await roles.evaluateAll(items => items.every(item => item.disabled));
  console.log(JSON.stringify({ evidenceLocked, declarationCount: await declarations.count() }));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [
            shutil.which("node"),
            "-e",
            browser_script,
            str(PLAYWRIGHT_MODULE.resolve()),
            str(html_path),
            str(locked_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "evidenceLocked": True,
        "declarationCount": 4,
    }
    annotation = workflow.ReviewerAnnotationBundle.model_validate_json(
        locked_path.read_text(encoding="utf-8")
    )
    workflow.validate_annotation_binding(public, annotation)
    workflow._validate_annotation_setup_binding(public, setup, annotation)
