"""Synthetic-only tests for the offline OOC adjudication page generator."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys

import pytest

from scripts import generate_ooc_adjudication_page as page
from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_eval_contract as contract
from tests.test_generate_ooc_annotation_page import (
    CASE_ID,
    DATASET_ID,
    EVIDENCE_IDS,
    _fixture,
)


MANUAL_VERSION = "ooc-manual-v1"


def _annotation(
    public: contract.PublicInputBundle,
    *,
    reviewer_id: str,
    variant: str,
    manual_version: str = MANUAL_VERSION,
) -> workflow.ReviewerAnnotationBundle:
    if variant == "formal":
        case = {
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
                {"event_group_id": "event-a", "evidence_ids": [EVIDENCE_IDS[2]]},
                {"event_group_id": "event-b", "evidence_ids": [EVIDENCE_IDS[3]]},
            ],
            "explanation_links": [],
            "phenomena": ["core_trait_reversal"],
            "reason_codes": ["two_independent_events"],
            "confidence": 5,
            "review_note": "两次独立行为均与作者冻结基线相反。",
        }
    elif variant == "review":
        case = {
            "case_id": CASE_ID,
            "dimension": "core_trait",
            "outcome": "indeterminate",
            "surface": "review_clue",
            "conflict_level": "L2",
            "material_coverage": "complete",
            "evidence": [
                {"evidence_id": EVIDENCE_IDS[0], "role": "B"},
                {"evidence_id": EVIDENCE_IDS[1], "role": "P"},
                {"evidence_id": EVIDENCE_IDS[2], "role": "C"},
            ],
            "independent_event_groups": [],
            "explanation_links": [
                {
                    "support_evidence_id": EVIDENCE_IDS[1],
                    "applicable_current_ids": [EVIDENCE_IDS[2]],
                    "causal_relation": "ambiguous",
                    "temporal_relation": "before",
                }
            ],
            "phenomena": ["possible_reversal"],
            "reason_codes": ["single_current_event"],
            "confidence": 3,
            "review_note": "仅确认一个当前事件，保留为复核线索。",
        }
    else:  # pragma: no cover - test helper guard
        raise AssertionError(variant)
    return workflow.ReviewerAnnotationBundle.model_validate_json(
        json.dumps(
            {
                "dataset_id": public.dataset_id,
                "public_input_sha256": contract.canonical_sha256(public),
                "reviewer_kind": "human",
                "reviewer_id": reviewer_id,
                "manual_version": manual_version,
                "blinded_to_peer": True,
                "blinded_to_system_prediction": True,
                "independence_declaration": True,
                "cases": [case],
            },
            ensure_ascii=False,
        )
    )


def _annotations(public: contract.PublicInputBundle):
    return (
        _annotation(public, reviewer_id="reviewer-a", variant="formal"),
        _annotation(public, reviewer_id="reviewer-b", variant="review"),
    )


def _write_locked(path, value: workflow.ReviewerAnnotationBundle) -> None:
    path.write_text(
        json.dumps(value.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )


def _embedded_data(html: str) -> dict:
    match = re.search(
        r'<script id="adjudication-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _inline_javascript(html: str) -> str:
    matches = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, flags=re.DOTALL)
    assert len(matches) == 2
    return matches[1]


def test_page_embeds_frozen_inputs_context_and_two_human_submissions(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    left, right = _annotations(public)
    data = page.build_page_data(public, root, setup, freeze, left, right)
    html = page.render_adjudication_page(data)
    embedded = _embedded_data(html)

    assert embedded["dataset_id"] == DATASET_ID
    assert embedded["public_input_sha256"] == contract.canonical_sha256(public)
    assert embedded["author_setup_sha256"] == contract.canonical_sha256(setup)
    assert embedded["execution_freeze_sha256"] == contract.canonical_sha256(freeze)
    assert embedded["manual_version"] == MANUAL_VERSION
    assert [row["reviewer_id"] for row in embedded["reviewers"]] == [
        "reviewer-a",
        "reviewer-b",
    ]
    assert [row["annotation_sha256"] for row in embedded["reviewers"]] == [
        contract.canonical_sha256(left),
        contract.canonical_sha256(right),
    ]
    case = embedded["cases"][0]
    assert case["dimension"] == "core_trait"
    assert case["candidate_selector"]["character_key"] == "沈砚"
    assert case["author_axis"]["display_name"] == "守约倾向"
    assert case["documents"][0]["phase"] == "baseline"
    assert case["documents"][1]["phase"] == "target"
    assert case["agreement"]["dimension_agrees"] is True
    assert case["agreement"]["outcome_agrees"] is False
    assert case["agreement"]["evidence_agrees"] is False
    assert "</script><script>globalThis.annotationLeak" not in html
    assert html.count("<script") == 2
    assert "connect-src 'none'" in html
    assert "src=\"http" not in html
    assert "href=\"http" not in html
    assert not any(
        key in embedded
        for key in ("gold", "private_gold", "prediction", "system_prediction")
    )


def test_javascript_targets_exact_adjudication_schema_and_active_confirmations(
    tmp_path,
):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    left, right = _annotations(public)
    javascript = _inline_javascript(
        page.render_adjudication_page(
            page.build_page_data(
                public,
                root,
                setup,
                freeze,
                left,
                right,
                third_human_id="reviewer-c",
            )
        )
    )

    for field in workflow.AdjudicationBundle.model_fields:
        assert field in javascript
    for field in workflow.AdjudicatedCase.model_fields:
        assert field in javascript
    assert "dimension: source.dimension" in javascript
    assert "roleAllowed(caseIndex, item.evidence_id, item.role)" in javascript
    assert "state.confirmations.reviewerA" in javascript
    assert "state.confirmations.reviewerB" in javascript
    assert "reviewerA: false" in javascript
    assert "reviewerB: false" in javascript
    assert "thirdHumanBlind: false" in javascript
    assert "DATA.author_setup_sha256" in javascript
    assert "DATA.execution_freeze_sha256" in javascript
    assert "consensus_confirmed_by" in javascript
    assert "以 A ·" in javascript
    assert "填入全部一致项" in javascript
    assert "不会自动定案" in javascript

    valid = workflow.AdjudicationBundle.model_validate_json(
        json.dumps(
            {
                "dataset_id": public.dataset_id,
                "public_input_sha256": contract.canonical_sha256(public),
                "reviewer_ids": [left.reviewer_id, right.reviewer_id],
                "annotation_sha256": [
                    contract.canonical_sha256(left),
                    contract.canonical_sha256(right),
                ],
                "consensus_confirmed_by": [left.reviewer_id, right.reviewer_id],
                "cases": [
                    {
                        "case_id": CASE_ID,
                        "dimension": "core_trait",
                        "final_outcome": "conflict",
                        "final_surface": "formal_issue",
                        "conflict_level": "L3",
                        "material_coverage": "complete",
                        "required_evidence": [
                            {"evidence_id": EVIDENCE_IDS[0], "role": "B"},
                            {"evidence_id": EVIDENCE_IDS[2], "role": "C"},
                            {"evidence_id": EVIDENCE_IDS[3], "role": "C"},
                        ],
                        "forbidden_evidence_ids": [EVIDENCE_IDS[1]],
                        "independent_event_groups": [
                            {
                                "event_group_id": "event-a",
                                "evidence_ids": [EVIDENCE_IDS[2]],
                            },
                            {
                                "event_group_id": "event-b",
                                "evidence_ids": [EVIDENCE_IDS[3]],
                            },
                        ],
                        "explanation_links": [],
                        "phenomena": ["core_trait_reversal"],
                        "reason_codes": ["two_independent_events"],
                        "resolution_mode": "reviewer_consensus",
                        "adjudicator_id": None,
                        "adjudicator_kind": None,
                        "adjudicator_blinded_to_system_prediction": None,
                        "resolution_note": "两位标注者复核原文后确认采用正式冲突结论。",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )
    workflow._validate_adjudication_setup_binding(public, setup, valid)
    workflow._build_private_gold(public, left, right, valid)


def test_third_human_mode_uses_explicit_human_identity_and_blindness(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    left, right = _annotations(public)
    data = page.build_page_data(
        public,
        root,
        setup,
        freeze,
        left,
        right,
        third_human_id="reviewer-c",
        default_resolution_mode="third_human",
    )
    assert data["third_human_id"] == "reviewer-c"
    assert data["default_resolution_mode"] == "third_human"

    source = left.cases[0]
    adjudication = workflow.AdjudicationBundle.model_validate_json(
        json.dumps(
            {
                "dataset_id": public.dataset_id,
                "public_input_sha256": contract.canonical_sha256(public),
                "reviewer_ids": [left.reviewer_id, right.reviewer_id],
                "annotation_sha256": [
                    contract.canonical_sha256(left),
                    contract.canonical_sha256(right),
                ],
                "consensus_confirmed_by": [left.reviewer_id, right.reviewer_id],
                "cases": [
                    {
                        "case_id": source.case_id,
                        "dimension": source.dimension,
                        "final_outcome": source.outcome,
                        "final_surface": source.surface,
                        "conflict_level": source.conflict_level,
                        "material_coverage": source.material_coverage,
                        "required_evidence": [
                            row.model_dump(mode="json") for row in source.evidence
                        ],
                        "forbidden_evidence_ids": [EVIDENCE_IDS[1]],
                        "independent_event_groups": [
                            row.model_dump(mode="json")
                            for row in source.independent_event_groups
                        ],
                        "explanation_links": [],
                        "phenomena": list(source.phenomena),
                        "reason_codes": list(source.reason_codes),
                        "resolution_mode": "third_human",
                        "adjudicator_id": "reviewer-c",
                        "adjudicator_kind": "human",
                        "adjudicator_blinded_to_system_prediction": True,
                        "resolution_note": "第三人独立复核全部原文并记录取舍，两位原标注者确认最终稿。",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )
    workflow._validate_adjudication_setup_binding(public, setup, adjudication)
    workflow._build_private_gold(public, left, right, adjudication)


def test_generator_is_exclusive_and_direct_cli_works_from_arbitrary_cwd(tmp_path):
    public, public_file, root, _, setup_file, _, freeze_file = _fixture(tmp_path)
    left, right = _annotations(public)
    left_file = tmp_path / "reviewer-a.locked.json"
    right_file = tmp_path / "reviewer-b.locked.json"
    _write_locked(left_file, left)
    _write_locked(right_file, right)
    output = tmp_path / "adjudication.html"

    assert page.generate_adjudication_page(
        public_file,
        root,
        setup_file,
        freeze_file,
        left_file,
        right_file,
        output_path=output,
    ) == output.resolve()
    with pytest.raises(page.AdjudicationPageError, match="output_already_exists"):
        page.generate_adjudication_page(
            public_file,
            root,
            setup_file,
            freeze_file,
            left_file,
            right_file,
            output_path=output,
        )

    direct_output = tmp_path / "direct-adjudication.html"
    result = subprocess.run(
        [
            sys.executable,
            str(page.REPO_ROOT / "scripts" / "generate_ooc_adjudication_page.py"),
            "--public-input",
            str(public_file),
            "--bundle-root",
            str(root),
            "--author-setup",
            str(setup_file),
            "--execution-freeze",
            str(freeze_file),
            "--left-annotation",
            str(left_file),
            "--right-annotation",
            str(right_file),
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


def test_generator_rejects_unfrozen_or_incompatible_inputs(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    left, right = _annotations(public)
    unlocked = tmp_path / "reviewer-a.json"
    _write_locked(unlocked, left)
    with pytest.raises(page.AdjudicationPageError, match="locked_json"):
        page._load_locked_annotation(unlocked)
    with pytest.raises(page.AdjudicationPageError, match="two_distinct_reviewers"):
        page.build_page_data(public, root, setup, freeze, left, left)
    wrong_manual = _annotation(
        public,
        reviewer_id="reviewer-b",
        variant="review",
        manual_version="ooc-manual-v2",
    )
    with pytest.raises(page.AdjudicationPageError, match="same_manual_version"):
        page.build_page_data(public, root, setup, freeze, left, wrong_manual)
    with pytest.raises(page.AdjudicationPageError, match="third_human_mode_requires"):
        page.build_page_data(
            public,
            root,
            setup,
            freeze,
            left,
            right,
            default_resolution_mode="third_human",
        )
    with pytest.raises(page.AdjudicationPageError, match="third_human_must_be_distinct"):
        page.build_page_data(
            public,
            root,
            setup,
            freeze,
            left,
            right,
            third_human_id=left.reviewer_id,
        )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_inline_javascript_has_valid_syntax(tmp_path):
    public, _, root, setup, _, freeze, _ = _fixture(tmp_path)
    left, right = _annotations(public)
    html = page.render_adjudication_page(
        page.build_page_data(public, root, setup, freeze, left, right)
    )
    script = tmp_path / "adjudication-page.js"
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
def test_browser_requires_two_active_confirmations_and_exports_valid_bundle(tmp_path):
    public, public_file, root, setup, setup_file, _, freeze_file = _fixture(tmp_path)
    left, right = _annotations(public)
    left_file = tmp_path / "reviewer-a.locked.json"
    right_file = tmp_path / "reviewer-b.locked.json"
    _write_locked(left_file, left)
    _write_locked(right_file, right)
    html_path = tmp_path / "adjudication-browser.html"
    page.generate_adjudication_page(
        public_file,
        root,
        setup_file,
        freeze_file,
        left_file,
        right_file,
        output_path=html_path,
    )
    locked_path = tmp_path / "adjudication-browser.locked.json"
    browser_script = r"""
const { chromium } = require(process.argv[1]);
const { pathToFileURL } = require("url");
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.goto(pathToFileURL(process.argv[2]).href);
  if (await page.locator("#outcome-0").inputValue() !== "") throw new Error("case silently decided");
  if (await page.locator("#dimension-0").evaluate(node => node.tagName) === "SELECT") throw new Error("dimension is editable");
  await page.getByRole("button", { name: /以 A .* 为起点/ }).click();
  const evidenceControls = page.locator('select[aria-label*="最终必要证据角色"]');
  await page.locator('.evidence-row').nth(1).locator('input[type="checkbox"]').check();
  await page.locator("#note-0").fill("两位标注者共同复核原文后，确认采用 A 的正式冲突结论。 ");
  await page.locator("#save-draft").click();
  await page.reload();
  if (!(await page.locator("#note-0").inputValue()).includes("共同复核")) throw new Error("local draft was not restored");
  const blocked = page.waitForEvent("download", { timeout: 800 }).then(() => false).catch(() => true);
  await page.locator("#download-locked").click();
  if (!await blocked) throw new Error("download was not blocked");
  const errorText = await page.locator("#error-summary").innerText();
  if (!errorText.includes("标注者 A") || !errorText.includes("标注者 B")) throw new Error(`confirmation errors missing: ${errorText}`);
  const confirmations = page.locator('#final-confirmations input[type="checkbox"]');
  if (await confirmations.count() !== 2) throw new Error("expected two reviewer confirmations");
  await confirmations.nth(0).check();
  await confirmations.nth(1).check();
  await page.locator("#note-0").fill("两位标注者共同复核原文后，确认采用 A 的正式冲突结论；这是最终修订。 ");
  if (await page.locator('#final-confirmations input[type="checkbox"]:checked').count() !== 0) throw new Error("editing did not revoke confirmations");
  await confirmations.nth(0).check();
  await confirmations.nth(1).check();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#download-locked").click();
  const download = await downloadPromise;
  await download.saveAs(process.argv[3]);
  const evidenceLocked = await evidenceControls.evaluateAll(items => items.every(item => item.disabled));
  await page.locator("#unlock-editing").click();
  await page.locator("#import-file").setInputFiles(process.argv[3]);
  await page.waitForFunction(() => document.querySelectorAll('select[aria-label*="最终必要证据角色"]')[0]?.disabled === true);
  const importLocked = await evidenceControls.evaluateAll(items => items.every(item => item.disabled));
  console.log(JSON.stringify({ evidenceLocked, importLocked, confirmationCount: await confirmations.count() }));
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
        "importLocked": True,
        "confirmationCount": 2,
    }
    adjudication = workflow.AdjudicationBundle.model_validate_json(
        locked_path.read_text(encoding="utf-8")
    )
    assert adjudication.reviewer_ids == (left.reviewer_id, right.reviewer_id)
    assert adjudication.annotation_sha256 == (
        contract.canonical_sha256(left),
        contract.canonical_sha256(right),
    )
    assert adjudication.consensus_confirmed_by == (
        left.reviewer_id,
        right.reviewer_id,
    )
    assert adjudication.cases[0].forbidden_evidence_ids == (EVIDENCE_IDS[1],)
    workflow._validate_adjudication_setup_binding(public, setup, adjudication)
    workflow._build_private_gold(public, left, right, adjudication)


@pytest.mark.skipif(
    shutil.which("node") is None or not PLAYWRIGHT_MODULE.exists(),
    reason="Playwright is unavailable",
)
def test_browser_third_human_requires_identity_and_blindness_confirmations(tmp_path):
    public, public_file, root, setup, setup_file, _, freeze_file = _fixture(tmp_path)
    left, right = _annotations(public)
    left_file = tmp_path / "reviewer-a.locked.json"
    right_file = tmp_path / "reviewer-b.locked.json"
    _write_locked(left_file, left)
    _write_locked(right_file, right)
    html_path = tmp_path / "third-human.html"
    page.generate_adjudication_page(
        public_file,
        root,
        setup_file,
        freeze_file,
        left_file,
        right_file,
        output_path=html_path,
        third_human_id="reviewer-c",
        default_resolution_mode="third_human",
    )
    locked_path = tmp_path / "third-human.locked.json"
    browser_script = r"""
const { chromium } = require(process.argv[1]);
const { pathToFileURL } = require("url");
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(pathToFileURL(process.argv[2]).href);
  await page.getByRole("button", { name: /以 A .* 为起点/ }).click();
  await page.locator('.evidence-row').nth(1).locator('input[type="checkbox"]').check();
  await page.locator("#note-0").fill("第三人 reviewer-c 独立复核原文并记录取舍；A、B 共同确认最终稿。 ");
  const confirmations = page.locator('#final-confirmations input[type="checkbox"]');
  if (await confirmations.count() !== 4) throw new Error("expected reviewer and third-human confirmations");
  await confirmations.nth(0).check();
  await confirmations.nth(1).check();
  const blocked = page.waitForEvent("download", { timeout: 800 }).then(() => false).catch(() => true);
  await page.locator("#download-locked").click();
  if (!await blocked) throw new Error("third-human download was not blocked");
  const errorText = await page.locator("#error-summary").innerText();
  if (!errorText.includes("第三人")) throw new Error(`third-human confirmation error missing: ${errorText}`);
  await confirmations.nth(2).check();
  await confirmations.nth(3).check();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#download-locked").click();
  await (await downloadPromise).saveAs(process.argv[3]);
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
    adjudication = workflow.AdjudicationBundle.model_validate_json(
        locked_path.read_text(encoding="utf-8")
    )
    case = adjudication.cases[0]
    assert case.resolution_mode == "third_human"
    assert case.adjudicator_id == "reviewer-c"
    assert case.adjudicator_kind == "human"
    assert case.adjudicator_blinded_to_system_prediction is True
    assert adjudication.consensus_confirmed_by == (
        left.reviewer_id,
        right.reviewer_id,
    )
    workflow._validate_adjudication_setup_binding(public, setup, adjudication)
    workflow._build_private_gold(public, left, right, adjudication)
