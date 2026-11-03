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
    _event_policy_fixture,
    _relation_policy_fixture,
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
    if manual_version == "ooc-manual-v3-event":
        case["assessment_policy"] = "single-clear-event-v3"
        case["formal_evidence_confirmation"] = {
            "explicit_opposition": variant == "formal",
            "same_scope": variant == "formal",
            "explanations_checked": variant == "formal",
        }
        if variant == "formal":
            case["evidence"] = case["evidence"][:2]
            case["independent_event_groups"] = case["independent_event_groups"][:1]
            case["reason_codes"] = ["single_clear_event"]
            case["review_note"] = "一次明确表现与作者冻结基线直接相反，已核对适用范围和解释。"
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


def _directional_annotation(public, reviewer_id):
    legacy = public.model_copy(update={"relation_policy": None})
    payload = _annotation(legacy, reviewer_id=reviewer_id, variant="formal", manual_version="ooc-manual-v3-event").model_dump(mode="json")
    payload.update(public_input_sha256=contract.canonical_sha256(public), manual_version="ooc-manual-v4-relations")
    case = payload["cases"][0]
    case.update(relation_policy="directional-evidence-v1", evidence=[
        {"evidence_id": EVIDENCE_IDS[0], "role": "B"},
        {"evidence_id": EVIDENCE_IDS[1], "role": "G"},
        {"evidence_id": EVIDENCE_IDS[2], "role": "C"},
        {"evidence_id": EVIDENCE_IDS[3], "role": "C"},
    ], independent_event_groups=[
        {"event_group_id": "event-a", "evidence_ids": [EVIDENCE_IDS[2]]},
        {"event_group_id": "event-b", "evidence_ids": [EVIDENCE_IDS[3]]},
    ], explanation_links=[
        {"support_evidence_id": EVIDENCE_IDS[1], "applicable_current_ids": [EVIDENCE_IDS[2]], "relation_direction": "explains", "causal_relation": "bounded", "temporal_relation": "before"},
        {"support_evidence_id": EVIDENCE_IDS[1], "applicable_current_ids": [EVIDENCE_IDS[3]], "relation_direction": "opposes", "causal_relation": "explicit", "temporal_relation": "before"},
    ])
    return workflow.ReviewerAnnotationBundle.model_validate_json(json.dumps(payload, ensure_ascii=False))


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
    assert "以 A 的判断为起点" in javascript
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


def test_v3_adjudication_preserves_policy_but_does_not_adopt_human_confirmation(tmp_path):
    public, root, setup, freeze = _event_policy_fixture(tmp_path)
    left = _annotation(public, reviewer_id="reviewer-a", variant="formal", manual_version="ooc-manual-v3-event")
    right = _annotation(public, reviewer_id="reviewer-b", variant="formal", manual_version="ooc-manual-v3-event")
    data = page.build_page_data(public, root, setup, freeze, left, right)
    assert data["assessment_policy"] == "single-clear-event-v3"
    assert data["cases"][0]["reviewer_a"]["formal_evidence_confirmation"]["explicit_opposition"] is True
    javascript = _inline_javascript(page.render_adjudication_page(data))
    assert 'input.id = `formal-confirm-${caseIndex}`' in javascript
    assert "explicit_opposition: false, same_scope: false, explanations_checked: false" in javascript
    assert 'result.assessment_policy = "single-clear-event-v3"' in javascript
    assert "result.formal_evidence_confirmation" in javascript
    copier = javascript.split("function copyReviewer", 1)[1].split("function adoptField", 1)[0]
    assert "const next = blankCase()" in copier
    assert "formalEvidenceConfirmation = source" not in copier
    assert "formal_evidence_confirmation" not in copier
    assert "不能直接继承 A／B 的确认" in javascript


@pytest.mark.skipif(
    shutil.which("node") is None or not (page.REPO_ROOT / "frontend" / "node_modules" / "@playwright" / "test").exists(),
    reason="Playwright is unavailable",
)
def test_browser_v3_single_event_adjudication_requires_own_confirmation_and_roundtrips(tmp_path):
    public, root, setup, freeze = _event_policy_fixture(tmp_path)
    left = _annotation(public, reviewer_id="reviewer-a", variant="formal", manual_version="ooc-manual-v3-event")
    right = _annotation(public, reviewer_id="reviewer-b", variant="formal", manual_version="ooc-manual-v3-event")
    html_path = tmp_path / "adjudication-v3.html"
    html_path.write_text(page.render_adjudication_page(page.build_page_data(public, root, setup, freeze, left, right)), encoding="utf-8")
    locked_path = tmp_path / "adjudication-v3.locked.json"
    browser_script = r'''
const { chromium } = require(process.argv[1]);
const { pathToFileURL } = require("url");
(async()=>{
  const browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  await page.goto(pathToFileURL(process.argv[2]).href);
  await page.getByRole("button",{name:"以 A 的判断为起点"}).click();
  if(await page.locator("#formal-confirm-0").isChecked()) throw new Error("adjudicator confirmation silently copied");
  await page.locator("#note-0").fill("两位真人复核一次明确事件，确认人物、范围及解释条件。");
  await page.locator("#download-locked").click();
  if(!(await page.locator("#error-summary").innerText()).includes("仲裁者主动核对")) throw new Error("fresh human confirmation not required");
  await page.locator("#formal-confirm-0").check();
  await page.locator("#coverage-0").selectOption("partial");
  if(await page.locator("#formal-confirm-0").isChecked()) throw new Error("material edit retained stale confirmation");
  await page.locator("#coverage-0").selectOption("complete");
  await page.locator("#formal-confirm-0").check();
  const declarations=page.locator('#final-confirmations input[type="checkbox"]');
  await declarations.nth(0).check(); await declarations.nth(1).check();
  const downloadPromise=page.waitForEvent("download");
  await page.locator("#download-locked").click();
  await (await downloadPromise).saveAs(process.argv[3]);
  await page.locator("#unlock-editing").click();
  await page.locator("#import-file").setInputFiles(process.argv[3]);
  await page.waitForFunction(()=>document.querySelector("#unlock-editing")?.hidden===false && document.querySelector("#formal-confirm-0")?.matches(":disabled")===true);
  if(!await page.locator("#formal-confirm-0").isChecked()) throw new Error("locked confirmation did not roundtrip");
  console.log("v3-adjudication-browser-ok");
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1);});
'''
    result = subprocess.run([
        shutil.which("node"), "-e", browser_script,
        str((page.REPO_ROOT / "frontend" / "node_modules" / "@playwright" / "test").resolve()),
        str(html_path), str(locked_path),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v3-adjudication-browser-ok"
    adjudication = workflow.AdjudicationBundle.model_validate_json(locked_path.read_text(encoding="utf-8"))
    workflow._validate_adjudication_setup_binding(public, setup, adjudication)
    gold = workflow._build_private_gold(public, left, right, adjudication)
    assert adjudication.cases[0].assessment_policy == "single-clear-event-v3"
    assert adjudication.cases[0].formal_evidence_confirmation.explicit_opposition is True
    assert gold.cases[0].assessment_policy == "single-clear-event-v3"


def test_v4_adjudication_carries_mixed_target_directions_without_adopting_confirmation(tmp_path):
    public, root, setup, freeze = _relation_policy_fixture(tmp_path)
    left = _directional_annotation(public, "reviewer-a")
    right = _directional_annotation(public, "reviewer-b")
    data = page.build_page_data(public, root, setup, freeze, left, right)
    assert data["relation_policy"] == "directional-evidence-v1"
    links = data["cases"][0]["reviewer_a"]["explanation_links"]
    assert [link["relation_direction"] for link in links] == ["explains", "opposes"]
    javascript = _inline_javascript(page.render_adjudication_page(data))
    assert 'restoreLinks(next.explanations, source.explanation_links)' in javascript
    assert 'relation_policy = "directional-evidence-v1"' in javascript
    assert 'const controlId = `relation-${caseIndex}-${supportIndex}-${targetIndex}`' in javascript
    assert '未知关系（需核对）' in javascript
    assert 'causal.disabled = state.locked || value.direction === "uncertain" || role === "P"' in javascript
    assert '[["ambiguous", "关系尚不确定"]]' in javascript


@pytest.mark.skipif(shutil.which("node") is None or not (page.REPO_ROOT / "frontend" / "node_modules" / "@playwright" / "test").exists(), reason="Playwright unavailable")
def test_browser_v4_mixed_growth_directions_adjudication_roundtrips_and_keeps_gold_context(tmp_path):
    public, root, setup, freeze = _relation_policy_fixture(tmp_path)
    left = _directional_annotation(public, "reviewer-a")
    right = _directional_annotation(public, "reviewer-b")
    html_path = tmp_path / "adjudication-v4.html"
    html_path.write_text(page.render_adjudication_page(page.build_page_data(public, root, setup, freeze, left, right)), encoding="utf-8")
    locked_path = tmp_path / "adjudication-v4.locked.json"
    script = r'''
const {chromium}=require(process.argv[1]);const {pathToFileURL}=require("node:url");
(async()=>{const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}});
await page.goto(pathToFileURL(process.argv[2]).href);await page.getByRole("button",{name:"以 A 的判断为起点"}).click();
if(await page.locator("#relation-0-0-0").inputValue()!=="explains"||await page.locator("#relation-0-0-1").inputValue()!=="opposes")throw new Error("mixed same-growth directions lost on copy");
if(await page.locator("#formal-confirm-0").isChecked())throw new Error("reviewer confirmation copied");
await page.locator("#note-0").fill("合成仲裁：分别核对同一成长对两件事情的不同方向，不依据片段数定案。");
await page.locator("#formal-confirm-0").check();await page.locator("#relation-0-0-1").selectOption("uncertain");
if(await page.locator("#formal-confirm-0").isChecked())throw new Error("direction change retained stale confirmation");
await page.locator("#download-locked").click();if(!(await page.locator("#error-summary").innerText()).includes("所选事件均有"))throw new Error("unresolved relation did not block formal issue");
await page.locator("#relation-0-0-1").selectOption("opposes");await page.locator("#formal-confirm-0").check();
await page.locator('#final-confirmations input[type="checkbox"]').nth(0).check();await page.locator('#final-confirmations input[type="checkbox"]').nth(1).check();
const promise=page.waitForEvent("download");await page.locator("#download-locked").click();await(await promise).saveAs(process.argv[3]);
await page.locator("#unlock-editing").click();
const valid=JSON.parse(require("node:fs").readFileSync(process.argv[3],"utf8"));
for(const invalid of ["unknown-direction","extra-field","missing-direction"]){
 const payload=JSON.parse(JSON.stringify(valid));
 if(invalid==="unknown-direction")payload.cases[0].explanation_links[0].relation_direction="invented";
 if(invalid==="extra-field")payload.cases[0].explanation_links[0].unsupported_field="invented";
 if(invalid==="missing-direction")delete payload.cases[0].explanation_links[0].relation_direction;
 await page.locator("#status").evaluate(node=>{node.textContent="";});
 await page.locator("#import-file").setInputFiles({name:"invalid.locked.json",mimeType:"application/json",buffer:Buffer.from(JSON.stringify(payload))});
 await page.waitForFunction(()=>document.querySelector("#status").textContent.length>0);
 if(await page.locator("#relation-0-0-1").isDisabled())throw new Error("invalid relation import locked working state");
 if(await page.locator("#relation-0-0-0").inputValue()!=="explains"||await page.locator("#relation-0-0-1").inputValue()!=="opposes")throw new Error("invalid import changed relation state");
}
await page.locator("#import-file").setInputFiles(process.argv[3]);await page.waitForFunction(()=>document.querySelector("#relation-0-0-1")?.matches(":disabled")===true);
if(await page.locator("#relation-0-0-0").inputValue()!=="explains"||await page.locator("#relation-0-0-1").inputValue()!=="opposes")throw new Error("directions lost after locked import");
console.log("v4-adjudication-directions-ok");await browser.close();})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run([shutil.which("node"), "-e", script, str((page.REPO_ROOT / "frontend/node_modules/@playwright/test").resolve()), str(html_path), str(locked_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v4-adjudication-directions-ok"
    adjudication = workflow.AdjudicationBundle.model_validate_json(locked_path.read_text(encoding="utf-8"))
    workflow._validate_adjudication_setup_binding(public, setup, adjudication)
    gold = workflow._build_private_gold(public, left, right, adjudication)
    assert {link.relation_direction for link in adjudication.cases[0].explanation_links} == {"explains", "opposes"}
    assert {link.relation_direction for link in gold.cases[0].auxiliary_relations} == {"explains", "opposes"}


@pytest.mark.skipif(shutil.which("node") is None or not (page.REPO_ROOT / "frontend/node_modules/@playwright/test").exists(), reason="Playwright unavailable")
def test_browser_readable_review_categories_and_mobile_layout_preserve_empty_decisions(tmp_path):
    """Synthetic UI-only cases: no real annotation or private answer is read."""
    from copy import deepcopy

    public, root, setup, freeze = _relation_policy_fixture(tmp_path)
    left = _directional_annotation(public, "reviewer-a")
    right = _directional_annotation(public, "reviewer-b")
    data = page.build_page_data(public, root, setup, freeze, left, right, third_human_id="reviewer-c", default_resolution_mode="third_human")
    agreed = data["cases"][0]
    outcome_diff = deepcopy(agreed)
    outcome_diff["agreement"].update(outcome_agrees=False, surface_agrees=False)
    basis_diff = deepcopy(agreed)
    basis_diff["agreement"]["evidence_agrees"] = False
    data["cases"] = [agreed, outcome_diff, basis_diff]
    html_path = tmp_path / "readable-review-synthetic.html"
    html_path.write_text(page.render_adjudication_page(data), encoding="utf-8")
    script = r'''
const {chromium}=require(process.argv[1]);const {pathToFileURL}=require("node:url");
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:375,height:1000}});
 const errors=[],external=[];page.on("pageerror",error=>errors.push(String(error)));
 page.on("request",request=>{if(/^https?:/.test(request.url()))external.push(request.url())});
 await page.goto(pathToFileURL(process.argv[2]).href);
 const summary=await page.locator("#review-summary").innerText();
 for(const label of ["结论不同 1 题","依据不同 1 题","全部一致 1 题"])if(!summary.includes(label))throw new Error("review category counts missing");
 const options=await page.locator("#case-select option").allTextContents();
 for(const label of ["全部一致","结论不同","依据不同"])if(!options.some(value=>value.includes(label)))throw new Error("mobile category missing");
 if(await page.locator("#outcome-0").inputValue()!=="")throw new Error("decision preselected");
 if(await page.locator('#final-confirmations input:checked').count())throw new Error("human declaration preselected");
 if(await page.locator('input[name="mode-0"]:checked').inputValue()!=="third_human")throw new Error("third human mode binding lost");
 const text=await page.locator("body").innerText();
 for(const value of ["00000000-0000", "最终 outcome", "最终 surface", "core_trait", "case_id", "trait_type"])
  if(text.includes(value))throw new Error("technical value in main view: "+value);
 const leaked=await page.evaluate(()=>{
  const data=JSON.parse(document.querySelector("#adjudication-data").textContent),text=document.body.innerText;
  return [data.public_input_sha256,data.cases[0].case_id,data.cases[0].documents[0].path].filter(value=>text.includes(value));
 });if(leaked.length)throw new Error("raw binding visible by default");
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw new Error("375px horizontal overflow");
 await page.screenshot({path:process.argv[3],fullPage:true});
 await page.locator("#note-0").fill("尚未完成的合成复核进度，仅验证能保存。");
 await page.getByRole("button",{name:"保存进度",exact:true}).click();await page.reload();
 if(!(await page.locator("#note-0").inputValue()).includes("尚未完成"))throw new Error("incomplete draft not restored");
 if(await page.locator("#outcome-0").inputValue()!=="")throw new Error("saving draft decided case");
 await page.setViewportSize({width:1440,height:1000});
 const categories=await page.locator(".case-category").allTextContents();
 if(categories.join("|")!=="全部一致|结论不同|依据不同")throw new Error("desktop category labels mismatch");
 await page.screenshot({path:process.argv[4],fullPage:true});
 await page.locator("#download-locked").click();
 await page.locator('#error-list button').filter({hasText:"第 1 题：请保留或填写观察到的现象"}).click();
 await page.waitForFunction(()=>document.activeElement?.id==="phenomena-0");
 if(!await page.locator("#phenomena-0").isVisible())throw new Error("error target remains hidden in collapsed details");
 if(errors.length||external.length)throw new Error("browser errors or network requests");
 console.log("readable-mobile-review-ok");await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
'''
    result = subprocess.run([
        shutil.which("node"), "-e", script,
        str((page.REPO_ROOT / "frontend/node_modules/@playwright/test").resolve()),
        str(html_path), str(tmp_path / "review-mobile-375.png"), str(tmp_path / "review-desktop-1440.png"),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "readable-mobile-review-ok"


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
  await page.getByRole("button", { name: "以 A 的判断为起点" }).click();
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
  await page.getByRole("button", { name: "以 A 的判断为起点" }).click();
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
