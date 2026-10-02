import test from "node:test";
import assert from "node:assert/strict";
import { candidateBusyLeaveConfirmation, candidateInputAfterSubmission, candidateLeaveConfirmation, hasPendingCandidateInput } from "../src/features/characters/candidateDraftProtection.ts";
import { scopedAxisReviewBoundary } from "../src/features/characters/scopedAxisReviewCopy.ts";

const blank = { comment: "", axisChoice: "", axisName: "", axisDefinition: "", axisPositiveProposition: "", applicabilityScope: "", legacyAxisProposition: "", alignmentChoice: "uncertain", scopeConfirmed: false };

test("candidate form defaults alone are not an unsaved draft", () => {
  assert.equal(hasPendingCandidateInput(blank), false);
  assert.equal(hasPendingCandidateInput({ ...blank, comment: "  \n" }), false);
});

test("every actual unsubmitted candidate field participates in leave protection", () => {
  for (const field of ["comment", "axisChoice", "axisName", "axisDefinition", "axisPositiveProposition", "applicabilityScope", "legacyAxisProposition"]) {
    assert.equal(hasPendingCandidateInput({ ...blank, [field]: "作者输入" }), true, field);
  }
  assert.equal(hasPendingCandidateInput({ ...blank, alignmentChoice: "same" }), true);
  assert.equal(hasPendingCandidateInput({ ...blank, alignmentChoice: "opposite" }), true);
  assert.equal(hasPendingCandidateInput({ ...blank, scopeConfirmed: true }), true);
});

test("a successful submission clears the unchanged submitted field only", () => {
  assert.equal(candidateInputAfterSubmission("已提交备注", "已提交备注", ""), "");
  assert.equal(candidateInputAfterSubmission("后来补写的备注", "已提交备注", ""), "后来补写的备注");
  assert.equal(candidateInputAfterSubmission("same", "same", "uncertain"), "uncertain");
  assert.equal(candidateInputAfterSubmission("opposite", "same", "uncertain"), "opposite");
  assert.equal(candidateInputAfterSubmission(true, true, false), false);
});

test("created axis is server state, while its unsubmitted binding remains pending", () => {
  assert.equal(hasPendingCandidateInput({ ...blank, createdAxis: { id: "server-axis" } }), false);
  assert.equal(hasPendingCandidateInput({ ...blank, axisChoice: "server-axis" }), true);
  const afterAxisCreate = { ...blank, axisName: candidateInputAfterSubmission("轴名称", "轴名称", ""), axisChoice: "server-axis" };
  assert.equal(hasPendingCandidateInput(afterAxisCreate), true);
});

test("leave messages distinguish memory-only input from completed server mutations", () => {
  assert.match(candidateLeaveConfirmation, /仅在本页内存/);
  assert.match(candidateLeaveConfirmation, /已创建的作者轴.*不会撤销/);
  assert.match(candidateBusyLeaveConfirmation, /服务器仍可能完成/);
  assert.match(candidateBusyLeaveConfirmation, /返回后请读取核对/);
});

test("scoped-axis user copy marks implementation, deployment opt-in and unverified quality", () => {
  assert.match(scopedAxisReviewBoundary, /已实现为实验性能力/);
  assert.match(scopedAxisReviewBoundary, /需部署端启用/);
  assert.match(scopedAxisReviewBoundary, /覆盖以本次运行诊断为准/);
  assert.match(scopedAxisReviewBoundary, /尚未完成人工封闭质量验收/);
  assert.doesNotMatch(scopedAxisReviewBoundary, /待开发|仍在开发中/);
});
