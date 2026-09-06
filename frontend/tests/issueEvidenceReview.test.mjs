import test from 'node:test';
import assert from 'node:assert/strict';
import {
  describeIssueEvidenceReview,
  describeIssueEvidenceReviewDiagnostic,
} from '../src/issueEvidenceReview.ts';

function metadata(verdict, overrides={}) {
  return {
    ai_evidence_review: {
      schema_version: 'issue-evidence-review-v1',
      verdict,
      note: '检索证据支持这项复核判断。',
      citations: ['E01', 'E03'],
      consumed_evidence: [
        {citation_id:'E01', document_id:'doc-1', document_version:2, line_start:7, line_end:9, cited:true, content_sha256:'hidden-hash'},
        {citation_id:'E02', document_id:'doc-2', document_version:1, line_start:4, line_end:4, cited:false, text_sha256:'hidden-hash'},
        {citation_id:'E03', document_id:'doc-3', document_version:5, line_start:21, line_end:21, cited:true, profile_id:'hidden-profile'},
      ],
      retrieval: {mode:'hybrid', profile_id:'hidden-profile', request_id:'hidden-request'},
      provider_call: {endpoint:'https://private.invalid/v1', raw:'raw prompt', key:'sk-private'},
      ...overrides,
    },
  };
}

test('all three verdicts receive distinct user-facing labels', () => {
  const expected = {
    supports_issue: '支持问题',
    contextual_exception: '存在上下文例外',
    insufficient_evidence: '证据不足',
  };
  for (const [verdict,label] of Object.entries(expected)) {
    assert.equal(describeIssueEvidenceReview(metadata(verdict))?.verdictLabel, label);
  }
});

test('only cited evidence exposes friendly document, version, lines, and hybrid mode', () => {
  const view = describeIssueEvidenceReview(metadata('supports_issue'), {'doc-1':'第一章.docx'});
  assert.equal(view?.retrievalModeLabel, '关键词 + 向量混合检索');
  assert.deepEqual(view?.citations, ['E01','E03']);
  assert.deepEqual(view?.evidence, [
    {citationId:'E01', documentLabel:'第一章.docx', documentVersion:2, lineStart:7, lineEnd:9},
    {citationId:'E03', documentLabel:'doc-3', documentVersion:5, lineStart:21, lineEnd:21},
  ]);
  const rendered = JSON.stringify(view);
  assert.doesNotMatch(rendered, /hidden-hash|hidden-profile|hidden-request|private\.invalid|raw prompt|sk-private/);
});

test('notes remain plain text and internal data is never projected to the view', () => {
  const view = describeIssueEvidenceReview(metadata('contextual_exception', {
    note:'<img src=x onerror="alert(1)">上下文例外',
  }));
  assert.equal(view?.note, '＜img src=x onerror="alert(1)"＞上下文例外');
  assert.doesNotMatch(view?.note ?? '', /<img/u);
});

test('only hybrid retrieval can produce a visible AI annotation', () => {
  for (const mode of ['lexical_only','dense_only','unavailable']) {
    assert.equal(describeIssueEvidenceReview(metadata('supports_issue',{retrieval:{mode}})),null);
  }
  assert.notEqual(describeIssueEvidenceReview(metadata('supports_issue',{retrieval:{mode:'hybrid'}})),null);
});

test('malformed annotations and legacy issues fail closed without throwing', () => {
  const invalid = [
    undefined, null, {}, {ai_evidence_review:'bad'},
    metadata('unknown'),
    metadata('supports_issue',{citations:['E99']}),
    metadata('supports_issue',{consumed_evidence:[{citation_id:'E01'}]}),
    metadata('supports_issue',{consumed_evidence:[
      {citation_id:'E01',document_id:'doc-1',document_version:1,line_start:1,line_end:1,cited:true},
      {citation_id:'E02',document_id:'doc-2',document_version:1,line_start:1,line_end:1,cited:true},
      {citation_id:'E03',document_id:'doc-3',document_version:1,line_start:1,line_end:1,cited:true},
    ]}),
    metadata('supports_issue',{retrieval:{mode:'raw-secret-mode'}}),
    metadata('supports_issue',{note:'x'.repeat(301)}),
  ];
  for (const value of invalid) assert.doesNotThrow(() => assert.equal(describeIssueEvidenceReview(value), null));
});

test('enabled review without annotations reports skipped, degraded, and failure safely', () => {
  const cases = [
    [{enabled:true,outcome:'skipped',reason_codes:['chat_not_configured'],selected_issues:2,reviewed_issues:0,skipped_issues:2},'已跳过','复核模型尚未配置'],
    [{enabled:true,outcome:'degraded',reason_codes:['hybrid_unavailable'],selected_issues:2,reviewed_issues:0,skipped_issues:2},'已降级','混合检索不可用'],
    [{enabled:true,outcome:'failed',reason_codes:['internal_failure','secret internal exception'],selected_issues:2,reviewed_issues:0,skipped_issues:2,raw:'sk-secret'},'复核失败','复核过程未能安全完成'],
  ];
  for (const [input,label,reason] of cases) {
    const view = describeIssueEvidenceReviewDiagnostic(input,0);
    assert.equal(view?.label,label);
    assert.match(view?.detail ?? '',new RegExp(reason));
    assert.doesNotMatch(JSON.stringify(view),/secret internal exception|sk-secret/);
  }
  assert.equal(describeIssueEvidenceReviewDiagnostic({enabled:false},0),null);
});

test('contradictory diagnostic counts never imply completion', () => {
  const invalid = [
    [{enabled:true,outcome:'completed',reason_codes:[],selected_issues:0,reviewed_issues:0,skipped_issues:0},1],
    [{enabled:true,outcome:'completed',reason_codes:[],selected_issues:1,reviewed_issues:2,skipped_issues:0},1],
    [{enabled:true,outcome:'partial',reason_codes:[],selected_issues:2,reviewed_issues:1,skipped_issues:0},1],
    [{enabled:true,outcome:'completed',reason_codes:[],selected_issues:2,reviewed_issues:1,skipped_issues:1},2],
    [{enabled:true,outcome:'completed',reason_codes:[],selected_issues:'2',reviewed_issues:1,skipped_issues:1},1],
  ];
  for (const [diagnostic,visible] of invalid) {
    const view=describeIssueEvidenceReviewDiagnostic(diagnostic,visible);
    assert.equal(view?.state,'unknown');
    assert.equal(view?.label,'复核状态未知');
    assert.doesNotMatch(view?.label ?? '',/完成/u);
  }
});
