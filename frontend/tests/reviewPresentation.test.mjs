import test from 'node:test';
import assert from 'node:assert/strict';
import {
  clarificationKindName, completedResultPaths, recordDeveloperFields,
  summarizeRecord, visualizationPath, recordKindName,
} from '../src/reviewPresentation.ts';

const evidence = {document_id:'d1', document_name:'chapter.md', line_start:7,
  line_end:7, text:'林澈的身份是领航员。'};

test('record card uses a natural summary and hides empty time by default', () => {
  const row = {id:'r1', kind:'fact', evidence, attrs:{time:'', subject:'林澈',
    predicate:'身份', value:'领航员', modality:'asserted'}};
  assert.equal(summarizeRecord(row), '林澈的身份为“领航员”');
  assert.doesNotMatch(summarizeRecord(row), /time=|modality=/);
  assert.deepEqual(recordDeveloperFields(row), [
    ['subject','林澈'], ['predicate','身份'], ['value','领航员'],
    ['modality','asserted'],
  ]);
});

test('clarifications have a separate natural label from confirmed issues', () => {
  assert.equal(clarificationKindName('clarification'), '待作者澄清');
  assert.equal(clarificationKindName('open_question'), '原文开放问题');
  const paths = completedResultPaths('run-1');
  assert.equal(paths.issues, '/api/v1/analysis-runs/run-1/issues');
  assert.equal(paths.clarifications, '/api/v1/analysis-runs/run-1/clarifications');
  assert.notEqual(paths.issues, paths.clarifications);
});

test('completed result loading never includes graph or timeline requests', () => {
  const urls = Object.values(completedResultPaths('run-1'));
  assert.equal(urls.some(url => /\/(graph|timeline)$/.test(url)), false);
  assert.equal(visualizationPath('run-1','graph'), '/api/v1/analysis-runs/run-1/graph');
  assert.equal(visualizationPath('run-1','timeline'), '/api/v1/analysis-runs/run-1/timeline');
});

test('time and knowledge records describe bounds without internal identities or invented acquisition', () => {
  assert.equal(recordKindName('narrative_time_relation'), '时间关系');
  assert.equal(recordKindName('knowledge_constraint'), '知情约束');
  const row = {id:'constraint', kind:'knowledge_constraint', evidence,
    attrs:{character:'乔予', fact:'坐标', time:'14:00', relation:'known_by', knowledge_scope:'whole_content'}};
  assert.equal(summarizeRecord(row), '最迟到14:00，乔予对“坐标”已经知情；这不代表此时才获知');
  assert.match(summarizeRecord({...row, attrs:{...row.attrs, relation:'unknown_before'}}), /14:00之前.*尚不知情/);
  assert.match(summarizeRecord({...row, attrs:{...row.attrs, relation:'unknown_through'}}), /截至14:00.*仍不知情/);
  assert.match(summarizeRecord({...row, attrs:{...row.attrs, knowledge_scope:'channel_only'}}), /其他途径尚未核清/);
  assert.match(summarizeRecord({...row, attrs:{...row.attrs, relation:'unclear'}}), /仍需核对原文/);
  const relation = {...row, kind:'narrative_time_relation', evidence:{...evidence, text:'这两件事发生在同一天。'},
    attrs:{doc_ref:'d2', left_time:'12:00', right_time:'12:00', relation:'same_day'}};
  assert.equal(summarizeRecord(relation), '这两件事发生在同一天。');
  assert.doesNotMatch(summarizeRecord(relation), /doc_ref|same_day|d2/);
});
