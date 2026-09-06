import test from 'node:test';
import assert from 'node:assert/strict';
import {
  clarificationKindName, completedResultPaths, recordDeveloperFields,
  summarizeRecord, visualizationPath,
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
