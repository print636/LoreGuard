import test from 'node:test';
import assert from 'node:assert/strict';
import {documentContext, quickTextDocuments} from '../src/documentContext.ts';

test('empty scope becomes the explicit safe global scope', () => {
  assert.deepEqual(documentContext('chapter','  '), {
    document_role:'chapter', story_scope:'global',
  });
  assert.throws(() => documentContext('chapter','route/a'), /故事作用域/);
});

test('body-only mode creates one analyzable document without world canon', () => {
  assert.deepEqual(quickTextDocuments({mode:'body', world:'无须上传',
    chapter:'只有正文也能检查。', role:'chapter', scope:'global'}), [{
      name:'chapter.md', content:'只有正文也能检查。',
      document_role:'chapter', story_scope:'global',
    }]);
});

test('advanced mode sends explicit roles and shared branch scope', () => {
  const rows = quickTextDocuments({mode:'advanced', world:'规则。',
    chapter:'正文。', role:'reference', scope:'route_a'});
  assert.deepEqual(rows.map(row => [row.name,row.document_role,row.story_scope]), [
    ['world.md','canon','route_a'], ['chapter.md','chapter','route_a'],
  ]);
});
