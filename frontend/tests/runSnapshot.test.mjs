import test from 'node:test';
import assert from 'node:assert/strict';
import {
  hasRunSnapshot, retryLineage, retryState, runInputState,
  runSnapshotDocuments, shortIdentifier, snapshotDocumentLabels,
} from '../src/runSnapshot.ts';

const frozen={document_id:'doc-v1',document_name:'chapter.md',document_version:1,
  document_role:'chapter',story_scope:'route_a',content_sha256:'1234567890abcdef',
  char_count:42,ordinal:0};

test('available snapshots expose frozen version, context, and abbreviated hash', () => {
  const run={id:'R1',input_snapshot_available:true,input_documents:[frozen]};
  assert.equal(hasRunSnapshot(run),true);
  assert.equal(runInputState(run),'1 个冻结输入');
  assert.deepEqual(snapshotDocumentLabels(frozen),{
    identity:'chapter.md · v1',context:'role 故事正文 (chapter) · scope route_a',hash:'1234567890ab…',
  });
  assert.deepEqual(retryState(run),{allowed:true,label:'按本次输入重试（可能消耗 Token）'});
});

test('legacy runs never borrow current documents and cannot claim same-input retry', () => {
  const legacy={id:'legacy',input_snapshot_available:false};
  assert.deepEqual(runSnapshotDocuments(legacy),[]);
  assert.equal(runInputState(legacy),'输入未知');
  assert.deepEqual(retryState(legacy),{allowed:false,label:'不可同输入重试'});
});

test('inconsistent available flag with no documents stays conservative', () => {
  assert.equal(hasRunSnapshot({id:'broken',input_snapshot_available:true,input_documents:[]}),false);
  assert.equal(runInputState({id:'broken',input_documents:[frozen]}),'输入未知');
});

test('retry lineage and distinct versions remain explicit', () => {
  assert.equal(retryLineage({id:'R2',retried_from:'1234567890abcdef'}),'重试继承自 1234567890ab…');
  assert.notEqual(snapshotDocumentLabels(frozen).identity,
    snapshotDocumentLabels({...frozen,document_id:'doc-v2',document_version:2,content_sha256:'fedcba0987654321'}).identity);
  assert.equal(shortIdentifier('R3'), 'R3');
});
