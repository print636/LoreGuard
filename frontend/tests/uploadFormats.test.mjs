import test from 'node:test';
import assert from 'node:assert/strict';

import {
  docxImportBoundary, supportedUploadAccept, supportedUploadLabel,
} from '../src/uploadFormats.ts';

test('the file picker advertises every server-supported upload extension', () => {
  assert.equal(supportedUploadAccept, '.md,.txt,.json,.docx');
  assert.match(supportedUploadLabel, /DOCX/);
});

test('the DOCX boundary warns that active and external content is not imported', () => {
  assert.match(docxImportBoundary, /主文档正文段落与表格/);
  assert.match(docxImportBoundary, /宏和外部内容不会导入/);
});
