export const documentRoles = [
  ['chapter', '故事正文/章节'],
  ['canon', '权威世界观设定'],
  ['character_profile', '角色档案'],
  ['reference', '参考材料'],
] as const;

export type DocumentRole = typeof documentRoles[number][0];
export type QuickTextMode = 'body' | 'advanced';

const scopePattern = /^[A-Za-z0-9_\-\u4e00-\u9fff]{1,80}$/u;

export function documentContext(role: DocumentRole, rawScope: string) {
  const story_scope = rawScope.trim() || 'global';
  if (!scopePattern.test(story_scope)) {
    throw new Error('故事作用域只能包含中英文、数字、下划线或连字符，最长 80 位');
  }
  return {document_role: role, story_scope};
}

export function quickTextDocuments(input: {
  mode: QuickTextMode;
  world: string;
  chapter: string;
  role: DocumentRole;
  scope: string;
}) {
  const scope = documentContext(input.role, input.scope).story_scope;
  const chapter = input.chapter.trim();
  if (!chapter) throw new Error('请先输入需要审查的故事正文');
  if (input.mode === 'body') {
    return [{name: 'chapter.md', content: chapter, ...documentContext(input.role, scope)}];
  }
  const documents = [];
  if (input.world.trim()) {
    documents.push({
      name: 'world.md', content: input.world.trim(),
      ...documentContext('canon', scope),
    });
  }
  documents.push({
    name: 'chapter.md', content: chapter,
    ...documentContext('chapter', scope),
  });
  return documents;
}
