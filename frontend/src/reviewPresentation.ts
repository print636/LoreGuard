export type EvidenceView = {
  document_id: string;
  document_name: string;
  line_start: number;
  line_end: number;
  text: string;
};

export type RecordView = {
  id: string;
  kind: string;
  attrs: Record<string, unknown>;
  evidence: EvidenceView;
};

export type ClarificationView = {
  id: string;
  kind: 'clarification' | 'open_question';
  category: string | null;
  modality: string;
  source_scope: string;
  certainty: string;
  text: string;
  evidence: EvidenceView;
};

const kindNames: Record<string, string> = {
  fact: '事实记录', event: '事件记录', knows: '知识获得',
  claims_knows: '知识声称', item: '物品状态', uses: '物品使用',
  world_rule: '世界规则', world_assert: '章节断言',
  open_question: '原文开放问题', clarification: '待作者澄清',
};

export const modalityNames: Record<string, string> = {
  asserted: '明确陈述', negated: '否定陈述', uncertain: '不确定陈述',
  interrogative: '疑问', hypothetical: '假设', conditional_rule: '条件规则',
  reported: '转述',
};

export const sourceNames: Record<string, string> = {
  narrator: '旁白/正文', world_rule: '权威规则', character_dialogue: '角色对话',
  quoted_material: '引用材料', unverified_report: '未经证实的转述', unknown: '来源未确定',
};

export const certaintyNames: Record<string, string> = {
  certain: '确定', probable: '较可能', possible: '可能', unknown: '未确定',
};

function text(value: unknown): string {
  if (Array.isArray(value)) return value.map(item => String(item)).join('、');
  return typeof value === 'string' || typeof value === 'number' ? String(value).trim() : '';
}

export function recordKindName(kind: string): string {
  return kindNames[kind] ?? '结构化记录';
}

export function summarizeRecord(row: RecordView): string {
  const attrs = row.attrs ?? {};
  const time = text(attrs.time);
  const at = time ? `${time}，` : '';
  if (row.kind === 'fact') {
    return `${at}${text(attrs.subject)}的${text(attrs.predicate)}为“${text(attrs.value)}”`;
  }
  if (row.kind === 'event') {
    const people = text(attrs.participants) || '相关角色';
    const location = text(attrs.location);
    return `${at}${people}${location ? `出现在${location}` : '发生了一项事件'}`;
  }
  if (row.kind === 'knows' || row.kind === 'claims_knows') {
    const verb = row.kind === 'knows' ? '得知' : '声称知道';
    return `${at}${text(attrs.character)}${verb}“${text(attrs.fact)}”`;
  }
  if (row.kind === 'item') {
    return `${at}${text(attrs.item)}由${text(attrs.owner)}持有`;
  }
  if (row.kind === 'uses') {
    return `${at}${text(attrs.user)}使用${text(attrs.item)}`;
  }
  if (row.kind === 'world_rule' || row.kind === 'world_assert') {
    return `${text(attrs.key)}：${text(attrs.value)}`;
  }
  if (row.kind === 'open_question') return text(attrs.question) || row.evidence.text;
  if (row.kind === 'clarification') return text(attrs.summary) || row.evidence.text;
  return row.evidence.text || '这条记录暂时无法生成自然语言摘要。';
}

export function recordDeveloperFields(row: RecordView): Array<[string, string]> {
  return Object.entries(row.attrs ?? {})
    .map(([key, value]) => [key, text(value)] as [string, string])
    .filter(([, value]) => value.length > 0);
}

export function clarificationKindName(kind: ClarificationView['kind']): string {
  return kind === 'clarification' ? '待作者澄清' : '原文开放问题';
}

export function completedResultPaths(runId: string): Record<string, string> {
  const root = `/api/v1/analysis-runs/${runId}`;
  return {
    issues: `${root}/issues`,
    records: `${root}/records`,
    status: root,
    diagnostics: `${root}/diagnostics`,
    clarifications: `${root}/clarifications`,
  };
}

export function visualizationPath(runId: string, kind: 'graph' | 'timeline'): string {
  return `/api/v1/analysis-runs/${runId}/${kind}`;
}
