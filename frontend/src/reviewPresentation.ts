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
  narrative_time_relation: '时间关系', knowledge_constraint: '知情约束',
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
  if (row.kind === 'narrative_time_relation') {
    return row.evidence.text || '这条时间关系需结合原文核对。';
  }
  if (row.kind === 'knowledge_constraint') {
    const subject = `${text(attrs.character)}对“${text(attrs.fact)}”`;
    if (attrs.knowledge_scope !== 'whole_content') {
      return `${subject}的知情依据${attrs.knowledge_scope === 'channel_only' ? '仅限所引途径，其他途径尚未核清' : '范围尚未核清'}`;
    }
    if (attrs.relation === 'unknown_before') return `${time ? `${time}之前，` : ''}${subject}尚不知情`;
    if (attrs.relation === 'unknown_through') return `${time ? `截至${time}，` : ''}${subject}仍不知情`;
    if (attrs.relation === 'known_by') return `${time ? `最迟到${time}，` : ''}${subject}已经知情；这不代表此时才获知`;
    return `${subject}的知情时间仍需核对原文`;
  }
  if (row.kind === 'item') {
    if (attrs.state_kind === 'observed_possession') return `${at}观察到${text(attrs.owner)}持有${text(attrs.item)}；这不代表已核实交接`;
    if (attrs.state_kind === 'custody') return `${at}${text(attrs.item)}由${text(attrs.owner)}保管`;
    if (attrs.state_kind === 'transfer') return `${at}${text(attrs.transfer_from)}将${text(attrs.item)}交给${text(attrs.owner)}`;
    if (attrs.state_kind === 'retained_custody') return `${at}${attrs.through_time ? `截至${text(attrs.through_time)}，` : '所引期间，'}${text(attrs.owner)}仍保管${text(attrs.item)}，明确未交给别人`;
    if (attrs.state_kind) return `${at}${text(attrs.item)}与${text(attrs.owner)}的持有关系尚未核清`;
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
