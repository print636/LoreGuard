export type IssueEvidenceReviewVerdict =
  | 'supports_issue'
  | 'contextual_exception'
  | 'insufficient_evidence';

export type IssueEvidenceReviewEvidenceView = {
  citationId: string;
  documentLabel: string;
  documentVersion: number;
  lineStart: number;
  lineEnd: number;
};

export type IssueEvidenceReviewView = {
  verdict: IssueEvidenceReviewVerdict;
  verdictLabel: string;
  tone: 'supports' | 'exception' | 'insufficient';
  note: string;
  citations: string[];
  evidence: IssueEvidenceReviewEvidenceView[];
  retrievalModeLabel: string;
};

export type IssueEvidenceReviewDiagnosticView = {
  state: 'completed' | 'partial' | 'skipped' | 'degraded' | 'failed' | 'unknown';
  label: string;
  detail: string;
  counts: string;
};

const verdicts: Record<IssueEvidenceReviewVerdict, {label: string; tone: IssueEvidenceReviewView['tone']}> = {
  supports_issue: {label: '支持问题', tone: 'supports'},
  contextual_exception: {label: '存在上下文例外', tone: 'exception'},
  insufficient_evidence: {label: '证据不足', tone: 'insufficient'},
};

const reasonLabels: Record<string, string> = {
  feature_disabled: '该功能未启用',
  chat_not_configured: '复核模型尚未配置',
  not_configured: '复核模型尚未配置',
  token_budget: '本次运行的 Token 预算不足',
  token_budget_overrun: '复核调用超过本次预算上限',
  embedding_not_configured: '向量检索服务尚未配置',
  embedding_failed: '向量证据准备失败',
  embedding_input_rejected: '部分证据不符合向量服务输入要求',
  embedding_response_invalid: '向量服务返回结果不完整',
  embedding_retry_exhausted: '向量服务重试后仍不可用',
  embedding_provider_failed: '向量服务暂时不可用',
  concurrent_write_incomplete: '证据索引尚未完整就绪',
  index_incomplete: '证据索引尚未完整就绪',
  retrieval_failed: '候选证据检索失败',
  hybrid_unavailable: '混合检索不可用，未用降级结果冒充 AI 复核',
  no_evidence: '没有检索到足够证据',
  deadline: '复核超过本次运行的时间上限',
  invalid_model_response: '模型返回结果未通过结构校验',
  internal_failure: '复核过程未能安全完成',
  provider: '模型服务暂时不可用',
  rate_limit: '模型服务触发限流',
  upstream_5xx: '模型服务暂时异常',
  unauthorized: '模型服务鉴权失败',
  forbidden: '模型服务拒绝了请求',
  nonretry_http: '模型服务返回不可重试错误',
  body_json: '模型服务响应格式异常',
  response_shape: '模型服务响应结构异常',
  empty_content: '模型未返回可复核内容',
  usage_shape: '模型用量信息异常',
  truncated: '模型结果被截断',
  content_json: '模型结果不是有效结构化数据',
  connect_timeout: '连接模型服务超时',
  read_timeout: '等待模型结果超时',
  transport: '模型服务连接失败',
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function safeInteger(value: unknown, minimum = 0, maximum = 1_000_000_000): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= minimum && value <= maximum
    ? value
    : null;
}

function safeIdentifier(value: unknown, maxLength = 200): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > maxLength || /[\u0000-\u001f\u007f]/u.test(trimmed)) return null;
  return trimmed;
}

function safePlainText(value: unknown, maxLength: number): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > maxLength) return null;
  return trimmed
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/gu, '')
    .replaceAll('<', '＜')
    .replaceAll('>', '＞');
}

function reviewAnnotation(metadata: unknown): Record<string, unknown> | null {
  if (!isRecord(metadata)) return null;
  const annotation = metadata.ai_evidence_review;
  return isRecord(annotation) ? annotation : null;
}

export function describeIssueEvidenceReview(
  metadata: unknown,
  documentNames: Readonly<Record<string, string>> = {},
): IssueEvidenceReviewView | null {
  const annotation = reviewAnnotation(metadata);
  if (!annotation || annotation.schema_version !== 'issue-evidence-review-v1') return null;

  const verdict = annotation.verdict;
  if (typeof verdict !== 'string' || !(verdict in verdicts)) return null;
  const verdictValue = verdict as IssueEvidenceReviewVerdict;
  const note = safePlainText(annotation.note, 300);
  if (!note) return null;

  if (!Array.isArray(annotation.citations) || annotation.citations.length < 1 || annotation.citations.length > 6) return null;
  const citations = annotation.citations.map(value => safeIdentifier(value, 3));
  if (citations.some(value => value === null)) return null;
  const citationValues = citations as string[];
  if (citationValues.some(value => !/^E[0-9]{2}$/u.test(value)) || new Set(citationValues).size !== citationValues.length) return null;

  const retrieval = annotation.retrieval;
  if (!isRecord(retrieval) || retrieval.mode !== 'hybrid') return null;
  if (!Array.isArray(annotation.consumed_evidence)) return null;

  const evidenceByCitation = new Map<string, IssueEvidenceReviewEvidenceView>();
  for (const value of annotation.consumed_evidence) {
    if (!isRecord(value)) return null;
    const citationId = safeIdentifier(value.citation_id, 3);
    const documentId = safeIdentifier(value.document_id);
    const documentVersion = safeInteger(value.document_version, 1);
    const lineStart = safeInteger(value.line_start, 1);
    const lineEnd = safeInteger(value.line_end, 1);
    if (!citationId || !/^E[0-9]{2}$/u.test(citationId) || !documentId || documentVersion === null || lineStart === null || lineEnd === null || lineEnd < lineStart || typeof value.cited !== 'boolean') return null;
    if (!value.cited) continue;
    if (evidenceByCitation.has(citationId)) return null;
    const friendlyName = safeIdentifier(documentNames[documentId]);
    evidenceByCitation.set(citationId, {
      citationId,
      documentLabel: friendlyName ?? documentId,
      documentVersion,
      lineStart,
      lineEnd,
    });
  }
  const evidence = citationValues.map(citation => evidenceByCitation.get(citation));
  if (evidence.some(value => value === undefined) || evidenceByCitation.size !== citationValues.length) return null;

  return {
    verdict: verdictValue,
    verdictLabel: verdicts[verdictValue].label,
    tone: verdicts[verdictValue].tone,
    note,
    citations: citationValues,
    evidence: evidence as IssueEvidenceReviewEvidenceView[],
    retrievalModeLabel: '关键词 + 向量混合检索',
  };
}

export function describeIssueEvidenceReviewDiagnostic(
  value: unknown,
  visibleAnnotationCount: number,
): IssueEvidenceReviewDiagnosticView | null {
  if (!isRecord(value) || value.enabled !== true) return null;
  const outcome = typeof value.outcome === 'string' && ['completed', 'partial', 'skipped', 'degraded', 'failed'].includes(value.outcome)
    ? value.outcome as Exclude<IssueEvidenceReviewDiagnosticView['state'], 'unknown'>
    : null;
  const selected = safeInteger(value.selected_issues);
  const reviewed = safeInteger(value.reviewed_issues);
  const skipped = safeInteger(value.skipped_issues);
  const annotationCount = safeInteger(visibleAnnotationCount);
  if (
    outcome === null
    || selected === null
    || reviewed === null
    || skipped === null
    || annotationCount === null
    || reviewed > selected
    || skipped > selected
    || reviewed + skipped !== selected
    || annotationCount > reviewed
  ) {
    return {
      state: 'unknown',
      label: '复核状态未知',
      detail: '复核计数未通过一致性校验，不能确认本次完成情况。规则问题保持原样。',
      counts: '未采用不一致的复核计数',
    };
  }
  const reasons = Array.isArray(value.reason_codes)
    ? [...new Set(value.reason_codes.filter((reason): reason is string => typeof reason === 'string' && reason in reasonLabels))]
    : [];
  const detail = reasons.length > 0
    ? reasons.map(reason => reasonLabels[reason]).join('；')
    : annotationCount > 0
      ? '只展示通过前端结构校验的安全注释。'
      : selected === 0 && outcome === 'completed'
        ? '本次没有需要复核的规则问题。'
        : '本次没有生成可安全展示的 AI 注释，规则问题保持原样。';
  const label = annotationCount > 0
    ? outcome === 'completed' ? `已完成 ${annotationCount} 条注释` : `已展示 ${annotationCount} 条注释，部分未完成`
    : outcome === 'skipped' ? '已跳过'
      : outcome === 'degraded' ? '已降级'
        : outcome === 'failed' ? '复核失败'
          : outcome === 'partial' ? '部分完成，但无可展示注释'
            : selected === 0 ? '无需复核' : '已完成，但无可展示注释';
  return {
    state: outcome,
    label,
    detail,
    counts: `选择 ${selected} 条 · 服务端复核 ${reviewed} 条 · 跳过 ${skipped} 条`,
  };
}
