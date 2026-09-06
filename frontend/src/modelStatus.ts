export type ReviewAgentTraceDiagnostics = {
  action?: string;
  round?: number;
  candidate_hash?: string | null;
  doc_ref?: string | null;
  line_start?: number | null;
  line_end?: number | null;
  span_hash?: string | null;
  fields?: string[];
  validator_reason?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  elapsed_ms?: number;
  final?: string;
};

export type ReviewAgentRunDiagnostics = {
  protocol?: string;
  orchestrator?: string;
  decision_rounds?: number;
  tool_calls?: number;
  span_chars?: number;
  span_read_count?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  charged_tokens?: number;
  recovered_records?: number;
  unresolved_records?: number;
  abstained_records?: number;
  final_reason?: string;
  total_trace_events?: number;
  trace_truncated?: boolean;
  trace?: ReviewAgentTraceDiagnostics[];
};

export type RepairDiagnostics = {
  attempted?: boolean;
  succeeded?: boolean;
  failed?: boolean;
  skipped_reason?: string | null;
  pre_invalid?: number;
  post_invalid?: number;
  salvaged?: number;
  dropped?: number;
  observed_invalid?: number | null;
  unresolved_invalid?: number | null;
  recovered_invalid?: number | null;
  final_path?: string;
};

export type ModelDiagnostics = {
  used?: boolean; partial_fallback?: boolean; mode?: string;
  enabled?: boolean; configured?: boolean; total_chunks?: number;
  attempted_chunks?: number; succeeded_chunks?: number; failed_chunks?: number;
  skipped_chunks?: number; invalid_records?: number; empty_response_chunks?: number;
  unresolved_invalid_records?: number | null; recovered_invalid_records?: number | null;
  repair_attempted?: boolean; repair_succeeded?: boolean; repair_failed?: boolean;
  repair_skipped_reason?: string | null;
  repair_pre_invalid?: number; repair_post_invalid?: number;
  repair_salvaged?: number; repair_dropped?: number; repair_final_path?: string;
  repair?: RepairDiagnostics;
  review_agent_attempted?: boolean;
  review_agent_succeeded?: boolean;
  review_agent_abstained?: boolean;
  review_agent_runs?: ReviewAgentRunDiagnostics[];
  review_agent_total_runs?: number;
  review_agent_runs_truncated?: boolean;
  reason_codes?: string[];
};

export type ModelCoverage = 'full' | 'partial' | 'limited' | 'unknown';

export type ModelStatusView = {
  coverage: ModelCoverage;
  label: string;
  detail: string;
  counts: string;
  emptyCaveat: string;
};

export type RepairStatusView = {
  state: 'unknown' | 'not_attempted' | 'succeeded' | 'incomplete';
  label: string;
  detail: string;
  counts: string;
};

export type ReviewAgentTraceView = {
  label: string;
  metrics: string;
};

export type ReviewAgentRunView = {
  index: number;
  outcome: string;
  protocol: string;
  orchestrator: string;
  activity: string;
  tokens: string;
  finalReason: string;
  traceCount: string;
  trace: ReviewAgentTraceView[];
};

export type ReviewAgentStatusView = {
  supplied: boolean;
  valid: boolean;
  state: 'unknown' | 'not_attempted' | 'succeeded' | 'abstained' | 'failed' | 'mixed';
  label: string;
  detail: string;
  counts: string;
  coverageProven: boolean;
  proofDetail: string;
  runsTruncated: boolean;
  runs: ReviewAgentRunView[];
};

const reasons: Record<string, string> = {
  disabled: '模型已关闭', not_configured: '未配置模型', token_budget: 'Token 预算不足',
  chunk_limit: '分块超出上限', circuit_open: '熔断已开启', provider_error: '模型服务失败',
  document_aborted: '当前文档后续分块停止', schema_validation: '结构校验失败',
  record_validation: '记录校验失败', evidence_range: '证据行号越界',
  empty_evidence: '证据为空', lexical_support: '缺少原文支持',
  repair_failed: '语义标签修复失败', repair_run_limit: '语义标签修复次数受限',
};

const agentReasonNames: Record<string, string> = {
  accepted_protocol: '协议通过', read_ok: '证据范围读取完成', patch_ok: '补丁校验通过',
  explicit_abstain: '明确弃答', invalid_json: '应用层 JSON 无效', invalid_action: '动作无效',
  unknown_tool: '工具动作未知', cross_document: '跨文档请求被拒绝',
  evidence_range: '证据范围越界', span_budget: '读取字符预算耗尽',
  span_count_budget: '读取次数预算耗尽', tool_budget: '工具调用预算耗尽',
  token_budget: 'Token 预算耗尽', deadline: '总时限到期', provider_error: '模型服务失败',
  response_too_large: '响应超过字节上限', repeated_loop: '重复循环已停止',
  patch_field_forbidden: '补丁字段不允许', patch_duplicate_candidate: '候选补丁重复',
  patch_validation_failed: '补丁校验失败', read_required: '补丁前缺少证据读取',
  invalid_span: '读取范围无效', round_limit: '决策轮数耗尽', completed: '流程完成',
};

const actionNames: Record<string, string> = {
  DECISION: '决策', READ_SPAN: '读取证据范围', PATCH_RECORDS: '提交记录补丁',
  ABSTAIN: '弃答', FINALIZE: '结束',
};

const finalNames: Record<string, string> = {
  continue: '继续', accepted: '已采纳', abstained: '已弃答', rejected: '已拒绝',
};

const finalPathNames: Record<string, string> = {
  not_needed: '未触发', repaired: '修复完成', partial_repair: '部分修复',
  salvaged_and_baseline: '部分保留并回退基线', salvaged: '部分保留', baseline: '回退基线',
};

const agentKeys: Array<keyof ModelDiagnostics> = [
  'review_agent_attempted', 'review_agent_succeeded', 'review_agent_abstained',
  'review_agent_runs', 'review_agent_total_runs', 'review_agent_runs_truncated',
];
const agentActions = new Set(Object.keys(actionNames));
const agentFinals = new Set(Object.keys(finalNames));
const agentReasons = new Set(Object.keys(agentReasonNames));
const protocols = new Set(['application_json_tools_v1', 'unknown']);
const orchestrators = new Set(['langgraph_stategraph', 'unknown']);
const repairPaths = new Set(Object.keys(finalPathNames));
const safeField = /^[a-z][a-z0-9_]{0,63}$/;
const safeHash = /^[a-f0-9]{64}$/;
const safeDocRef = /^[A-Za-z0-9._-]{1,16}$/;

function hasOwn(source: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(source, key);
}

function isNonnegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0;
}

function unknownAgent(supplied: boolean, detail: string): ReviewAgentStatusView {
  return {
    supplied,
    valid: false,
    state: 'unknown',
    label: '受限证据修复 Agent 状态未知',
    detail,
    counts: 'attempted unknown · succeeded unknown · abstained unknown · Run unknown',
    coverageProven: false,
    proofDetail: 'Agent 诊断不可用于证明完整覆盖。',
    runsTruncated: false,
    runs: [],
  };
}

function normalizeTrace(row: unknown): ReviewAgentTraceView | null {
  if (!row || typeof row !== 'object') return null;
  const source = row as ReviewAgentTraceDiagnostics;
  const counters = [source.round, source.prompt_tokens, source.completion_tokens, source.elapsed_ms];
  if (!agentActions.has(source.action ?? '') || !agentFinals.has(source.final ?? '') ||
      !agentReasons.has(source.validator_reason ?? '') ||
      counters.some(value => !isNonnegativeInteger(value)) ||
      !Array.isArray(source.fields) || source.fields.length > 20 ||
      source.fields.some(field => typeof field !== 'string' || !safeField.test(field))) {
    return null;
  }
  if ((source.candidate_hash != null && (typeof source.candidate_hash !== 'string' ||
      !safeHash.test(source.candidate_hash))) ||
      (source.doc_ref != null && (typeof source.doc_ref !== 'string' ||
      !safeDocRef.test(source.doc_ref))) ||
      (source.span_hash != null && (typeof source.span_hash !== 'string' ||
      !safeHash.test(source.span_hash)))) {
    return null;
  }
  const hasStart = source.line_start != null;
  const hasEnd = source.line_end != null;
  if (hasStart !== hasEnd || (hasStart && (!isNonnegativeInteger(source.line_start) ||
      !isNonnegativeInteger(source.line_end) || source.line_end! < source.line_start!))) {
    return null;
  }
  const fieldSummary = source.fields.length ? ` · 字段 ${source.fields.join('、')}` : '';
  const spanSummary = hasStart ? ` · 行范围 ${source.line_start}–${source.line_end}` : '';
  return {
    label: `第 ${source.round} 轮 · ${actionNames[source.action!]} · ${agentReasonNames[source.validator_reason!]} · ${finalNames[source.final!]}`,
    metrics: `Token ${source.prompt_tokens}+${source.completion_tokens} · ${source.elapsed_ms} ms${spanSummary}${fieldSummary}`,
  };
}

function normalizeRun(row: unknown, index: number): ReviewAgentRunView | null {
  if (!row || typeof row !== 'object') return null;
  const source = row as ReviewAgentRunDiagnostics;
  const counters = [source.decision_rounds, source.tool_calls, source.span_chars,
    source.span_read_count, source.prompt_tokens, source.completion_tokens, source.charged_tokens,
    source.recovered_records, source.unresolved_records, source.abstained_records,
    source.total_trace_events];
  if (!protocols.has(source.protocol ?? '') || !orchestrators.has(source.orchestrator ?? '') ||
      !agentReasons.has(source.final_reason ?? '') ||
      counters.some(value => !isNonnegativeInteger(value)) ||
      typeof source.trace_truncated !== 'boolean' || !Array.isArray(source.trace) ||
      source.trace.length > 64 || source.total_trace_events! < source.trace.length ||
      (!source.trace_truncated && source.total_trace_events !== source.trace.length)) {
    return null;
  }
  const trace = source.trace.map(normalizeTrace);
  if (trace.some(item => item === null)) return null;
  const recovered = source.recovered_records!;
  const unresolved = source.unresolved_records!;
  const abstained = source.abstained_records!;
  let outcome = '未采纳记录补丁';
  if (recovered > 0 && abstained > 0) outcome = `Agent 恢复 ${recovered} 条，并弃答 ${abstained} 条`;
  else if (recovered > 0) outcome = `Agent 恢复 ${recovered} 条`;
  else if (abstained > 0 || source.final_reason === 'explicit_abstain') outcome = `Agent 弃答 ${abstained || 1} 条`;
  else if (unresolved > 0) outcome = `仍有 ${unresolved} 条未解决`;
  return {
    index,
    outcome,
    protocol: source.protocol === 'application_json_tools_v1'
      ? '应用层 JSON 工具协议 v1（非原生 function calling）'
      : '应用层工具协议未知（不能视为原生 function calling）',
    orchestrator: source.orchestrator === 'langgraph_stategraph'
      ? 'LangGraph StateGraph 编排' : '编排器未知',
    activity: `决策轮 ${source.decision_rounds} · 工具调用 ${source.tool_calls} · span read ${source.span_read_count} 次 / ${source.span_chars} 字符`,
    tokens: `Token：prompt ${source.prompt_tokens} · completion ${source.completion_tokens} · charged ${source.charged_tokens}`,
    finalReason: agentReasonNames[source.final_reason!],
    traceCount: `Trace：展示 ${source.trace.length} / 总计 ${source.total_trace_events}${source.trace_truncated ? '（已截断）' : ''}`,
    trace: trace as ReviewAgentTraceView[],
  };
}

function traceProvesRecoveredRecords(run: ReviewAgentRunDiagnostics): boolean {
  if (run.protocol !== 'application_json_tools_v1' ||
      run.orchestrator !== 'langgraph_stategraph' || run.final_reason !== 'completed' ||
      run.trace_truncated !== false || !Array.isArray(run.trace) ||
      run.total_trace_events !== run.trace.length || !isNonnegativeInteger(run.recovered_records) ||
      run.recovered_records === 0 || run.unresolved_records !== 0 || run.abstained_records !== 0) {
    return false;
  }
  const decisions = run.trace.filter(trace => trace.action === 'DECISION');
  const reads = run.trace.filter(trace => trace.action === 'READ_SPAN');
  const toolEvents = run.trace.filter(trace =>
    trace.action === 'READ_SPAN' || trace.action === 'PATCH_RECORDS' ||
    trace.action === 'ABSTAIN');
  const tracedPrompt = decisions.reduce((total, trace) => total + (trace.prompt_tokens ?? 0), 0);
  const tracedCompletion = decisions.reduce(
    (total, trace) => total + (trace.completion_tokens ?? 0), 0);
  if (decisions.length !== run.decision_rounds || reads.length !== run.span_read_count ||
      !isNonnegativeInteger(run.tool_calls) || run.tool_calls === 0 ||
      run.tool_calls > toolEvents.length || tracedPrompt !== run.prompt_tokens ||
      tracedCompletion !== run.completion_tokens || !isNonnegativeInteger(run.charged_tokens) ||
      run.charged_tokens < tracedPrompt + tracedCompletion) {
    return false;
  }
  const acceptedPatches = run.trace
    .map((trace, index) => ({trace, index}))
    .filter(({trace}) => trace.action === 'PATCH_RECORDS' &&
      trace.validator_reason === 'patch_ok' && trace.final === 'accepted');
  if (acceptedPatches.length !== run.recovered_records) return false;
  const recoveredKeys = new Set(acceptedPatches.map(({trace}) =>
    `${trace.candidate_hash ?? ''}:${trace.doc_ref ?? ''}`));
  if (recoveredKeys.size !== acceptedPatches.length) return false;
  return acceptedPatches.every(({trace: patch, index}) => {
    if (!patch.candidate_hash || !patch.doc_ref || !patch.span_hash) return false;
    return run.trace!.slice(0, index).some(read =>
      read.action === 'READ_SPAN' && read.validator_reason === 'read_ok' &&
      read.candidate_hash === patch.candidate_hash && read.doc_ref === patch.doc_ref &&
      read.span_hash === patch.span_hash && isNonnegativeInteger(read.line_start) &&
      isNonnegativeInteger(read.line_end) && read.line_start === patch.line_start &&
      read.line_end === patch.line_end);
  });
}

function fixedRepairRecovered(model: ModelDiagnostics): number | null {
  const nested = model.repair;
  if (!nested || typeof model.repair_attempted !== 'boolean' ||
      typeof model.repair_succeeded !== 'boolean' || typeof model.repair_failed !== 'boolean' ||
      typeof nested.attempted !== 'boolean' || typeof nested.succeeded !== 'boolean' ||
      typeof nested.failed !== 'boolean' || !isNonnegativeInteger(model.repair_pre_invalid) ||
      !isNonnegativeInteger(model.repair_post_invalid) ||
      !isNonnegativeInteger(model.repair_salvaged) ||
      !isNonnegativeInteger(model.repair_dropped) || !isNonnegativeInteger(nested.pre_invalid) ||
      !isNonnegativeInteger(nested.post_invalid) || !isNonnegativeInteger(nested.salvaged) ||
      !isNonnegativeInteger(nested.dropped) || typeof model.repair_final_path !== 'string' ||
      typeof nested.final_path !== 'string') {
    return null;
  }
  const consistent = model.repair_attempted === nested.attempted &&
    model.repair_succeeded === nested.succeeded && model.repair_failed === nested.failed &&
    model.repair_pre_invalid === nested.pre_invalid &&
    model.repair_post_invalid === nested.post_invalid &&
    model.repair_salvaged === nested.salvaged && model.repair_dropped === nested.dropped &&
    model.repair_final_path === nested.final_path;
  if (!consistent) return null;
  if (!model.repair_attempted) {
    return !model.repair_succeeded && !model.repair_failed &&
      model.repair_pre_invalid === 0 && model.repair_post_invalid === 0 &&
      model.repair_salvaged === 0 && model.repair_dropped === 0 &&
      model.repair_final_path === 'not_needed' ? 0 : null;
  }
  if (model.repair_succeeded && !model.repair_failed &&
      model.repair_pre_invalid > 0 && model.repair_post_invalid === 0 &&
      model.repair_salvaged === 0 && model.repair_dropped === 0 &&
      model.repair_final_path === 'repaired') {
    return model.repair_pre_invalid;
  }
  return null;
}

function agentCoverageProof(model: ModelDiagnostics): {proven: boolean; detail: string} {
  if (!model.review_agent_attempted || !model.review_agent_succeeded ||
      model.review_agent_abstained || model.review_agent_runs_truncated ||
      !isNonnegativeInteger(model.review_agent_total_runs) ||
      model.review_agent_total_runs === 0 || !Array.isArray(model.review_agent_runs) ||
      model.review_agent_runs.length !== model.review_agent_total_runs) {
    return {proven: false, detail: '成功汇总缺少完整、非截断且非空的 Run 证据。'};
  }
  if (!model.review_agent_runs.every(traceProvesRecoveredRecords)) {
    return {proven: false, detail: '至少一个 Run 的协议、编排、终态、计数或 READ_SPAN → accepted PATCH_RECORDS trace 不可完整回放。'};
  }
  const fixedRecovered = fixedRepairRecovered(model);
  if (!isNonnegativeInteger(model.recovered_invalid_records) || fixedRecovered === null) {
    return {proven: false, detail: '无法从总恢复数中守恒扣除可证明的固定标签 repair 恢复份额。'};
  }
  const agentRecovered = model.recovered_invalid_records - fixedRecovered;
  const runRecovered = model.review_agent_runs.reduce(
    (total, run) => total + (run.recovered_records ?? 0), 0);
  if (agentRecovered < 0 || agentRecovered !== runRecovered) {
    return {proven: false, detail: 'Agent Run 恢复数与扣除固定标签 repair 后的恢复份额不守恒。'};
  }
  return {proven: true, detail: `完整 Run trace 可回放，Agent 恢复 ${agentRecovered} 条且处置计数守恒。`};
}

export function describeReviewAgentStatus(model?: ModelDiagnostics): ReviewAgentStatusView {
  if (!model) return unknownAgent(false, '没有模型诊断；不能推断 Agent 是否运行。');
  const supplied = agentKeys.some(key => hasOwn(model, key));
  if (!supplied) {
    return unknownAgent(false, '旧运行未提供 Agent 结构化诊断；保持 unknown，不将缺失字段当作未运行。');
  }
  if (typeof model.review_agent_attempted !== 'boolean' ||
      typeof model.review_agent_succeeded !== 'boolean' ||
      typeof model.review_agent_abstained !== 'boolean' ||
      typeof model.review_agent_runs_truncated !== 'boolean' ||
      !isNonnegativeInteger(model.review_agent_total_runs) ||
      !Array.isArray(model.review_agent_runs)) {
    return unknownAgent(true, 'Agent 汇总字段缺失或非法；不会据此提高语义覆盖等级。');
  }
  const runs = model.review_agent_runs.map((row, index) => normalizeRun(row, index + 1));
  const inconsistent = runs.some(row => row === null) ||
    model.review_agent_total_runs < model.review_agent_runs.length ||
    (!model.review_agent_runs_truncated &&
      model.review_agent_total_runs !== model.review_agent_runs.length) ||
    (!model.review_agent_attempted && (model.review_agent_succeeded ||
      model.review_agent_abstained || model.review_agent_total_runs > 0)) ||
    (model.review_agent_succeeded && !model.review_agent_attempted) ||
    (model.review_agent_abstained && !model.review_agent_attempted);
  if (inconsistent) {
    return unknownAgent(true, 'Agent Run、汇总状态或截断计数彼此矛盾；不会据此提高语义覆盖等级。');
  }
  const counts = `attempted ${model.review_agent_attempted ? 'yes' : 'no'} · ` +
    `succeeded ${model.review_agent_succeeded ? 'yes' : 'no'} · ` +
    `abstained ${model.review_agent_abstained ? 'yes' : 'no'} · ` +
    `Run：展示 ${model.review_agent_runs.length} / 总计 ${model.review_agent_total_runs}` +
    (model.review_agent_runs_truncated ? '（已截断）' : '');
  if (!model.review_agent_attempted) {
    return {
      supplied: true, valid: true, state: 'not_attempted',
      label: '受限证据修复 Agent 未触发',
      detail: '没有候选进入受限证据 Agent；这与固定语义标签 repair 是独立路径。',
      counts, coverageProven: false, proofDetail: 'Agent 未触发，不参与覆盖提升。',
      runsTruncated: false, runs: [],
    };
  }
  let state: ReviewAgentStatusView['state'] = 'failed';
  let label = '受限证据修复 Agent 未成功';
  let detail = 'Agent 已尝试，但没有成功采纳记录补丁；不会将这次尝试计为完整覆盖。';
  const proof = agentCoverageProof(model);
  if (model.review_agent_succeeded && model.review_agent_abstained) {
    state = 'mixed';
    label = '受限证据修复 Agent 部分采纳、部分弃答';
    detail = '部分候选由 Agent 恢复，另有候选被保守弃答；Agent recovery 不是固定语义标签 repair。';
  } else if (model.review_agent_succeeded) {
    state = 'succeeded';
    label = proof.proven
      ? '受限证据修复 Agent 已采纳补丁'
      : '受限证据修复 Agent 报告成功，覆盖证据不足';
    detail = proof.proven
      ? 'Agent 在读取受限证据范围后采纳了校验通过的记录补丁；Agent recovery 不是固定语义标签 repair。'
      : '汇总声称 Agent 成功，但安全 Run 证据不足；可以展示诊断，不能据此提高完整覆盖。';
  } else if (model.review_agent_abstained) {
    state = 'abstained';
    label = '受限证据修复 Agent 已弃答';
    detail = 'Agent 已尝试并保守弃答，没有把未确认候选写成已修复记录。';
  }
  return {
    supplied: true, valid: true, state, label, detail, counts,
    coverageProven: proof.proven,
    proofDetail: proof.detail,
    runsTruncated: model.review_agent_runs_truncated,
    runs: runs as ReviewAgentRunView[],
  };
}

export function describeRepairStatus(model?: ModelDiagnostics): RepairStatusView {
  if (!model) {
    return {state: 'unknown', label: '固定语义标签 repair 状态未知',
      detail: '没有 repair 结构化诊断。', counts: ''};
  }
  const nested = model.repair;
  const source: RepairDiagnostics | undefined = nested ?? (
    hasOwn(model, 'repair_attempted') ? {
      attempted: model.repair_attempted, succeeded: model.repair_succeeded,
      failed: model.repair_failed, skipped_reason: model.repair_skipped_reason,
      pre_invalid: model.repair_pre_invalid, post_invalid: model.repair_post_invalid,
      salvaged: model.repair_salvaged, dropped: model.repair_dropped,
      final_path: model.repair_final_path,
    } : undefined
  );
  if (!source) {
    return {state: 'unknown', label: '固定语义标签 repair 状态未知',
      detail: '旧运行未提供独立 repair 诊断。', counts: ''};
  }
  const flatRepairSupplied = ['repair_attempted', 'repair_succeeded', 'repair_failed',
    'repair_pre_invalid', 'repair_post_invalid', 'repair_salvaged', 'repair_dropped',
    'repair_final_path'].some(key => hasOwn(model, key));
  if (nested && flatRepairSupplied) {
    const flatComplete = typeof model.repair_attempted === 'boolean' &&
      typeof model.repair_succeeded === 'boolean' && typeof model.repair_failed === 'boolean' &&
      isNonnegativeInteger(model.repair_pre_invalid) &&
      isNonnegativeInteger(model.repair_post_invalid) &&
      isNonnegativeInteger(model.repair_salvaged) && isNonnegativeInteger(model.repair_dropped) &&
      typeof model.repair_final_path === 'string';
    const matches = flatComplete && model.repair_attempted === nested.attempted &&
      model.repair_succeeded === nested.succeeded && model.repair_failed === nested.failed &&
      model.repair_pre_invalid === nested.pre_invalid &&
      model.repair_post_invalid === nested.post_invalid &&
      model.repair_salvaged === nested.salvaged && model.repair_dropped === nested.dropped &&
      model.repair_final_path === nested.final_path;
    if (!matches) {
      return {state: 'unknown', label: '固定语义标签 repair 状态未知',
        detail: 'repair 的扁平汇总与嵌套诊断不一致。', counts: ''};
    }
  }
  const counters = [source.pre_invalid, source.post_invalid, source.salvaged, source.dropped];
  if (typeof source.attempted !== 'boolean' || typeof source.succeeded !== 'boolean' ||
      typeof source.failed !== 'boolean' || counters.some(value => !isNonnegativeInteger(value)) ||
      !repairPaths.has(source.final_path ?? '') || source.post_invalid! > source.pre_invalid! ||
      (source.succeeded && source.pre_invalid === 0) ||
      (!source.succeeded && source.post_invalid! < source.pre_invalid!) ||
      (!source.attempted && (source.succeeded || source.failed || source.pre_invalid! > 0 ||
        source.post_invalid! > 0 || source.salvaged! > 0 || source.dropped! > 0 ||
        source.final_path !== 'not_needed'))) {
    return {state: 'unknown', label: '固定语义标签 repair 状态未知',
      detail: 'repair 汇总字段缺失、非法或互相矛盾。', counts: ''};
  }
  const counts = `输入无效 ${source.pre_invalid} · repair 后无效 ${source.post_invalid} · ` +
    `保留 ${source.salvaged} · 丢弃 ${source.dropped} · 路径 ${finalPathNames[source.final_path!]}`;
  if (!source.attempted) {
    return {state: 'not_attempted', label: '固定语义标签 repair 未触发',
      detail: '本路径只修补固定的 modality/source_scope/certainty 语义标签，不是 Agent。', counts};
  }
  if (source.succeeded && !source.failed && source.post_invalid === 0) {
    return {state: 'succeeded', label: '固定语义标签 repair 已完成',
      detail: '已完成固定标签修补；其结果与受限证据修复 Agent 分开计量。', counts};
  }
  return {state: 'incomplete', label: '固定语义标签 repair 未完整完成',
    detail: '仍有未解决、保留或回退记录；不能把它描述为 Agent 成功。', counts};
}

export function describeModelStatus(model?: ModelDiagnostics): ModelStatusView {
  const values = [model?.total_chunks, model?.attempted_chunks, model?.succeeded_chunks,
    model?.failed_chunks, model?.skipped_chunks, model?.invalid_records,
    model?.empty_response_chunks];
  if (!model || values.some(value => !isNonnegativeInteger(value))) {
    return {
      coverage: 'unknown', label: 'AI 语义审查执行状态未知',
      detail: '缺少结构化调用记录，不能据此判断模型是否参与或覆盖了多少正文。', counts: '',
      emptyCaveat: '执行覆盖情况未知，当前结果不能证明没有其他问题。',
    };
  }
  const agent = describeReviewAgentStatus(model);
  if (agent.supplied && !agent.valid) {
    return {
      coverage: 'unknown', label: 'AI 语义审查执行状态未知',
      detail: '受限证据修复 Agent 诊断不完整或矛盾，不能将这次运行标记为完整覆盖。', counts: '',
      emptyCaveat: '执行记录不完整，当前结果不能证明没有其他问题。',
    };
  }
  const total = model.total_chunks!;
  const attempted = model.attempted_chunks!;
  const succeeded = model.succeeded_chunks!;
  const failed = model.failed_chunks!;
  const skipped = model.skipped_chunks!;
  const invalid = model.invalid_records!;
  const empty = model.empty_response_chunks!;
  if (attempted !== succeeded + failed || total !== attempted + skipped || empty > succeeded ||
      (succeeded > 0 && (!model.enabled || !model.configured))) {
    return {
      coverage: 'unknown', label: 'AI 语义审查执行状态未知',
      detail: '结构化调用计数彼此矛盾，不能将这次运行标记为完整或降级。', counts: '',
      emptyCaveat: '执行记录不完整，当前结果不能证明没有其他问题。',
    };
  }
  const dispositionSupplied = model.unresolved_invalid_records != null ||
    model.recovered_invalid_records != null;
  let unresolved = invalid;
  let recovered = 0;
  if (dispositionSupplied) {
    const invalidDisposition = [model.unresolved_invalid_records, model.recovered_invalid_records,
      model.repair_post_invalid].some(value => !isNonnegativeInteger(value)) ||
      typeof model.repair_failed !== 'boolean';
    if (invalidDisposition) {
      return {
        coverage: 'unknown', label: 'AI 语义审查执行状态未知',
        detail: '无效记录的最终处置计数缺失或非法，不能判断模型覆盖是否完整。', counts: '',
        emptyCaveat: '执行记录不完整，当前结果不能证明没有其他问题。',
      };
    }
    unresolved = model.unresolved_invalid_records!;
    recovered = model.recovered_invalid_records!;
    if (invalid !== unresolved + recovered) {
      return {
        coverage: 'unknown', label: 'AI 语义审查执行状态未知',
        detail: '原始无效、已恢复与最终未解决记录不守恒，不能判断模型覆盖是否完整。', counts: '',
        emptyCaveat: '执行记录不完整，当前结果不能证明没有其他问题。',
      };
    }
  }
  const reasonDetail = (model.reason_codes ?? [])
    .map(code => reasons[code] ?? '其他执行限制').join('；');
  const invalidCounts = dispositionSupplied
    ? `原始无效 ${invalid} · 最终未解决 ${unresolved} · 总恢复 ${recovered} · 固定 repair 路径 ${model.repair_final_path ?? 'not_needed'}`
    : `拒绝记录 ${invalid}`;
  const counts = `计划 ${total} · 尝试 ${attempted} · 成功 ${succeeded} · ` +
    `失败 ${failed} · 跳过 ${skipped} · ${invalidCounts} · 空响应 ${empty}`;
  const repairIncomplete = dispositionSupplied &&
    (model.repair_failed === true || model.repair_post_invalid! > 0);
  const agentIncomplete = agent.supplied && agent.valid && agent.state !== 'not_attempted' &&
    !agent.coverageProven;
  if (succeeded > 0 && failed === 0 && skipped === 0 && unresolved === 0 &&
      !repairIncomplete && !agentIncomplete && empty === 0) {
    return {
      coverage: 'full', label: 'AI 语义审查完整',
      detail: recovered > 0
        ? `所有计划分块均完成；共恢复 ${recovered} 条原始无效记录，最终未解决为 0。恢复来源见下方两条独立诊断链；这不代表结论绝对正确。`
        : reasonDetail || '所有计划分块均完成模型语义抽取；这不代表结论绝对正确。',
      counts, emptyCaveat: '',
    };
  }
  if (empty > 0) {
    return {
      coverage: succeeded > 0 ? 'partial' : 'limited', label: 'AI 语义审查部分降级',
      detail: `模型对 ${empty} 个非空分块返回空结果，无法证明完整覆盖；已保留确定性基线有限预检。`,
      counts, emptyCaveat: '模型返回空结果，无法证明完整覆盖；当前 0 个问题不能证明没有其他问题。',
    };
  }
  if (agentIncomplete && succeeded > 0) {
    return {
      coverage: 'partial', label: 'AI 语义审查部分降级',
      detail: '受限证据修复 Agent 未成功完成或发生弃答；即使分块调用成功，也不能据此标记为完整覆盖。',
      counts, emptyCaveat: 'Agent 处置不完整，当前结果不能证明没有其他问题。',
    };
  }
  if (succeeded > 0) {
    return {
      coverage: 'partial', label: 'AI 语义审查部分降级',
      detail: reasonDetail || '部分正文未完成模型语义抽取，已由确定性基线提供有限兜底。',
      counts, emptyCaveat: '本次模型覆盖不完整，当前结果不能证明没有其他问题。',
    };
  }
  return {
    coverage: 'limited', label: '模型完全降级：仅完成有限预检',
    detail: reasonDetail || '本次没有分块完成模型语义抽取，仅运行确定性基线。', counts,
    emptyCaveat: '本次仅完成有限预检，当前结果不能证明没有其他问题。',
  };
}
