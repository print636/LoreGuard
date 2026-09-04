export type ModelDiagnostics = {
  used?: boolean; partial_fallback?: boolean; mode?: string;
  enabled?: boolean; configured?: boolean; total_chunks?: number;
  attempted_chunks?: number; succeeded_chunks?: number; failed_chunks?: number;
  skipped_chunks?: number; invalid_records?: number; empty_response_chunks?: number;
  reason_codes?: string[];
};

const reasons: Record<string, string> = {
  disabled: '模型已关闭', not_configured: '未配置模型', token_budget: 'Token 预算不足',
  chunk_limit: '分块超出上限', circuit_open: '熔断已开启', provider_error: '模型服务失败',
  document_aborted: '当前文档后续分块停止', schema_validation: '结构校验失败',
  record_validation: '记录校验失败', evidence_range: '证据行号越界',
  empty_evidence: '证据为空', lexical_support: '缺少原文支持',
};

export function describeModelStatus(model?: ModelDiagnostics) {
  const counts = [model?.attempted_chunks, model?.succeeded_chunks, model?.failed_chunks,
    model?.skipped_chunks, model?.invalid_records];
  if (!model || counts.some(value => !Number.isInteger(value) || (value ?? -1) < 0)) {
    return {label: '历史或未完成运行：模型覆盖情况未知',
      detail: '缺少结构化调用记录，不能据此判断模型是否参与。', counts: ''};
  }
  const detail = (model.reason_codes ?? []).map(code => reasons[code] ?? '其他执行限制').join('；');
  return {
    label: model.mode ?? '执行状态尚未汇总',
    detail: detail || (model.enabled ? '调用计数不代表剧情判断准确率。' : '主动使用确定性基线。'),
    counts: `尝试 ${model.attempted_chunks} · 成功 ${model.succeeded_chunks} · 失败 ${model.failed_chunks} · 跳过 ${model.skipped_chunks} · 拒绝记录 ${model.invalid_records}`,
  };
}
