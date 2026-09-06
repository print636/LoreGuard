export type ModelDiagnostics = {
  used?: boolean; partial_fallback?: boolean; mode?: string;
  enabled?: boolean; configured?: boolean; total_chunks?: number;
  attempted_chunks?: number; succeeded_chunks?: number; failed_chunks?: number;
  skipped_chunks?: number; invalid_records?: number; empty_response_chunks?: number;
  unresolved_invalid_records?: number | null; recovered_invalid_records?: number | null;
  repair_succeeded?: boolean; repair_failed?: boolean;
  repair_post_invalid?: number; repair_final_path?: string;
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

const reasons: Record<string, string> = {
  disabled: '模型已关闭', not_configured: '未配置模型', token_budget: 'Token 预算不足',
  chunk_limit: '分块超出上限', circuit_open: '熔断已开启', provider_error: '模型服务失败',
  document_aborted: '当前文档后续分块停止', schema_validation: '结构校验失败',
  record_validation: '记录校验失败', evidence_range: '证据行号越界',
  empty_evidence: '证据为空', lexical_support: '缺少原文支持',
  repair_failed: '语义标签修复失败', repair_run_limit: '语义标签修复次数受限',
};

export function describeModelStatus(model?: ModelDiagnostics): ModelStatusView {
  const values = [model?.total_chunks, model?.attempted_chunks, model?.succeeded_chunks,
    model?.failed_chunks, model?.skipped_chunks, model?.invalid_records,
    model?.empty_response_chunks];
  if (!model || values.some(value => !Number.isInteger(value) || (value ?? -1) < 0)) {
    return {
      coverage: 'unknown',
      label: 'AI 语义审查执行状态未知',
      detail: '缺少结构化调用记录，不能据此判断模型是否参与或覆盖了多少正文。',
      counts: '',
      emptyCaveat: '执行覆盖情况未知，当前结果不能证明没有其他问题。',
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
      coverage: 'unknown',
      label: 'AI 语义审查执行状态未知',
      detail: '结构化调用计数彼此矛盾，不能将这次运行标记为完整或降级。',
      counts: '',
      emptyCaveat: '执行记录不完整，当前结果不能证明没有其他问题。',
    };
  }
  const dispositionSupplied = model.unresolved_invalid_records != null ||
    model.recovered_invalid_records != null;
  let unresolved = invalid;
  let recovered = 0;
  if (dispositionSupplied) {
    const invalidDisposition = [model.unresolved_invalid_records, model.recovered_invalid_records,
      model.repair_post_invalid].some(value => !Number.isInteger(value) || (value ?? -1) < 0) ||
      typeof model.repair_failed !== 'boolean';
    if (invalidDisposition) {
      return {
        coverage: 'unknown',
        label: 'AI 语义审查执行状态未知',
        detail: '无效记录的最终处置计数缺失或非法，不能判断模型覆盖是否完整。',
        counts: '',
        emptyCaveat: '执行记录不完整，当前结果不能证明没有其他问题。',
      };
    }
    unresolved = model.unresolved_invalid_records!;
    recovered = model.recovered_invalid_records!;
    if (invalid !== unresolved + recovered) {
      return {
        coverage: 'unknown',
        label: 'AI 语义审查执行状态未知',
        detail: '原始无效、已恢复与最终未解决记录不守恒，不能判断模型覆盖是否完整。',
        counts: '',
        emptyCaveat: '执行记录不完整，当前结果不能证明没有其他问题。',
      };
    }
  }
  const reasonDetail = (model.reason_codes ?? [])
    .map(code => reasons[code] ?? '其他执行限制').join('；');
  const invalidCounts = dispositionSupplied
    ? `原始无效 ${invalid} · 最终未解决 ${unresolved} · 已恢复 ${recovered} · repair ${model.repair_final_path ?? 'not_needed'}`
    : `拒绝记录 ${invalid}`;
  const counts = `计划 ${total} · 尝试 ${attempted} · 成功 ${succeeded} · ` +
    `失败 ${failed} · 跳过 ${skipped} · ${invalidCounts} · 空响应 ${empty}`;
  const repairIncomplete = dispositionSupplied &&
    (model.repair_failed === true || model.repair_post_invalid! > 0);
  if (succeeded > 0 && failed === 0 && skipped === 0 && unresolved === 0 &&
      !repairIncomplete && empty === 0) {
    return {
      coverage: 'full',
      label: 'AI 语义审查完整',
      detail: recovered > 0
        ? `所有计划分块均完成；语义标签 repair pass 已恢复 ${recovered} 条原始无效记录，最终未解决为 0。这不代表结论绝对正确。`
        : reasonDetail || '所有计划分块均完成模型语义抽取；这不代表结论绝对正确。',
      counts,
      emptyCaveat: '',
    };
  }
  if (empty > 0) {
    return {
      coverage: succeeded > 0 ? 'partial' : 'limited',
      label: 'AI 语义审查部分降级',
      detail: `模型对 ${empty} 个非空分块返回空结果，无法证明完整覆盖；已保留确定性基线有限预检。`,
      counts,
      emptyCaveat: '模型返回空结果，无法证明完整覆盖；当前 0 个问题不能证明没有其他问题。',
    };
  }
  if (succeeded > 0) {
    return {
      coverage: 'partial',
      label: 'AI 语义审查部分降级',
      detail: reasonDetail || '部分正文未完成模型语义抽取，已由确定性基线提供有限兜底。',
      counts,
      emptyCaveat: '本次模型覆盖不完整，当前结果不能证明没有其他问题。',
    };
  }
  return {
    coverage: 'limited',
    label: '模型完全降级：仅完成有限预检',
    detail: reasonDetail || '本次没有分块完成模型语义抽取，仅运行确定性基线。',
    counts,
    emptyCaveat: '本次仅完成有限预检，当前结果不能证明没有其他问题。',
  };
}
