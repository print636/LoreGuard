/** Fixed copy only: never derive recovery UI or diagnostics from an exception. */
export const appRenderRecoveryCopy = Object.freeze({
  brand: "LoreGuard · 星轨书斋",
  heading: "页面显示遇到问题",
  explanation: "当前页面未能正常显示。请手动重新加载页面，再检查项目和任务状态。",
  warningHeading: "重新加载前请注意",
  draftWarning: "尚未保存的正文、备注或资料修改可能已丢失；重新加载不能保证找回这些内容。",
  submittedWarning: "重新加载只刷新页面，不会自动重新提交操作或启动模型分析。若此前已提交保存、导入或分析，请先核对结果，避免重复操作。",
  reload: "重新加载页面",
  feedback: "若问题仍然出现，请记录触发步骤后反馈。不要在反馈中发送 API Key 或私密正文。",
});

/**
 * React 19 otherwise reports caught/recoverable errors with their raw contents.
 * Deliberately do not read, stringify, retain, log or transmit either argument.
 * Uncaught root render failures still get a visible recovery surface.
 */
export function safeReactRenderErrorHandlers(showUncaughtFailure: () => void) {
  return {
    onCaughtError(_error: unknown, _errorInfo: unknown): void {},
    onRecoverableError(_error: unknown, _errorInfo: unknown): void {},
    onUncaughtError(_error: unknown, _errorInfo: unknown): void {
      showUncaughtFailure();
    },
  };
}
