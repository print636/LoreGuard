/** Scroll the reading surface, without moving its desktop shell under the header. */
export function revealWorkbenchTarget(target: HTMLElement | null, block: "start" | "nearest" = "start") {
  if (!target) return;
  const canvas = target.closest<HTMLElement>(".mainCanvas");
  const rect = target.getBoundingClientRect();
  const messageHeight = canvas?.querySelector<HTMLElement>(".storyOperationMessage")?.getBoundingClientRect().height || 0;
  const insideCanvas = canvas && /^(auto|scroll)$/.test(getComputedStyle(canvas).overflowY);
  const top = (insideCanvas ? canvas.getBoundingClientRect().top
    : document.querySelector<HTMLElement>(".storyTopBar")?.getBoundingClientRect().height || 0) + messageHeight + 12;
  const bottom = insideCanvas ? canvas.getBoundingClientRect().bottom - 12 : window.innerHeight - 12;
  if (block === "nearest" && rect.top >= top && rect.bottom <= bottom) return;
  const delta = block === "nearest" && rect.top >= top && rect.height < bottom - top
    ? rect.bottom - bottom : rect.top - top;
  if (insideCanvas) canvas.scrollTo({ top: canvas.scrollTop + delta, behavior: "auto" });
  else window.scrollTo({ top: window.scrollY + delta, behavior: "auto" });
}
