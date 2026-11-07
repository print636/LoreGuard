import { useLayoutEffect, useRef } from "react";

/** Keep operation feedback visible even when the optional result rail is absent. */
export default function OperationMessage({ message }: { message: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const node = ref.current;
    const canvas = node?.parentElement;
    const header = document.querySelector<HTMLElement>(".storyTopBar");
    if (!node || !canvas) return;
    const measure = () => {
      canvas.style.setProperty("--story-operation-height", `${node.getBoundingClientRect().height + 12}px`);
      canvas.style.setProperty("--story-header-height", `${header?.getBoundingClientRect().height || 0}px`);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    if (header) observer.observe(header);
    return () => observer.disconnect();
  }, []);
  return <div ref={ref} className="storyOperationMessage" role="status" aria-live="polite" aria-atomic="true">{message}</div>;
}
