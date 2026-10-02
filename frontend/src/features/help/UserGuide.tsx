import { useLayoutEffect, useRef, type KeyboardEvent } from "react";
import { userGuideTopics, type UserGuideTopicId } from "./userGuideContent";
import "./user-guide.css";

type Props = {
  initialTopic: UserGuideTopicId;
  trigger: HTMLButtonElement;
  onClose: () => void;
};

/** Reading help is an overlay: the underlying workspace and its drafts stay mounted. */
export default function UserGuide({ initialTopic, trigger, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement | null>(null);
  const readingRegion = useRef<HTMLDivElement | null>(null);
  const directoryTitle = useRef<HTMLHeadingElement | null>(null);

  function focusWithinReading(target: HTMLElement | null) {
    const region = readingRegion.current;
    if (!target || !region) return;
    const offset = target.getBoundingClientRect().top - region.getBoundingClientRect().top + region.scrollTop;
    region.scrollTo({ top: Math.max(0, offset - 16), behavior: "instant" });
    target.focus({ preventScroll: true });
  }

  function goToTopic(id: UserGuideTopicId) {
    focusWithinReading(dialog.current?.querySelector<HTMLHeadingElement>(`#user-guide-topic-${id}`) ?? null);
  }

  function goToDirectory() {
    readingRegion.current?.scrollTo({ top: 0, behavior: "instant" });
    directoryTitle.current?.focus({ preventScroll: true });
  }

  function closeGuide() {
    dialog.current?.close();
    onClose();
    if (trigger.isConnected) trigger.focus({ preventScroll: true });
  }

  function loopBoundaryFocus(event: KeyboardEvent<HTMLDialogElement>) {
    if (event.key !== "Tab" || event.altKey || event.ctrlKey || event.metaKey) return;
    const element = dialog.current;
    if (!element) return;
    const tabbable = Array.from(element.querySelectorAll<HTMLElement>("button, a[href], input, select, textarea, [tabindex]"))
      .filter((control) => control.tabIndex >= 0
        && !control.matches(":disabled, [aria-disabled='true']")
        && !control.closest("[hidden], [inert]")
        && control.getClientRects().length > 0
        && getComputedStyle(control).visibility !== "hidden");
    const first = tabbable[0];
    const last = tabbable[tabbable.length - 1];
    if (!first || !last) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
    // Keep the browser's natural order from programmatically focused headings.
  }

  useLayoutEffect(() => {
    const element = dialog.current;
    if (element && !element.open) element.showModal();
    goToTopic(initialTopic);
    return () => { if (element?.open) element.close(); };
  }, [initialTopic]);

  return (
    <dialog className="userGuide" ref={dialog} aria-labelledby="user-guide-title" aria-describedby="user-guide-purpose"
      onKeyDown={loopBoundaryFocus}
      onCancel={(event) => { event.preventDefault(); closeGuide(); }}>
      <div className="userGuideHeading">
        <div>
          <h2 id="user-guide-title">使用指南</h2>
          <p id="user-guide-purpose">按任务查阅操作与判断边界。阅读不会运行模型或改动文稿。</p>
        </div>
        <div className="userGuideHeadingActions">
          <button type="button" onClick={goToDirectory}>查看目录</button>
          <button type="button" onClick={closeGuide}>关闭使用指南</button>
        </div>
      </div>
      <div className="userGuideReading" ref={readingRegion} tabIndex={0} role="region" aria-label="使用指南正文">
        <div className="userGuideLayout">
          <nav className="userGuideDirectory" aria-label="使用指南目录">
            <h3 ref={directoryTitle} tabIndex={-1}>查阅目录</h3>
            <ul>
              {userGuideTopics.map((topic) => <li key={topic.id}>
                <button type="button" onClick={() => goToTopic(topic.id)} aria-controls={`user-guide-topic-${topic.id}`}>
                  {topic.title}
                </button>
              </li>)}
            </ul>
          </nav>
          <div className="userGuideTopics">
            {userGuideTopics.map((topic) => <section key={topic.id} className="userGuideTopic" aria-labelledby={`user-guide-topic-${topic.id}`}>
              <h3 id={`user-guide-topic-${topic.id}`} tabIndex={-1}>{topic.title}</h3>
              <p className="userGuideSummary">{topic.summary}</p>
              {topic.blocks.map((block, blockIndex) => <div className="userGuideBlock" key={`${topic.id}-${blockIndex}`}>
                <h4>{block.title}</h4>
                {block.steps && <ol>{block.steps.map((step, index) => <li key={`step-${index}`}>{step}</li>)}</ol>}
                {block.paragraphs?.map((paragraph, index) => <p key={`paragraph-${index}`}>{paragraph}</p>)}
                {block.points && <ul>{block.points.map((point, index) => <li key={`point-${index}`}>{point}</li>)}</ul>}
              </div>)}
              <button className="userGuideReturn" type="button" onClick={goToDirectory}>返回目录</button>
            </section>)}
          </div>
        </div>
      </div>
    </dialog>
  );
}
