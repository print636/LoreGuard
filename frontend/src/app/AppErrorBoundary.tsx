import { Component, useEffect, useRef, type ReactNode } from "react";
import { appRenderRecoveryCopy } from "./appRenderRecovery";

type BoundaryState = { failed: boolean };

/** No error object is stored in state, rendered, logged or sent elsewhere. */
export default class AppErrorBoundary extends Component<
  { children: ReactNode },
  BoundaryState
> {
  state: BoundaryState = { failed: false };

  static getDerivedStateFromError(): BoundaryState {
    return { failed: true };
  }

  render() {
    return this.state.failed ? <AppRenderFailure /> : this.props.children;
  }
}

/** Also usable independently after an uncaught root render error. */
export function AppRenderFailure() {
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true });
  }, []);

  return (
    <main className="productPage appRenderRecovery" aria-labelledby="app-render-recovery-heading">
      <section className="appRenderRecoveryPanel">
        <p className="appRenderRecoveryBrand">{appRenderRecoveryCopy.brand}</p>
        <svg className="appRenderRecoveryIcon" viewBox="0 0 48 48" fill="none" aria-hidden="true" focusable="false">
          <path d="M24 13c-4-3-10-4-17-3v26c7-1 13 0 17 3 4-3 10-4 17-3V10c-7-1-13 0-17 3Z" />
          <path d="M24 13v26M13 18h5M13 24h5M30 18h5M30 24h5" />
        </svg>
        <h1 id="app-render-recovery-heading" ref={headingRef} tabIndex={-1}>{appRenderRecoveryCopy.heading}</h1>
        <p className="appRenderRecoveryExplanation" role="alert">{appRenderRecoveryCopy.explanation}</p>
        <div className="appRenderRecoveryWarning">
          <h2>{appRenderRecoveryCopy.warningHeading}</h2>
          <p>{appRenderRecoveryCopy.draftWarning}</p>
          <p>{appRenderRecoveryCopy.submittedWarning}</p>
        </div>
        <button type="button" className="quietPrimary appRenderRecoveryReload" onClick={() => window.location.reload()}>
          {appRenderRecoveryCopy.reload}
        </button>
        <p className="appRenderRecoveryFeedback">{appRenderRecoveryCopy.feedback}</p>
      </section>
    </main>
  );
}
