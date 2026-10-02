import React from "react";
import ReactDOM from "react-dom/client";
import RootApp from "./app/RootApp";
import AppErrorBoundary from "./app/AppErrorBoundary";
import { safeReactRenderErrorHandlers } from "./app/appRenderRecovery";
import { renderStaticAppRenderFailure } from "./app/appRenderRecoveryDom";
import "./style.css";
import "./workflow.css";
import "./starrail-study.css";
import "./workspace-shell.css";
import "./revision.css";
import "./product-shell.css";
import "./features/characters/character-workspace.css";
import "./features/workflow/guided-review.css";
import "./app/error-boundary.css";

const rootElement = document.getElementById("root")!;
const root = ReactDOM.createRoot(rootElement, safeReactRenderErrorHandlers(() => {
  // React has cleared its failed tree. Do not recursively render a failed fallback.
  renderStaticAppRenderFailure(rootElement);
}));

root.render(
  <React.StrictMode>
    <AppErrorBoundary>
      <RootApp />
    </AppErrorBoundary>
  </React.StrictMode>,
);
