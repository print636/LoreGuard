import { appRenderRecoveryCopy } from "./appRenderRecovery.ts";

/**
 * Last-resort recovery after React has cleared an uncaught root render failure.
 * Use native DOM, not root.render(), so a failed React fallback is not retried.
 * All content is fixed; no exception, previous props or private DOM is copied.
 */
export function renderStaticAppRenderFailure(rootElement: HTMLElement): void {
  const doc = rootElement.ownerDocument;
  const main = doc.createElement("main");
  main.className = "productPage appRenderRecovery";
  main.setAttribute("aria-labelledby", "app-render-recovery-heading");
  const panel = doc.createElement("section");
  panel.className = "appRenderRecoveryPanel";
  main.append(panel);

  const appendCopy = (tag: "p" | "h1" | "h2", text: string, className = "", parent: HTMLElement = panel) => {
    const node = doc.createElement(tag);
    node.textContent = text;
    node.className = className;
    parent.append(node);
    return node;
  };

  appendCopy("p", appRenderRecoveryCopy.brand, "appRenderRecoveryBrand");
  const heading = appendCopy("h1", appRenderRecoveryCopy.heading);
  heading.id = "app-render-recovery-heading";
  heading.tabIndex = -1;
  const explanation = appendCopy("p", appRenderRecoveryCopy.explanation, "appRenderRecoveryExplanation");
  explanation.setAttribute("role", "alert");

  const warning = doc.createElement("div");
  warning.className = "appRenderRecoveryWarning";
  panel.append(warning);
  appendCopy("h2", appRenderRecoveryCopy.warningHeading, "", warning);
  appendCopy("p", appRenderRecoveryCopy.draftWarning, "", warning);
  appendCopy("p", appRenderRecoveryCopy.submittedWarning, "", warning);

  const reload = doc.createElement("button");
  reload.type = "button";
  reload.className = "quietPrimary appRenderRecoveryReload";
  reload.textContent = appRenderRecoveryCopy.reload;
  reload.addEventListener("click", () => doc.defaultView?.location.reload());
  panel.append(reload);
  appendCopy("p", appRenderRecoveryCopy.feedback, "appRenderRecoveryFeedback");

  rootElement.replaceChildren(main);
  heading.focus({ preventScroll: true });
}
