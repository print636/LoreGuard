import React from "react";
import ReactDOM from "react-dom/client";
import RootApp from "./app/RootApp";
import "./style.css";
import "./workflow.css";
import "./starrail-study.css";
import "./workspace-shell.css";
import "./revision.css";
import "./product-shell.css";
import "./features/characters/character-workspace.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RootApp />
  </React.StrictMode>,
);
