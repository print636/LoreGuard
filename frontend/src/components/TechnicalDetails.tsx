import type { ReactNode } from "react";
import "./technical-details.css";

/** Keep support coordinates available without making authors read system identifiers. */
export default function TechnicalDetails({ children }: { children: ReactNode }) {
  return <details className="technicalDetails"><summary>排错信息</summary><div>{children}</div></details>;
}
