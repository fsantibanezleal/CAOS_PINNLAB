import type { ReactNode } from "react";

import { Heat1dContext } from "./Heat1dContext";
import { PoissonContext } from "./PoissonContext";
import { Wave1dContext } from "./Wave1dContext";

/** case_id -> deep bilingual Context component. Cases are added here as they are migrated to the workbench
 *  structure (ADR-0016 §9). A missing entry renders a clear "in preparation" note (never a blank panel). */
const CASE_CONTEXT: Record<string, (p: { lang: "en" | "es" }) => ReactNode> = {
  "bench-poisson2d": PoissonContext,
  "bench-heat1d": Heat1dContext,
  "bench-wave1d": Wave1dContext,
};

export function ContextFor(caseId: string, lang: "en" | "es"): ReactNode {
  const C = CASE_CONTEXT[caseId];
  if (C) return <C lang={lang} />;
  return <p className="muted">{lang === "es" ? "Contexto en preparación." : "Context in preparation."}</p>;
}

/** A short, human tab label from the case id (strip the category prefix). */
export function caseLabel(caseId: string): string {
  return caseId.replace(/^(bench|mine|poll|ind|ctrl|env)-/, "");
}
