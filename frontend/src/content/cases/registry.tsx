import type { ReactNode } from "react";

import { AllenCahnContext } from "./AllenCahnContext";
import { Burgers1dContext } from "./Burgers1dContext";
import { ComminutionContext } from "./ComminutionContext";
import { FlotationContext } from "./FlotationContext";
import { HeapLeachContext } from "./HeapLeachContext";
import { Heat1dContext } from "./Heat1dContext";
import { Heat2dInverseContext } from "./Heat2dInverseContext";
import { HelmholtzContext } from "./HelmholtzContext";
import { NavierCavityContext } from "./NavierCavityContext";
import { OceanTransportContext } from "./OceanTransportContext";
import { PoissonContext } from "./PoissonContext";
import { SoilBarrierContext } from "./SoilBarrierContext";
import { SoilHeatRealContext } from "./SoilHeatRealContext";
import { SourceUqBpinnContext } from "./SourceUqBpinnContext";
import { TailingsSeepageContext } from "./TailingsSeepageContext";
import { ThickenerContext } from "./ThickenerContext";
import { Wave1dContext } from "./Wave1dContext";
import { ZeroSourceContext } from "./ZeroSourceContext";

/** case_id -> deep bilingual Context component. Cases are added here as they are migrated to the workbench
 *  structure (ADR-0016 §9). A missing entry renders a clear "in preparation" note (never a blank panel). */
const CASE_CONTEXT: Record<string, (p: { lang: "en" | "es" }) => ReactNode> = {
  "bench-poisson2d": PoissonContext,
  "bench-heat1d": Heat1dContext,
  "bench-wave1d": Wave1dContext,
  "bench-burgers1d": Burgers1dContext,
  "bench-allencahn": AllenCahnContext,
  "poll-ocean-transport": OceanTransportContext,
  "mine-flotation-kinetics": FlotationContext,
  "ctrl-zero-source": ZeroSourceContext,
  "poll-tailings-seepage": TailingsSeepageContext,
  "poll-soil-barrier": SoilBarrierContext,
  "ind-heat2d-inverse": Heat2dInverseContext,
  "mine-thickener-settling": ThickenerContext,
  "mine-heap-leach-rt": HeapLeachContext,
  "mine-comminution-pbe": ComminutionContext,
  "ind-helmholtz": HelmholtzContext,
  "bench-navier-cavity": NavierCavityContext,
  "env-soil-heat-real": SoilHeatRealContext,
  "poll-source-uq-bpinn": SourceUqBpinnContext,
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
