import type { ReactNode } from "react";

import type { CaseManifest, FieldTrace, VariantEntry } from "../../lib/contract";

/** ADR-0063: a render kit owns the Field-tab VISUALIZATION for one `system_type`. The host
 *  (`CaseExperiment`) supplies the manifest, the active variant + its loaded trace, and the language;
 *  shared chrome (governing equation, expected-band note, the method/engine/L2/parity meta-strip) stays
 *  in the host so every kit looks consistent. A kit handles its own `trace == null` loading state. */
export interface KitProps {
  manifest: CaseManifest;
  trace: FieldTrace | null;
  active: VariantEntry;
  lang: "en" | "es";
}

export type KitComponent = (props: KitProps) => ReactNode;
