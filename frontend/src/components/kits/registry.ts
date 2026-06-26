import { HeatmapKit } from "./HeatmapKit";
import { InverseOverlayKit } from "./InverseOverlayKit";
import { SpatioTemporalKit } from "./SpatioTemporalKit";
import { TimeEvolutionKit } from "./TimeEvolutionKit";
import { TrajectoryAnimationKit } from "./TrajectoryAnimationKit";
import { UQBandKit } from "./UQBandKit";
import { VectorFieldKit } from "./VectorFieldKit";
import type { KitComponent } from "./types";

/** ADR-0063: the view-kit registry. `manifest.view_kit` selects which kit renders the Field tab; an
 *  unknown/absent value falls back to HeatmapKit (so pre-kit manifests render exactly as before — zero
 *  regression). New kits are added here as each phase lands; the seam never changes. */
export const KITS: Record<string, KitComponent> = {
  HeatmapKit,
  TimeEvolutionKit,
  SpatioTemporalKit,
  TrajectoryAnimationKit,
  VectorFieldKit,
  UQBandKit,
  InverseOverlayKit,
  // Phase 3: PhasePortraitKit
  // Phase 4: ModeShapeKit
};

export function resolveKit(viewKit?: string): KitComponent {
  return (viewKit && KITS[viewKit]) || HeatmapKit;
}
