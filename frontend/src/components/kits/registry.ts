import { HeatmapKit } from "./HeatmapKit";
import type { KitComponent } from "./types";

/** ADR-0063: the view-kit registry. `manifest.view_kit` selects which kit renders the Field tab; an
 *  unknown/absent value falls back to HeatmapKit (so pre-kit manifests render exactly as before — zero
 *  regression). New kits are added here as each phase lands; the seam never changes. */
export const KITS: Record<string, KitComponent> = {
  HeatmapKit,
  // Phase 1: TimeEvolutionKit, SpatioTemporalKit
  // Phase 2: TrajectoryAnimationKit
  // Phase 3: VectorFieldKit, UQBandKit, PhasePortraitKit
  // Phase 4: ModeShapeKit, InverseOverlayKit
};

export function resolveKit(viewKit?: string): KitComponent {
  return (viewKit && KITS[viewKit]) || HeatmapKit;
}
