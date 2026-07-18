import type { CaseManifest, EstimateMarker, VariantEntry } from "../../lib/contract";

/** Resolve the case's estimate markers for the active variant (issue #49 S3). */
export function markersFor(manifest: CaseManifest, active?: VariantEntry | null): EstimateMarker[] {
  const e = manifest.estimate;
  if (!e) return [];
  const byV = active ? e.markers_by_variant?.[active.id] : undefined;
  return [...(e.markers ?? []), ...(byV ?? [])];
}

/** The answer, drawn on the field: an absolutely-positioned overlay for any map container whose axes span
 *  [a0,a1] x [b0,b1] in field_axes units (b up). Put inside a position:relative wrapper covering the map. */
export function MarkerLayer({
  markers,
  a0,
  a1,
  b0,
  b1,
  lang,
}: {
  markers: EstimateMarker[];
  a0: number;
  a1: number;
  b0: number;
  b1: number;
  lang: "en" | "es";
}) {
  if (!markers.length) return null;
  const es = lang === "es";
  return (
    <>
      {markers.map((mk, i) => {
        const left = ((mk.a - a0) / (a1 - a0 || 1)) * 100;
        const top = (1 - (mk.b - b0) / (b1 - b0 || 1)) * 100;
        if (!Number.isFinite(left) || !Number.isFinite(top) || left < -2 || left > 102 || top < -2 || top > 102) return null;
        const label = es ? mk.label_es : mk.label_en;
        // alternate label placement per marker so close-by answers (e.g. recovered vs true center) don't collide
        const below = i % 2 === 1;
        return (
          <div key={i} className={"pl-marker" + (below ? " below" : "")} style={{ left: `${left}%`, top: `${top}%` }} title={`${label} (${mk.a}, ${mk.b})`}>
            <span className="pl-marker-dot" />
            <span className="pl-marker-label">{label}</span>
          </div>
        );
      })}
    </>
  );
}
