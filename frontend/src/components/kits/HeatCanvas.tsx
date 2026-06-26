import { useEffect, useMemo, useRef } from "react";

import { fieldRange, viridis } from "../../lib/colormap";

/** A reusable viridis heatmap canvas. `field[iH][iV]` with iH on the horizontal axis and larger iV drawn
 *  at the TOP (math convention, same as FieldView). Optional horizontal cursor at `vCursor` (0..1 from the
 *  bottom) and click-to-seek along the vertical axis — used to make the x–t carpet a seek-bar. A fixed
 *  `range` keeps the color scale stable across animation frames (no per-frame flicker). */
export function HeatCanvas({
  field,
  range,
  vCursor,
  onSeekV,
  ariaLabel,
}: {
  field: number[][];
  range?: [number, number];
  vCursor?: number;
  onSeekV?: (frac: number) => void;
  ariaLabel?: string;
}) {
  const nH = field.length;
  const nV = field[0]?.length ?? 0;
  const ref = useRef<HTMLCanvasElement>(null);
  const [lo, hi] = useMemo(() => range ?? fieldRange(field), [field, range]);
  const span = hi - lo || 1;

  useEffect(() => {
    const c = ref.current;
    if (!c || !nH || !nV) return;
    c.width = nH;
    c.height = nV;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(nH, nV);
    for (let ih = 0; ih < nH; ih++) {
      for (let iv = 0; iv < nV; iv++) {
        const [r, g, b] = viridis((field[ih][iv] - lo) / span);
        const idx = ((nV - 1 - iv) * nH + ih) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [field, nH, nV, lo, span]);

  function onClick(e: React.MouseEvent) {
    if (!onSeekV) return;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const frac = 1 - (e.clientY - rect.top) / rect.height;
    onSeekV(Math.max(0, Math.min(1, frac)));
  }

  return (
    <div className="heatcanvas" onClick={onClick} role="img" aria-label={ariaLabel}>
      <canvas
        ref={ref}
        className="field-canvas"
        style={{ aspectRatio: `${nH} / ${nV || 1}`, cursor: onSeekV ? "pointer" : "default" }}
      />
      {vCursor != null && <div className="carpet-cursor" style={{ top: `${(1 - vCursor) * 100}%` }} />}
    </div>
  );
}
