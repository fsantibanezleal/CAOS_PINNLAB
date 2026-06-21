import { useEffect, useRef } from "react";

// Compact viridis colormap (10 anchor stops, linearly interpolated).
const VIRIDIS: number[][] = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];
function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Render a 2D field[ix][iy] as a heatmap. The first axis maps to horizontal (x), the second to vertical, with the
 *  larger second-axis value drawn at the top (math convention). Symmetric fields auto-scale to [min,max]. */
export function FieldHeatmap({ field }: { field: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const nx = field.length;
  const ny = field[0]?.length ?? 0;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !nx || !ny) return;
    canvas.width = nx;
    canvas.height = ny;
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of field) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = hi - lo || 1;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(nx, ny);
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const t = (field[ix][iy] - lo) / span;
        const [r, g, b] = viridis(t);
        const idx = ((ny - 1 - iy) * nx + ix) * 4; // flip vertical so larger iy is at the top
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [field, nx, ny]);

  return <canvas ref={ref} className="field" style={{ aspectRatio: `${nx} / ${ny || 1}` }} />;
}
