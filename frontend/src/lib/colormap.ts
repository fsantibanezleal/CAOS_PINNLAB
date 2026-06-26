// viridis colormap (10 anchors, linear interp) — the single source shared by every render kit so heatmaps,
// carpets and 2-D animation frames all use the identical perceptually-uniform scale.
const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];

/** viridis(t): t in [0,1] -> [r,g,b]. Clamps out-of-range t. */
export function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Min/max over a 2-D field (NaN-safe: skips non-finite). Returns [lo,hi] with a hi>lo guarantee. */
export function fieldRange(field: number[][]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of field) {
    for (const v of row) {
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}
