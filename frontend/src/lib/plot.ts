/** Shared plot helpers (issue #40): nice tick generation + smart number formatting, so every chart in the app
 *  shows real, readable axes instead of two raw-exponential endpoints. Dataviz rules: recessive grids, plain
 *  decimals wherever they fit, exponentials only when the magnitude demands them. */

/** Smart tick label: plain decimals in the human range, exponential only outside it. 0 stays "0". */
export function fmtTick(v: number): string {
  if (!Number.isFinite(v)) return "-";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 10000 || a < 0.001) {
    return v.toExponential(1).replace("e+0", "e").replace("e-0", "e-").replace("e+", "e");
  }
  // up to 3 significant decimals, trailing zeros trimmed
  const dec = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3;
  return Number(v.toFixed(dec)).toString();
}

/** Value read-out (more precision than a tick, same smartness). */
export function fmtVal(v: number): string {
  if (!Number.isFinite(v)) return "-";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 100000 || a < 0.001) return v.toExponential(3);
  return Number(v.toPrecision(4)).toString();
}

/** Integer-ish labels for iteration counts: 0, 250, 1k, 12k. */
export function fmtCount(v: number): string {
  const r = Math.round(v);
  if (Math.abs(r) >= 1000 && r % 100 === 0) return `${r / 1000}k`;
  return String(r);
}

/** "Nice" linear ticks covering [lo, hi] with ~count steps (1/2/5 ladder). Always includes the ends' neighborhood. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo];
  const span = hi - lo;
  const rawStep = span / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step) * step;
  const out: number[] = [];
  for (let v = start; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out.length >= 2 ? out : [lo, hi];
}
