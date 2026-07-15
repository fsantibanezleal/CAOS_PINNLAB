import { useCallback, useEffect, useState } from "react";

export interface GridFit {
  cols: number;
  rows: number;
  cellW: number;
  cellH: number;
}

/** Deterministic multi-panel contain-fit. Measures a STABLE container and, for `n` panels of aspect `aspect`
 *  (mapW/mapH) each carrying a `capH`px caption, returns the (cols, rows) arrangement that MAXIMISES the per-map
 *  area, plus the exact map cell size. This is what makes a 4-map inverse view FILL the stage instead of
 *  collapsing to tiny floating canvases: a wide stage picks 1x4 large maps, a narrow Results column picks 2x2.
 *
 *  The ref must sit on an element whose size does NOT depend on the returned size (no ResizeObserver feedback):
 *  put it on the flex/grid PARENT, then size the children from the result. */
export function useGridFit<T extends HTMLElement>(
  n: number,
  aspect: number,
  opts: { gap?: number; capH?: number; pad?: number } = {},
): { areaRef: (node: T | null) => void } & GridFit {
  const { gap = 10, capH = 18, pad = 2 } = opts;
  const [el, setEl] = useState<T | null>(null);
  const areaRef = useCallback((node: T | null) => setEl(node), []);
  const [fit, setFit] = useState<GridFit>({ cols: 1, rows: 1, cellW: 0, cellH: 0 });

  useEffect(() => {
    if (!el || n < 1) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const W = r.width - pad;
      const H = r.height - pad;
      if (W <= 0 || H <= 0 || !Number.isFinite(aspect) || aspect <= 0) return;
      let best: GridFit = { cols: 1, rows: n, cellW: 0, cellH: 0 };
      let bestArea = -1;
      for (let cols = 1; cols <= n; cols++) {
        const rows = Math.ceil(n / cols);
        const availW = (W - (cols - 1) * gap) / cols;
        const availH = (H - (rows - 1) * gap) / rows - capH; // each cell reserves its caption
        if (availW <= 0 || availH <= 0) continue;
        let mapW = availW;
        let mapH = mapW / aspect;
        if (mapH > availH) {
          mapH = availH;
          mapW = mapH * aspect;
        }
        const area = mapW * mapH;
        if (area > bestArea) {
          bestArea = area;
          best = { cols, rows, cellW: Math.floor(mapW), cellH: Math.floor(mapH) };
        }
      }
      setFit((s) =>
        s.cols === best.cols && s.rows === best.rows && Math.abs(s.cellW - best.cellW) < 1 && Math.abs(s.cellH - best.cellH) < 1
          ? s
          : best,
      );
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el, n, aspect, gap, capH, pad]);

  return { areaRef, ...fit };
}
