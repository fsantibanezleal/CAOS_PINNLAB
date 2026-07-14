import { useEffect, useRef, useState } from "react";

/** Deterministic contain-fit (real-review plan E2): measures a container and returns the largest {w, h}
 *  with the given aspect ratio that fits it. Used to size map boxes EXACTLY (the box = the canvas), so the
 *  %-positioned overlays (crosshair, markers, observation dots) stay truthful: no CSS letterboxing guesswork.
 *  Re-measures on resize. */
export function useFitBox<T extends HTMLElement>(
  aspect: number,
  pad = 0,
  cols = 1,
  reservedPerCell = 0,
): { areaRef: React.RefObject<T | null>; w: number; h: number } {
  const areaRef = useRef<T | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const gap = 10;
      const aw = Math.max(0, (r.width - pad - (cols - 1) * gap) / Math.max(1, cols));
      const ah = Math.max(0, r.height - pad - reservedPerCell);
      if (!aw || !ah || !Number.isFinite(aspect) || aspect <= 0) return;
      let w = aw;
      let h = w / aspect;
      if (h > ah) {
        h = ah;
        w = h * aspect;
      }
      setSize((s) => (Math.abs(s.w - w) > 1 || Math.abs(s.h - h) > 1 ? { w: Math.floor(w), h: Math.floor(h) } : s));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [aspect, pad, cols, reservedPerCell]);

  return { areaRef, w: size.w, h: size.h };
}
