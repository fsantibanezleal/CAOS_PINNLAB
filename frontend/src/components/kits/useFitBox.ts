import { useCallback, useEffect, useState } from "react";

/** Deterministic contain-fit (real-review plan E2): measures a container and returns the largest {w, h}
 *  with the given aspect ratio that fits it. Used to size map boxes EXACTLY (the box = the canvas), so the
 *  %-positioned overlays (crosshair, markers, observation dots) stay truthful: no CSS letterboxing guesswork.
 *
 *  The ref is a CALLBACK ref: the ResizeObserver attaches the moment the measured node MOUNTS, not when the
 *  hook first runs. This matters because several kits render the measured element only AFTER their data loads
 *  async (the element is behind an `if (!data) return`), and for a square field the aspect never changes across
 *  that transition, so a deps-gated effect would never re-attach the observer and the map would collapse. The
 *  callback ref makes mount itself the trigger. */
export function useFitBox<T extends HTMLElement>(
  aspect: number,
  pad = 0,
  cols = 1,
  reservedPerCell = 0,
): { areaRef: (node: T | null) => void; w: number; h: number } {
  const [el, setEl] = useState<T | null>(null);
  const areaRef = useCallback((node: T | null) => setEl(node), []);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
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
  }, [el, aspect, pad, cols, reservedPerCell]);

  return { areaRef, w: size.w, h: size.h };
}
