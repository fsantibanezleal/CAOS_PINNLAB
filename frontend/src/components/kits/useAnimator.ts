import { useCallback, useEffect, useRef, useState } from "react";

export interface Animator {
  frame: number;
  setFrame: (f: number) => void;
  playing: boolean;
  toggle: () => void;
  setPlaying: (p: boolean) => void;
  speed: number;
  setSpeed: (s: number) => void;
  loop: boolean;
  setLoop: (l: boolean) => void;
  nFrames: number;
}

/** A frame animator driven by requestAnimationFrame (NOT setInterval), advancing on wall-clock time so
 *  `speed` is consistent regardless of render rate. Stops honestly at the last frame when not looping.
 *  Clamps when `nFrames` shrinks (switching to a case with fewer frames). */
export function useAnimator(nFrames: number, opts?: { fps?: number; startPlaying?: boolean }): Animator {
  const baseFps = opts?.fps ?? 12;
  const [frame, setFrameState] = useState(0);
  const [playing, setPlaying] = useState(opts?.startPlaying ?? true);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(true);

  const frameRef = useRef(0);
  const raf = useRef(0);
  const last = useRef(0);
  const acc = useRef(0);

  const setFrame = useCallback((f: number) => {
    const c = Math.max(0, Math.min(nFrames - 1, Math.round(f)));
    frameRef.current = c;
    setFrameState(c);
  }, [nFrames]);

  // clamp when the frame count changes
  useEffect(() => {
    if (frameRef.current > nFrames - 1) setFrame(nFrames - 1);
  }, [nFrames, setFrame]);

  useEffect(() => {
    if (!playing || nFrames <= 1) return;
    const tick = (t: number) => {
      if (!last.current) last.current = t;
      const dt = t - last.current;
      last.current = t;
      acc.current += dt * speed;
      const interval = 1000 / baseFps;
      let advanced = false;
      while (acc.current >= interval) {
        acc.current -= interval;
        let nf = frameRef.current + 1;
        if (nf >= nFrames) {
          if (loop) {
            nf = 0;
          } else {
            frameRef.current = nFrames - 1;
            setFrameState(nFrames - 1);
            setPlaying(false);
            return;
          }
        }
        frameRef.current = nf;
        advanced = true;
      }
      if (advanced) setFrameState(frameRef.current);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf.current);
      last.current = 0;
      acc.current = 0;
    };
  }, [playing, nFrames, speed, loop, baseFps]);

  const toggle = useCallback(() => setPlaying((p) => !p), []);

  return { frame, setFrame, playing, toggle, setPlaying, speed, setSpeed, loop, setLoop, nFrames };
}
