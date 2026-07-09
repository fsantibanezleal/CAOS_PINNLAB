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
  // DEFAULT PAUSED + NON-LOOPING (no autoplay): nothing computes until the user presses Play, and a Play runs ONCE
  // through and stops: never an unbounded replay that pins a CPU core. The user opts into looping per case.
  const [playing, setPlaying] = useState(opts?.startPlaying ?? false);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(false);

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

  // pressing Play at the end restarts from the first frame (don't silently no-op)
  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (!p && frameRef.current >= nFrames - 1) {
        frameRef.current = 0;
        setFrameState(0);
      }
      return !p;
    });
  }, [nFrames]);

  // HARD SAFETY: stop animating the moment the tab is hidden (no background CPU); the rAF effect tears down.
  useEffect(() => {
    const onVis = () => { if (typeof document !== "undefined" && document.hidden) setPlaying(false); };
    if (typeof document !== "undefined") document.addEventListener("visibilitychange", onVis);
    return () => { if (typeof document !== "undefined") document.removeEventListener("visibilitychange", onVis); };
  }, []);

  return { frame, setFrame, playing, toggle, setPlaying, speed, setSpeed, loop, setLoop, nFrames };
}
