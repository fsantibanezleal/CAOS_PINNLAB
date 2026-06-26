import type { Animator } from "./useAnimator";

/** Shared play/pause + scrub + speed + loop bar for every animated kit. The scrub pauses on drag so the
 *  user can inspect a frame; the readout shows the animated-axis value + frame index. */
export function Transport({
  anim,
  lang,
  axisLabel,
  axisValue,
}: {
  anim: Animator;
  lang: "en" | "es";
  axisLabel: string;
  axisValue: number;
}) {
  const es = lang === "es";
  return (
    <div className="transport">
      <button type="button" className="t-btn" onClick={anim.toggle} aria-label={anim.playing ? (es ? "Pausar" : "Pause") : (es ? "Reproducir" : "Play")}>
        {anim.playing ? "❚❚" : "▶"}
      </button>
      <input
        className="scrub t-scrub"
        type="range"
        min={0}
        max={Math.max(0, anim.nFrames - 1)}
        step={1}
        value={anim.frame}
        aria-label={es ? "Línea de tiempo" : "Timeline"}
        onChange={(e) => {
          anim.setPlaying(false);
          anim.setFrame(Number(e.target.value));
        }}
      />
      <span className="t-readout mono">{axisLabel}={axisValue.toFixed(3)} · {anim.frame + 1}/{anim.nFrames}</span>
      <label className="t-ctl">
        <span className="muted">{es ? "Vel" : "Speed"}</span>
        <select value={anim.speed} onChange={(e) => anim.setSpeed(Number(e.target.value))}>
          <option value={0.25}>0.25×</option>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
          <option value={4}>4×</option>
        </select>
      </label>
      <label className="t-ctl">
        <input type="checkbox" checked={anim.loop} onChange={(e) => anim.setLoop(e.target.checked)} /> {es ? "Bucle" : "Loop"}
      </label>
    </div>
  );
}
