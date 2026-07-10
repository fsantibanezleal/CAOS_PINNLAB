/** Export a visualization to a PNG the user can drop into slides / a paper. Two element kinds:
 *  a <canvas> (the heatmap panels) exports directly; an <svg> (the diagnostics charts) is serialized, drawn onto a
 *  2x canvas over a solid background (so it is legible on white), then downloaded. Self-contained, no dependency. */

function download(dataUrl: string, name: string) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = `${name}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function snapshotCanvas(canvas: HTMLCanvasElement | null, name: string) {
  if (!canvas) return;
  try {
    download(canvas.toDataURL("image/png"), name);
  } catch {
    /* toDataURL can throw on a tainted canvas; our canvases are same-origin so this should not happen */
  }
}

export function snapshotSvg(svg: SVGSVGElement | null, name: string, bg = "#0e1424") {
  if (!svg) return;
  const vb = svg.viewBox?.baseVal;
  const w = (vb && vb.width) || svg.clientWidth || 640;
  const h = (vb && vb.height) || svg.clientHeight || 300;
  const xml = new XMLSerializer().serializeToString(svg);
  const src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(xml)));
  const img = new Image();
  img.onload = () => {
    const scale = 2;
    const c = document.createElement("canvas");
    c.width = w * scale;
    c.height = h * scale;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0, c.width, c.height);
    download(c.toDataURL("image/png"), name);
  };
  img.src = src;
}

/** Snapshot the first <canvas> or <svg> found inside a container element. */
export function snapshotElement(el: HTMLElement | null, name: string) {
  if (!el) return;
  const canvas = el.querySelector("canvas");
  if (canvas) return snapshotCanvas(canvas as HTMLCanvasElement, name);
  const svg = el.querySelector("svg");
  if (svg) return snapshotSvg(svg as SVGSVGElement, name);
}
