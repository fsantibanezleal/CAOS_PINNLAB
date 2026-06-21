// The LIVE lane: onnxruntime-web inference of the exported PINN. The browser feeds an [n, d] coordinate batch to the
// baked .onnx and gets the field back — re-evaluable at arbitrary resolution / cursor probes in real time. This is
// the live counterpart of the Python pipeline; the replay lane (committed field traces) is the fallback.
import * as ort from "onnxruntime-web";

// Fetch the wasm runtime from a CDN pinned to the installed version (single-threaded -> no COOP/COEP headers needed,
// which GitHub Pages cannot set).
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.numThreads = 1;

const sessions = new Map<string, Promise<ort.InferenceSession>>();

export function getSession(url: string): Promise<ort.InferenceSession> {
  let s = sessions.get(url);
  if (!s) {
    s = ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    sessions.set(url, s);
  }
  return s;
}

/** Run the trained PINN on an [n, d] coordinate batch -> one Float32Array per model output (u, or u/v/p, ...). */
export async function evalNet(
  url: string,
  coords: Float32Array,
  n: number,
  d: number,
): Promise<Record<string, Float32Array>> {
  const session = await getSession(url);
  const inName = session.inputNames[0];
  const feeds: Record<string, ort.Tensor> = { [inName]: new ort.Tensor("float32", coords, [n, d]) };
  const out = await session.run(feeds);
  const result: Record<string, Float32Array> = {};
  for (const name of session.outputNames) result[name] = out[name].data as Float32Array;
  return result;
}

export interface Axis {
  lo: number;
  hi: number;
  n: number;
}

/** Flattened [n, d] coordinate matrix over a meshgrid, C-order over the named axes (matches the Python
 *  linspace_grid / np.meshgrid(..., indexing="ij") so the field reshapes back the same way). */
export function buildGrid(axes: Axis[]): { coords: Float32Array; shape: number[]; n: number } {
  const shape = axes.map((a) => a.n);
  const d = axes.length;
  const n = shape.reduce((a, b) => a * b, 1);
  const grids = axes.map((a) =>
    Array.from({ length: a.n }, (_, i) => a.lo + ((a.hi - a.lo) * i) / Math.max(1, a.n - 1)),
  );
  const coords = new Float32Array(n * d);
  const idx = new Array(d).fill(0);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < d; j++) coords[k * d + j] = grids[j][idx[j]];
    for (let j = d - 1; j >= 0; j--) {
      if (++idx[j] < shape[j]) break;
      idx[j] = 0;
    }
  }
  return { coords, shape, n };
}
