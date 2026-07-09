// Loaders for the committed artifacts (CONTRACT 2). All paths are under <BASE>/data, overlaid by copy-data.mjs.
import type { CaseIndex, CaseManifest, FieldTrace } from "./contract";
import { APP_VERSION } from "./version";

const BASE = import.meta.env.BASE_URL || "/";
const DATA = `${BASE.replace(/\/$/, "")}/data`;
// Cache-bust every artifact fetch with the app version: the data files are NOT content-hashed, so a fresh deploy
// would otherwise be served stale from the CDN/browser cache. Stamping the URL forces the current build's data.
const V = `?v=${APP_VERSION}`;

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url + V);
  if (!r.ok) throw new Error(`fetch ${url} -> ${r.status}`);
  return (await r.json()) as T;
}

export function loadIndex(): Promise<CaseIndex> {
  return getJson<CaseIndex>(`${DATA}/derived/manifests/index.json`);
}

export function loadManifest(caseId: string): Promise<CaseManifest> {
  return getJson<CaseManifest>(`${DATA}/derived/manifests/${caseId}.json`);
}

/** Load a case's compact replay field trace. `artifactPath` is the manifest's artifact.path (e.g. "bench-poisson2d/field.json"). */
export function loadTrace(artifactPath: string): Promise<FieldTrace> {
  return getJson<FieldTrace>(`${DATA}/derived/${artifactPath}`);
}

/** URL of a case's exported ONNX, for onnxruntime-web. `onnxName` is the manifest's onnx.path (e.g. "bench-poisson2d.onnx"). */
export function onnxUrl(onnxName: string): string {
  return `${DATA}/models/${onnxName}${V}`;
}

/** Load a case's method-ladder comparison trace (standard / naive / adapted / errors on one grid). */
export function loadComparison(tracePath: string): Promise<import("./contract").CompareTrace> {
  return getJson(`${DATA}/derived/${tracePath}`);
}

/** Load a case's diagnostics (wavenumber sweep, radial spectrum, per-lane L2). */
export function loadDiagnostics(path: string): Promise<import("./contract").Diagnostics> {
  return getJson(`${DATA}/derived/${path}`);
}
