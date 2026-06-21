// CONTRACT 2 (processing -> web), TypeScript mirror of data-pipeline/pinnlab/core/{manifest,trace}.py.
// The web loads ONLY these committed artifacts; it never recomputes. If the Python schema changes and this file is
// not updated in lockstep, the build/typecheck must fail — that is the point of the mirror (ADR-0057).

export const MANIFEST_SCHEMA = "pinnlab.manifest/v1";
export const TRACE_SCHEMA = "pinnlab.field/v1";
export const INDEX_SCHEMA = "pinnlab.index/v1";

export type Lane = "live" | "precompute";

/** The measured live-vs-precompute gate verdict (core/gate.py). */
export interface GateVerdict {
  lane: Lane;
  onnx_bytes: number;
  infer_ms: number;
  trace_bytes: number;
  reasons: string[];
}

/** Pointer to the exported ONNX network + the ONNX-vs-model parity check. */
export interface OnnxRef {
  path: string; // file name under models/, e.g. "bench-poisson2d.onnx"
  bytes: number;
  parity_max_abs: number;
  opset: number;
  input_dim: number; // number of input coordinates (d)
}

/** The compact committed artifact pointer. */
export interface ArtifactRef {
  path: string; // relative to data/derived/, e.g. "bench-poisson2d/field.json"
  format: "json";
  trace_schema: string;
  bytes: number;
}

export interface EngineRef {
  package: string; // "pinnlab"
  version: string; // X.XX.XXX
  framework: string; // "deepxde" | "physicsnemo" | "neuraloperator"
}

/** One baked case manifest (core/manifest.py::build_case_manifest). */
export interface CaseManifest {
  schema: typeof MANIFEST_SCHEMA;
  case_id: string;
  category: string;
  title: string;
  governing_equations: string; // LaTeX-ish
  method: string; // the SOTA method exercised (dossier §4 key)
  real_or_synthetic: "synthetic" | "synthetic-illustrative" | "validated" | string;
  expected_band: string;
  validation_anchor: string; // "analytic" | "dataset" | "benchmark-ghia" | "fem-ref" | "none"
  inputs: string[]; // e.g. ["x", "y"] or ["x", "t"]
  outputs: string[]; // e.g. ["u"] or ["u", "v", "p"]
  engine: EngineRef;
  seed: number;
  artifact: ArtifactRef;
  onnx: OnnxRef;
  lane: Lane;
  gate: GateVerdict;
  flags: Array<Record<string, unknown>>;
  inverse: Record<string, unknown>;
  metrics: Record<string, number | string>; // e.g. { l2_relative, max_abs_error, onnx_parity_max_abs, ... }
  run_ms: number;
}

/** The compact replay artifact = a decimated solution field (core/trace.py::build_trace). */
export interface FieldTrace {
  schema: typeof TRACE_SCHEMA;
  case_id: string;
  dims: string[]; // axis order of `fields` (e.g. ["x", "y"] or ["x", "t"])
  axes: Record<string, number[]>; // axis name -> decimated 1D coordinates
  fields: Record<string, NestedNumberArray>; // output name -> field on the meshgrid (shape = dims lengths)
  summary: Record<string, number | string>;
  inverse: Record<string, unknown>;
}

/** Recursive nested numeric array (1D/2D/3D field), matching np.ndarray.tolist() of any rank. */
export type NestedNumberArray = number[] | NestedNumberArray[];

/** The flat inventory (core/manifest.py::build_index). */
export interface CaseIndexEntry {
  case_id: string;
  category: string;
  title: string;
  manifest_path: string; // relative to data/derived/, e.g. "manifests/bench-poisson2d.json"
}

export interface CaseIndex {
  schema: typeof INDEX_SCHEMA;
  engine_version: string;
  n_cases: number;
  categories: string[];
  cases: CaseIndexEntry[];
}

/** The four case categories (cases/base.py::CATEGORIES). */
export type Category =
  | "canonical-benchmark"
  | "mining-mineral-processing"
  | "pollution-environmental"
  | "industrial-fluids-heat"
  | "control";

export const CATEGORY_LABELS: Record<string, { en: string; es: string }> = {
  "canonical-benchmark": { en: "Canonical benchmarks", es: "Benchmarks canónicos" },
  "mining-mineral-processing": { en: "Mining & mineral processing", es: "Minería y procesamiento mineral" },
  "pollution-environmental": { en: "Pollution & environmental", es: "Polución y ambiental" },
  "industrial-fluids-heat": { en: "Industrial fluids & heat", es: "Fluidos e ingeniería térmica" },
  control: { en: "Controls", es: "Controles" },
};
