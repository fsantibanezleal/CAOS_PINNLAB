// CONTRACT 2 (processing -> web), TypeScript mirror of data-pipeline/pinnlab/core/{manifest,trace}.py.
// The web loads ONLY these committed artifacts; it never recomputes. A drift here vs the Python schema fails the
// build (ADR-0057). Manifest is `pinnlab.manifest/v2`: a case carries a FAMILY OF VARIANTS (parameter regimes), each
// with its own replay trace + metrics, plus the tunable `param_specs` and ONE shared exported ONNX (parametric cases
// sweep the parameter live via the ONNX in the `Live` tab).

export const MANIFEST_SCHEMA = "pinnlab.manifest/v2";
export const TRACE_SCHEMA = "pinnlab.field/v1";
export const INDEX_SCHEMA = "pinnlab.index/v1";

export type Lane = "live" | "precompute";
export type NestedNumberArray = number[] | NestedNumberArray[];

/** A tunable physical parameter (a network input on a parametric case) — the `Live` slider. */
export interface ParamSpec {
  key: string;
  label_en: string;
  label_es: string;
  default: number;
  min: number;
  max: number;
  step: number;
  unit: string;
}

/** One baked parameter regime: its params, its replay trace, and per-regime metrics. */
export interface VariantEntry {
  id: string;
  label_en: string;
  label_es: string;
  note_en: string;
  note_es: string;
  params: Record<string, number>;
  trace: { path: string; bytes: number };
  metrics: Record<string, number | string>;
}

export interface GateVerdict {
  lane: Lane;
  onnx_bytes: number;
  infer_ms: number;
  trace_bytes: number;
  reasons: string[];
}

export interface OnnxRef {
  path: string; // file name under models/, e.g. "bench-poisson2d.onnx"
  bytes: number;
  parity_max_abs: number;
  opset: number;
  input_dim: number;
}

export interface EngineRef {
  package: string;
  version: string;
  framework: string;
}

/** One baked case manifest (core/manifest.py::build_case_manifest, schema v2). */
export interface CaseManifest {
  schema: string;
  case_id: string;
  category: string;
  title: string;
  governing_equations: string;
  method: string;
  real_or_synthetic: string;
  expected_band: string;
  validation_anchor: string;
  inputs: string[]; // ALL network inputs (incl. parameter axes)
  outputs: string[];
  field_axes: string[]; // the 2-D heatmap axes
  param_specs: ParamSpec[];
  engine: EngineRef;
  seed: number;
  trace_schema: string;
  onnx: OnnxRef;
  lane: Lane;
  gate: GateVerdict;
  flags: Array<Record<string, unknown>>;
  inverse: Record<string, unknown>;
  run_ms: number;
  variants: VariantEntry[];
}

/** The compact replay artifact = a decimated solution field (core/trace.py::build_trace). */
export interface FieldTrace {
  schema?: string;
  case_id: string;
  dims: string[]; // axis order of `fields` (the field/heatmap axes)
  axes: Record<string, number[]>;
  fields: Record<string, NestedNumberArray>; // output name -> field on the meshgrid
  summary?: Record<string, number | string>;
  inverse?: Record<string, unknown>;
}

export interface CaseIndexEntry {
  case_id: string;
  category: string;
  title: string;
  manifest_path: string;
}

export interface CaseIndex {
  schema: string;
  engine_version: string;
  n_cases: number;
  categories: string[];
  cases: CaseIndexEntry[];
}

export const CATEGORY_LABELS: Record<string, { en: string; es: string }> = {
  "canonical-benchmark": { en: "Canonical benchmarks", es: "Benchmarks canónicos" },
  "mining-mineral-processing": { en: "Mining & mineral processing", es: "Minería y procesamiento mineral" },
  "pollution-environmental": { en: "Pollution & environmental", es: "Polución y ambiental" },
  "industrial-fluids-heat": { en: "Industrial fluids & heat", es: "Fluidos e ingeniería térmica" },
  control: { en: "Controls", es: "Controles" },
};
