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
  // ADR-0063: an orthogonal "what kind of system is this" axis that SELECTS the render kit.
  // `category` stays the physics-domain bucket (Experiments/Benchmark grouping); `system_type`
  // drives `view_kit`. Optional so pre-kit manifests still validate — the app falls back to HeatmapKit.
  system_type?: string; // ode-dynamical | time-evol-1d | time-evol-2d | steady-elliptic | vector-flow | inverse-assim | eigen-modal | uq-bayesian | operator-surrogate
  view_kit?: string; // HeatmapKit | TimeEvolutionKit | SpatioTemporalKit | VectorFieldKit | TrajectoryAnimationKit | PhasePortraitKit | ModeShapeKit | UQBandKit | InverseOverlayKit
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
  // enriched (index/v1): drives the App's grouped nav + the functionality badges on each case card
  system_type?: string;
  view_kit?: string;
  method?: string;
  real_or_synthetic?: string;
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

/** One-line "what this scenario domain is" — the group intro on the App page. */
export const CATEGORY_INTRO: Record<string, { en: string; es: string }> = {
  "canonical-benchmark": {
    en: "The reference PDEs and one chaotic ODE that harden the engine + the train→ONNX→web contract: elliptic, parabolic, hyperbolic, advection, an operator, and a dynamical system.",
    es: "Las PDEs de referencia y una EDO caótica que endurecen el motor + el contrato train→ONNX→web: elíptica, parabólica, hiperbólica, advección, un operador y un sistema dinámico.",
  },
  "mining-mineral-processing": {
    en: "Reduced-model processes from mineral processing — comminution, flotation, thickening, heap leaching — each a standard engineering closure, honestly labelled synthetic-illustrative.",
    es: "Procesos de modelo reducido del procesamiento mineral — conminución, flotación, espesamiento, lixiviación en pilas — cada uno un cierre de ingeniería estándar, etiquetado con honestidad como sintético-ilustrativo.",
  },
  "pollution-environmental": {
    en: "Transport, seepage and uncertainty in environmental settings — an advecting plume, a barrier, unsaturated seepage, a Bayesian source, and the one case trained on REAL data (NOAA soil temperatures).",
    es: "Transporte, filtración e incertidumbre en entornos ambientales — una pluma que se advecta, una barrera, filtración no saturada, una fuente bayesiana, y el único caso entrenado con datos REALES (temperaturas de suelo NOAA).",
  },
  "industrial-fluids-heat": {
    en: "Where PINNs genuinely win: an inverse conductivity recovery from sparse sensors, and a Helmholtz field with Fourier-feature inputs.",
    es: "Donde las PINNs ganan de verdad: una recuperación inversa de conductividad desde sensores dispersos, y un campo de Helmholtz con entradas de Fourier features.",
  },
  control: {
    en: "Degenerate sanity anchors — the engine must not crash on a trivial (zero-source) case.",
    es: "Anclas de sanidad degeneradas — el motor no debe fallar en un caso trivial (fuente cero).",
  },
};

/** Short badge label for the view kit = the FUNCTIONALITY each case exercises (shown on the case card). */
export const VIEW_KIT_LABELS: Record<string, { en: string; es: string }> = {
  HeatmapKit: { en: "Field heatmap", es: "Mapa de campo" },
  TimeEvolutionKit: { en: "Time evolution", es: "Evolución temporal" },
  SpatioTemporalKit: { en: "2-D over time", es: "2-D en el tiempo" },
  TrajectoryAnimationKit: { en: "Trajectory (animated)", es: "Trayectoria (animada)" },
  VectorFieldKit: { en: "Vector flow", es: "Flujo vectorial" },
  UQBandKit: { en: "Uncertainty band", es: "Banda de incertidumbre" },
  InverseOverlayKit: { en: "Inverse overlay", es: "Overlay inverso" },
};

/** Honesty label for the data provenance (shown on the case card). */
export const DATA_LABELS: Record<string, { en: string; es: string }> = {
  synthetic: { en: "synthetic", es: "sintético" },
  "synthetic-illustrative": { en: "synthetic-illustrative", es: "sintético-ilustrativo" },
  "validated-real": { en: "REAL data", es: "datos REALES" },
};
