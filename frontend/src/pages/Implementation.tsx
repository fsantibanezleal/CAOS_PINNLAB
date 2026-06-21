import { useUI } from "../store";

const STAGES = [
  ["preprocess", "validate config / ingest observations (CONTRACT 1)"],
  ["feature_extraction", "collocation / sampling plan"],
  ["train", "DeepXDE/PhysicsNeMo → trained PINN → ONNX export + parity check"],
  ["infer", "evaluate the field on the eval grid"],
  ["evaluate", "relative-L2 vs analytic/FEM reference (the TEST stage)"],
  ["export", "compact replay artifact + manifest (CONTRACT 2)"],
];

export function Implementation() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  return (
    <div className="prose">
      <h1>{es ? "Implementación — el pipeline" : "Implementation — the pipeline"}</h1>
      <p>
        {es
          ? "El pipeline offline es determinista dado (caso, semilla). Entrena el PINN con el motor SOTA, lo valida, exporta la red entrenada a ONNX, y hornea un artefacto compacto + manifest. El navegador corre ese ONNX (onnxruntime-web) para inferencia en vivo y reproduce el artefacto horneado como respaldo."
          : "The offline pipeline is deterministic given (case, seed). It trains the PINN with the SOTA engine, validates it, exports the trained net to ONNX, and bakes a compact artifact + manifest. The browser runs that ONNX (onnxruntime-web) for live inference and replays the baked artifact as a fallback."}
      </p>
      <h2>{es ? "Etapas" : "Stages"}</h2>
      <ol>
        {STAGES.map(([name, desc]) => (
          <li key={name}>
            <code style={{ color: "var(--accent)" }}>{name}</code> — {desc}
          </li>
        ))}
      </ol>
      <h2>{es ? "El puente entrenamiento → web" : "The train → web bridge"}</h2>
      <p>
        {es
          ? "La red entrenada se exporta a ONNX (opset 18) y se verifica que coincide con model.predict (paridad < 1e-4). El gate mide tamaño-ONNX + tiempo de inferencia ort-web + bytes del artefacto para decidir live vs precompute. Las transformaciones de salida (hard constraints, Fourier features) son operaciones tensoriales puras, así que quedan capturadas en el grafo ONNX."
          : "The trained net is exported to ONNX (opset 18) and verified to match model.predict (parity < 1e-4). The gate measures ONNX size + ort-web inference time + artifact bytes to decide live vs precompute. Output transforms (hard constraints, Fourier features) are pure tensor ops, so they are captured in the ONNX graph."}
      </p>
      <p className="muted">docs/architecture/ · docs/frameworks/ · docs/methods/ · docs/cases/</p>
    </div>
  );
}
