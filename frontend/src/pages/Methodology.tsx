import { useUI } from "../store";

const METHODS: Array<{ group: string; en: string; es: string; items: string }> = [
  { group: "adaptive-sampling", en: "Adaptive sampling", es: "Muestreo adaptativo", items: "RAR / RAR-D / RAR-G / RAD — add collocation where the residual is largest (sharp fronts: Burgers, Allen-Cahn)." },
  { group: "causal-curriculum", en: "Causal & curriculum", es: "Causal y currículo", items: "Causal training + time-marching for stiff/advective time-dependent PDEs." },
  { group: "loss-weighting", en: "Loss / gradient weighting", es: "Pesado de pérdidas/gradientes", items: "NTK, grad-norm, Self-Adaptive (SA-PINN), gPINN — balance residual vs BC/IC terms (cavity NS)." },
  { group: "architectures", en: "Architectures", es: "Arquitecturas", items: "Fourier features, modified-MLP, PirateNets, SIREN, hard constraints — spectral-bias remedies (Helmholtz, wave, Poisson)." },
  { group: "domain-decomposition", en: "Domain decomposition", es: "Descomposición de dominio", items: "cPINN / XPINN / FBPINN — subdomain nets stitched by interface conditions." },
  { group: "variational-scalable", en: "Variational & scalable", es: "Variacional y escalable", items: "hp-VPINN (weak form) + SPINN (separable, >1e7 points)." },
  { group: "optimization", en: "Optimization", es: "Optimización", items: "Adam → L-BFGS two-stage (every case) + SOAP / gradient alignment (2025 SOTA)." },
  { group: "operator-learning", en: "Operator learning", es: "Aprendizaje de operadores", items: "DeepONet, FNO, PINO — learn solution operators for instant many-query inference (Darcy operator)." },
  { group: "inverse-uq", en: "Inverse problems & UQ", es: "Problemas inversos y UQ", items: "dde.Variable inverse + Bayesian PINN (B-PINN) + ensembles — sparse-data / real-data assimilation." },
];

export function Methodology() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  return (
    <div className="prose">
      <h1>{es ? "Metodología — métodos SOTA" : "Methodology — SOTA methods"}</h1>
      <p className="muted">
        {es
          ? "Cada método se ejerce en al menos un caso y se documenta en docs/methods/. La receta base Adam→L-BFGS se usa en todos."
          : "Each method is exercised in at least one case and documented in docs/methods/. The Adam→L-BFGS base recipe is used everywhere."}
      </p>
      <div className="grid-cards">
        {METHODS.map((m) => (
          <div key={m.group} className="panel">
            <h3 style={{ color: "var(--accent)" }}>{es ? m.es : m.en}</h3>
            <p style={{ fontSize: 14 }}>{m.items}</p>
            <code className="muted">docs/methods/{m.group}.md</code>
          </div>
        ))}
      </div>
    </div>
  );
}
