import katex from "katex";
import { useMemo } from "react";

import { useUI } from "../store";

function Eq({ tex }: { tex: string }) {
  const html = useMemo(() => katex.renderToString(tex, { throwOnError: false, displayMode: true }), [tex]);
  return <div className="eq" style={{ margin: "8px 0" }} dangerouslySetInnerHTML={{ __html: html }} />;
}

interface Method {
  group: string;
  en: string;
  es: string;
  bodyEn: string;
  bodyEs: string;
  eq?: string;
  cases: string;
  ref: number; // index into REFS
}

const METHODS: Method[] = [
  {
    group: "adaptive-sampling",
    en: "Adaptive sampling (RAR)",
    es: "Muestreo adaptativo (RAR)",
    bodyEn:
      "A PINN is only as good as where it enforces its PDE. Residual-Adaptive Refinement (RAR / RAR-D / RAD) periodically evaluates the residual on a dense candidate pool and ADDS collocation points where it is largest, so the network concentrates capacity on sharp fronts instead of wasting it on smooth regions. This is what lets a smooth tanh network resolve a shock or a thin interface.",
    bodyEs:
      "Una PINN vale lo que valen los puntos donde impone su EDP. El refinamiento adaptativo por residual (RAR / RAR-D / RAD) evalúa el residual sobre un conjunto candidato denso y AÑADE puntos de colocación donde es mayor, concentrando la capacidad de la red en frentes abruptos en vez de malgastarla en zonas suaves. Es lo que permite a una red tanh suave resolver un choque o una interfaz delgada.",
    eq: String.raw`r_\theta(\mathbf{x})=\mathcal{N}[u_\theta](\mathbf{x}),\quad \text{add } \mathbf{x}^\star=\arg\max_{\mathbf{x}\in\mathcal{P}} \big|r_\theta(\mathbf{x})\big|`,
    cases: "bench-burgers1d (shock), bench-allencahn (interface)",
    ref: 1,
  },
  {
    group: "causal-curriculum",
    en: "Causal & curriculum training",
    es: "Entrenamiento causal y por currículo",
    bodyEn:
      "Standard time-dependent PINNs can converge to a solution that violates causality — fitting late times before the early dynamics are right. Causal training weights the temporal residual so a time slice is only emphasised once all earlier slices are already small, recovering the natural march of information; curriculum/time-marching does the same by solving on a growing time window.",
    bodyEs:
      "Las PINN temporales estándar pueden converger a una solución que viola la causalidad — ajustando tiempos tardíos antes de que la dinámica temprana sea correcta. El entrenamiento causal pondera el residual temporal para que una rebanada de tiempo solo se enfatice cuando las anteriores ya son pequeñas, recuperando la marcha natural de la información; el currículo/time-marching hace lo mismo resolviendo en una ventana temporal creciente.",
    eq: String.raw`w_i=\exp\!\Big(-\varepsilon\sum_{j<i}\mathcal{L}_r(t_j)\Big),\qquad \mathcal{L}=\sum_i w_i\,\mathcal{L}_r(t_i)`,
    cases: "time-dependent cases (heat, wave, Burgers, Allen-Cahn)",
    ref: 2,
  },
  {
    group: "loss-weighting",
    en: "Loss / gradient weighting",
    es: "Pesado de pérdidas / gradientes",
    bodyEn:
      "A PINN minimises a sum of very differently-scaled terms — the PDE residual, each boundary/initial condition, data. If one term dominates the gradient, the others are never satisfied. NTK-, gradient-norm- and Self-Adaptive (SA-PINN) weighting rebalance the terms automatically so no loss starves; PINN-Lab uses fixed per-term weights where the residual scales differ.",
    bodyEs:
      "Una PINN minimiza una suma de términos de escalas muy distintas — el residual de la EDP, cada condición de borde/inicial, los datos. Si un término domina el gradiente, los demás nunca se satisfacen. El pesado por NTK, por norma del gradiente y Self-Adaptive (SA-PINN) rebalancea los términos automáticamente; PINN-Lab usa pesos fijos por término donde las escalas del residual difieren.",
    eq: String.raw`\mathcal{L}=\sum_k \lambda_k\,\mathcal{L}_k,\qquad \lambda_k \;\propto\; \big(\text{tr}\,K_{kk}\big)^{-1}\ \text{(NTK)}`,
    cases: "bench-navier-cavity (PDE vs BC vs pressure gauge)",
    ref: 3,
  },
  {
    group: "architectures",
    en: "Architectures & spectral bias",
    es: "Arquitecturas y sesgo espectral",
    bodyEn:
      "Plain MLPs have a spectral bias: they learn low frequencies fast and high frequencies slowly, so oscillatory solutions stall. Fourier-feature input encodings and SIREN (sinusoidal activations) inject high frequencies directly; hard-constraint output transforms bake the boundary/initial conditions into the network so they are satisfied EXACTLY (and survive the ONNX export).",
    bodyEs:
      "Los MLP simples tienen sesgo espectral: aprenden rápido las bajas frecuencias y lento las altas, así que las soluciones oscilatorias se estancan. Las codificaciones de Fourier y SIREN (activaciones sinusoidales) inyectan altas frecuencias directamente; las restricciones duras por output-transform hornean las condiciones de borde/iniciales en la red para que se cumplan EXACTAMENTE (y sobrevivan a la exportación a ONNX).",
    eq: String.raw`\gamma(\mathbf{x})=\big[\cos(2\pi B\mathbf{x}),\,\sin(2\pi B\mathbf{x})\big];\qquad u_\theta=g(\mathbf{x})+\ell(\mathbf{x})\,\mathcal{N}_\theta(\mathbf{x})`,
    cases: "ind-helmholtz (Fourier), bench-wave1d (SIREN), bench-poisson2d (hard BC)",
    ref: 4,
  },
  {
    group: "domain-decomposition",
    en: "Domain decomposition (FBPINN)",
    es: "Descomposición de dominio (FBPINN)",
    bodyEn:
      "When the solution has a kink or a high-contrast jump, one global network strains across it. Domain-decomposition PINNs (cPINN / XPINN / FBPINN) assign a sub-network to each subdomain and blend them with an overlapping partition of unity, so the kink is produced by two networks handing off rather than one tanh straining — better conditioning and parallelism.",
    bodyEs:
      "Cuando la solución tiene un quiebre o un salto de alto contraste, una red global se tensiona a través de él. Las PINN por descomposición de dominio (cPINN / XPINN / FBPINN) asignan una sub-red a cada subdominio y las mezclan con una partición de la unidad solapada, así el quiebre lo producen dos redes que se entregan el relevo en vez de una tanh forzándose.",
    eq: String.raw`u_\theta(\mathbf{x})=\sum_{j} w_j(\mathbf{x})\,\mathcal{N}_{\theta_j}(\mathbf{x}),\qquad \sum_j w_j\equiv 1`,
    cases: "poll-soil-barrier (low-permeability barrier kink)",
    ref: 5,
  },
  {
    group: "variational-scalable",
    en: "Variational & scalable",
    es: "Variacional y escalable",
    bodyEn:
      "The strong-form residual is not the only option. hp-VPINN minimises the WEAK (variational) residual against test functions, lowering the derivative order and handling lower-regularity solutions; separable PINNs (SPINN) factor the network per axis to scale to >10^7 collocation points. These are the routes to large, stiff, high-dimensional problems.",
    bodyEs:
      "El residual en forma fuerte no es la única opción. hp-VPINN minimiza el residual DÉBIL (variacional) contra funciones de prueba, bajando el orden de derivación y manejando soluciones de menor regularidad; las PINN separables (SPINN) factorizan la red por eje para escalar a >10^7 puntos. Son las rutas a problemas grandes, rígidos y de alta dimensión.",
    eq: String.raw`\int_\Omega \mathcal{N}[u_\theta]\,v\,d\mathbf{x}=0\quad \forall\,v\in V_h \ \text{(weak form)}`,
    cases: "documented (docs/methods/variational-scalable.md)",
    ref: 6,
  },
  {
    group: "optimization",
    en: "Optimization (Adam → L-BFGS)",
    es: "Optimización (Adam → L-BFGS)",
    bodyEn:
      "The house recipe in every PINN-Lab case: first-order Adam drives the loss into the right basin, then quasi-Newton L-BFGS polishes a well-conditioned PINN by another order of magnitude. Recent work (SOAP, gradient alignment) pushes this further; the two-stage schedule is the dependable baseline that makes the relative-L2 numbers reproducible.",
    bodyEs:
      "La receta de la casa en cada caso de PINN-Lab: Adam (primer orden) lleva la pérdida a la cuenca correcta y luego L-BFGS (cuasi-Newton) pule una PINN bien condicionada otro orden de magnitud. Trabajos recientes (SOAP, alineación de gradientes) lo empujan más; el esquema de dos etapas es la línea base confiable que hace reproducibles los números de L2 relativo.",
    eq: String.raw`\theta \xleftarrow{\text{Adam}} \text{good basin}\ \xrightarrow{\text{L-BFGS}}\ \theta^\star`,
    cases: "every DeepXDE case",
    ref: 7,
  },
  {
    group: "operator-learning",
    en: "Operator learning (FNO / DeepONet)",
    es: "Aprendizaje de operadores (FNO / DeepONet)",
    bodyEn:
      "A PINN solves ONE boundary-value problem. An operator learns the solution MAP itself — G: a(x) ↦ u(x) — over a whole family of inputs, so one trained network answers any new instance in a single forward pass, no retraining. The Fourier Neural Operator does this by learning a global convolution in Fourier space: FFT, keep low modes, multiply by learnable weights, inverse FFT.",
    bodyEs:
      "Una PINN resuelve UN problema de valores de frontera. Un operador aprende el MAPA solución mismo — G: a(x) ↦ u(x) — sobre toda una familia de entradas, así una red entrenada responde cualquier instancia nueva en una pasada, sin reentrenar. El Fourier Neural Operator lo hace aprendiendo una convolución global en el espacio de Fourier: FFT, retener modos bajos, multiplicar por pesos aprendidos, FFT inversa.",
    eq: String.raw`(\mathcal{K}v)(\mathbf{x})=\mathcal{F}^{-1}\!\big(R_\phi\cdot\mathcal{F}v\big)(\mathbf{x});\qquad \mathcal{G}_\theta:\ a\mapsto u`,
    cases: "bench-darcy-operator (FNO, held-out test L2 5.6%)",
    ref: 8,
  },
  {
    group: "inverse-uq",
    en: "Inverse problems & UQ",
    es: "Problemas inversos y UQ",
    bodyEn:
      "PINNs shine where classical solvers struggle: recovering an unknown field or parameter from sparse, noisy observations by adding a data term to the PDE residual. Uncertainty is then essential — a deep ensemble (a cheap Bayesian approximation) returns a predictive mean AND a calibrated error bar that grows where data is sparse. PINN-Lab does both, including one inverse case on REAL measured data.",
    bodyEs:
      "Las PINN brillan donde los solvers clásicos sufren: recuperar un campo o parámetro desconocido desde observaciones dispersas y ruidosas, sumando un término de datos al residual de la EDP. La incertidumbre es esencial — un deep ensemble (aproximación bayesiana barata) da una media predictiva Y una barra de error calibrada que crece donde faltan datos. PINN-Lab hace ambos, incluido un caso inverso sobre datos REALES medidos.",
    eq: String.raw`\min_{\theta,\,\xi}\ \mathcal{L}_{\text{PDE}}(\theta;\xi)+\mathcal{L}_{\text{data}}(\theta);\qquad \bar u=\tfrac1K\!\sum_k u_{\theta_k},\ \sigma=\mathrm{std}_k\,u_{\theta_k}`,
    cases: "ind-heat2d-inverse, env-soil-heat-real (REAL), poll-source-uq-bpinn (UQ)",
    ref: 9,
  },
];

interface Ref {
  cite: string;
  doi: string;
}
const REFS: Ref[] = [
  { cite: "Wu, Zhu, Tan, Kartha & Lu (2023). A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs. CMAME 403:115671.", doi: "10.1016/j.cma.2022.115671" },
  { cite: "Wang, Sankaran & Perdikaris (2024). Respecting causality for training physics-informed neural networks. CMAME 421:116813.", doi: "10.1016/j.cma.2024.116813" },
  { cite: "Wang, Yu & Perdikaris (2022). When and why PINNs fail to train: a neural tangent kernel perspective. JCP 449:110768.", doi: "10.1016/j.jcp.2021.110768" },
  { cite: "Tancik et al. (2020). Fourier features let networks learn high-frequency functions in low-dimensional domains. NeurIPS 33. / Sitzmann et al. (2020), SIREN, NeurIPS 33.", doi: "10.48550/arXiv.2006.10739" },
  { cite: "Moseley, Markham & Nissen-Meyer (2023). Finite basis PINNs (FBPINNs): scalable domain decomposition. Adv. Comput. Math. 49:62.", doi: "10.1007/s10444-023-10065-9" },
  { cite: "Kharazmi, Zhang & Karniadakis (2021). hp-VPINNs: variational physics-informed neural networks. CMAME 374:113547.", doi: "10.1016/j.cma.2020.113547" },
  { cite: "Lu, Meng, Mao & Karniadakis (2021). DeepXDE: a deep learning library for solving differential equations. SIAM Review 63(1):208-228.", doi: "10.1137/19M1274067" },
  { cite: "Li et al. (2021). Fourier Neural Operator for parametric partial differential equations. ICLR 2021. / Lu et al. (2021), DeepONet, Nat. Mach. Intell. 3:218-229.", doi: "10.48550/arXiv.2010.08895" },
  { cite: "Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks. JCP 378:686-707. / Yang, Meng & Karniadakis (2021), B-PINNs, JCP 425:109913. / Lakshminarayanan et al. (2017), deep ensembles, NeurIPS 30.", doi: "10.1016/j.jcp.2018.10.045" },
];

export function Methodology() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  return (
    <div className="prose" style={{ maxWidth: 1100 }}>
      <h1>{es ? "Metodología — métodos SOTA" : "Methodology — SOTA methods"}</h1>
      <p className="muted">
        {es
          ? "Cada familia de métodos del estado del arte se EJERCE en al menos un caso de PINN-Lab (no solo se nombra) y se documenta en docs/methods/. Abajo: la idea, su formulación, los casos que la ejercitan y la referencia primaria. La receta base Adam→L-BFGS se usa en todos."
          : "Each state-of-the-art method family is EXERCISED in at least one PINN-Lab case (not merely named) and documented in docs/methods/. Below: the idea, its formulation, the cases that exercise it, and the primary reference. The Adam→L-BFGS base recipe is used everywhere."}
      </p>

      {METHODS.map((m) => (
        <section key={m.group} className="panel" style={{ marginBottom: 14 }}>
          <h3 style={{ color: "var(--accent)", marginTop: 0 }}>{es ? m.es : m.en}</h3>
          <p style={{ fontSize: 14 }}>{es ? m.bodyEs : m.bodyEn}</p>
          {m.eq && <Eq tex={m.eq} />}
          <div className="row" style={{ gap: 16, marginTop: 6 }}>
            <span style={{ fontSize: 13 }}>
              <strong className="muted">{es ? "Casos: " : "Cases: "}</strong>
              <span className="mono">{m.cases}</span>
            </span>
            <code className="muted" style={{ fontSize: 12 }}>docs/methods/{m.group}.md</code>
          </div>
          <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>
            {REFS[m.ref - 1]?.cite}{" "}
            <a href={`https://doi.org/${REFS[m.ref - 1]?.doi}`} target="_blank" rel="noreferrer noopener">
              doi:{REFS[m.ref - 1]?.doi}
            </a>
          </p>
        </section>
      ))}

      <h2>{es ? "Referencias" : "References"}</h2>
      <ol style={{ fontSize: 12.5 }}>
        {REFS.map((r, i) => (
          <li key={i} style={{ margin: "5px 0" }}>
            {r.cite}{" "}
            <a href={`https://doi.org/${r.doi}`} target="_blank" rel="noreferrer noopener">
              doi:{r.doi}
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}
