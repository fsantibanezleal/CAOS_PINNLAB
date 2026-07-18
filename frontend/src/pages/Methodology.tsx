import katex from "katex";
import { useMemo } from "react";

import { SubTabs } from "../components/SubTabs";
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
      "Standard time-dependent PINNs can converge to a solution that violates causality: fitting late times before the early dynamics are right. Causal training weights the temporal residual so a time slice is only emphasised once all earlier slices are already small, recovering the natural march of information; curriculum/time-marching does the same by solving on a growing time window.",
    bodyEs:
      "Las PINN temporales estándar pueden converger a una solución que viola la causalidad: ajustando tiempos tardíos antes de que la dinámica temprana sea correcta. El entrenamiento causal pondera el residual temporal para que una rebanada de tiempo solo se enfatice cuando las anteriores ya son pequeñas, recuperando la marcha natural de la información; el currículo/time-marching hace lo mismo resolviendo en una ventana temporal creciente.",
    eq: String.raw`w_i=\exp\!\Big(-\varepsilon\sum_{j<i}\mathcal{L}_r(t_j)\Big),\qquad \mathcal{L}=\sum_i w_i\,\mathcal{L}_r(t_i)`,
    cases: "time-dependent cases (heat, wave, Burgers, Allen-Cahn)",
    ref: 2,
  },
  {
    group: "loss-weighting",
    en: "Loss / gradient weighting",
    es: "Pesado de pérdidas / gradientes",
    bodyEn:
      "A PINN minimises a sum of very differently-scaled terms: the PDE residual, each boundary/initial condition, data. If one term dominates the gradient, the others are never satisfied. NTK-, gradient-norm- and Self-Adaptive (SA-PINN) weighting rebalance the terms automatically so no loss starves; PINN-Lab uses fixed per-term weights where the residual scales differ.",
    bodyEs:
      "Una PINN minimiza una suma de términos de escalas muy distintas: el residual de la EDP, cada condición de borde/inicial, los datos. Si un término domina el gradiente, los demás nunca se satisfacen. El pesado por NTK, por norma del gradiente y Self-Adaptive (SA-PINN) rebalancea los términos automáticamente; PINN-Lab usa pesos fijos por término donde las escalas del residual difieren.",
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
      "Los MLP simples tienen sesgo espectral: aprenden rápido las bajas frecuencias y lento las altas, así que las soluciones oscilatorias se estancan. Las codificaciones de Fourier y SIREN (activaciones sinusoidales) inyectan altas frecuencias directamente; las restricciones duras por output-transform precalculan las condiciones de borde/iniciales en la red para que se cumplan EXACTAMENTE (y sobrevivan a la exportación a ONNX).",
    eq: String.raw`\gamma(\mathbf{x})=\big[\cos(2\pi B\mathbf{x}),\,\sin(2\pi B\mathbf{x})\big];\qquad u_\theta=g(\mathbf{x})+\ell(\mathbf{x})\,\mathcal{N}_\theta(\mathbf{x})`,
    cases: "ind-helmholtz (Fourier), bench-wave1d (SIREN), bench-poisson2d (hard BC)",
    ref: 4,
  },
  {
    group: "domain-decomposition",
    en: "Domain decomposition (FBPINN)",
    es: "Descomposición de dominio (FBPINN)",
    bodyEn:
      "When the solution has a kink or a high-contrast jump, one global network strains across it. Domain-decomposition PINNs (cPINN / XPINN / FBPINN) assign a sub-network to each subdomain and blend them with an overlapping partition of unity, so the kink is produced by two networks handing off rather than one tanh straining: better conditioning and parallelism.",
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
    cases: "documented in the method dossier (roadmap method)",
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
      "A PINN solves ONE boundary-value problem. An operator learns the solution MAP itself: G: a(x) ↦ u(x): over a whole family of inputs, so one trained network answers any new instance in a single forward pass, no retraining. The Fourier Neural Operator does this by learning a global convolution in Fourier space: FFT, keep low modes, multiply by learnable weights, inverse FFT.",
    bodyEs:
      "Una PINN resuelve UN problema de valores de frontera. Un operador aprende el MAPA solución mismo: G: a(x) ↦ u(x): sobre toda una familia de entradas, así una red entrenada responde cualquier instancia nueva en una pasada, sin reentrenar. El Fourier Neural Operator lo hace aprendiendo una convolución global en el espacio de Fourier: FFT, retener modos bajos, multiplicar por pesos aprendidos, FFT inversa.",
    eq: String.raw`(\mathcal{K}v)(\mathbf{x})=\mathcal{F}^{-1}\!\big(R_\phi\cdot\mathcal{F}v\big)(\mathbf{x});\qquad \mathcal{G}_\theta:\ a\mapsto u`,
    cases: "bench-darcy-operator (FNO, held-out test L2 5.6%)",
    ref: 8,
  },
  {
    group: "dynamical-systems",
    en: "Dynamical systems & chaos (the Lyapunov horizon)",
    es: "Sistemas dinámicos y caos (el horizonte de Lyapunov)",
    bodyEn:
      "An ODE case has no spatial field: the network maps time t to the STATE (a double pendulum's angles). The physics loss is the ODE residual at collocation times, with the initial condition enforced softly (a t² hard-constraint kills the gradient near t=0 and fails). For a CHAOTIC system no fixed network can track the true trajectory past a finite horizon: so the honest metric is the leave-time, not a long-term match: PINN-Lab bakes a high-accuracy RK45 anchor beside the PINN and reports where they first separate.",
    bodyEs:
      "Un caso EDO no tiene campo espacial: la red mapea el tiempo t al ESTADO (los ángulos de un péndulo doble). La pérdida física es el residual de la EDO en puntos de colocación, con la condición inicial impuesta de forma blanda (una restricción dura t² anula el gradiente cerca de t=0 y falla). Para un sistema CAÓTICO ninguna red fija puede seguir la trayectoria verdadera más allá de un horizonte finito: así que la métrica honesta es el leave-time, no un calce a largo plazo: PINN-Lab hornea un ancla RK45 de alta precisión junto a la PINN y reporta dónde se separan.",
    eq: String.raw`\mathcal{L}=\tfrac1N\sum_t\big(r_1^2+r_2^2\big),\quad r_i=\ddot{\theta}_i^{\,\theta}-f_i;\qquad t_{\text{leave}}\sim \tfrac{1}{\lambda_{\max}}\ln\tfrac{1}{\varepsilon_0}`,
    cases: "dyn-double-pendulum (chaotic; leave-time 1.99 s, RK45 anchor)",
    ref: 2,
  },
  {
    group: "inverse-uq",
    en: "Hybrid data + physics: inverse problems & UQ (where PINNs win)",
    es: "Híbrido datos + física: problemas inversos y UQ (donde las PINN ganan)",
    bodyEn:
      "PINNs shine where classical solvers struggle: recovering an unknown field or parameter from sparse, noisy observations by adding a data term to the PDE residual. Uncertainty is then essential: a deep ensemble (a cheap Bayesian approximation) returns a predictive mean AND a calibrated error bar that grows where data is sparse. PINN-Lab does both, including one inverse case on REAL measured data.",
    bodyEs:
      "Las PINN brillan donde los solvers clásicos sufren: recuperar un campo o parámetro desconocido desde observaciones dispersas y ruidosas, sumando un término de datos al residual de la EDP. La incertidumbre es esencial: un deep ensemble (aproximación bayesiana barata) da una media predictiva Y una barra de error calibrada que crece donde faltan datos. PINN-Lab hace ambos, incluido un caso inverso sobre datos REALES medidos.",
    eq: String.raw`\min_{\theta,\,\xi}\ \mathcal{L}_{\text{PDE}}(\theta;\xi)+\mathcal{L}_{\text{data}}(\theta);\qquad \bar u=\tfrac1K\!\sum_k u_{\theta_k},\ \sigma=\mathrm{std}_k\,u_{\theta_k}`,
    cases: "ind-heat2d-inverse, ind-hidden-velocity (HFM flagship), env-soil-heat-real (REAL), poll-source-uq-bpinn (UQ)",
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
  { cite: "Anagnostopoulos, Toscano, Stergiopulos & Karniadakis (2024). Residual-based attention in physics-informed neural networks. CMAME 421:116805.", doi: "10.48550/arXiv.2307.00379" },
  { cite: "Wang, Li, Chen & Perdikaris (2024). PirateNets: physics-informed deep learning with residual adaptive networks. JMLR 25.", doi: "10.48550/arXiv.2402.00326" },
  { cite: "Toscano et al. (2024). From PINNs to PIKANs: recent advances in physics-informed machine learning (physics-informed Kolmogorov-Arnold networks).", doi: "10.48550/arXiv.2410.13228" },
  { cite: "Cho, Nam, Yang, Yun, Hong & Park (2023). Separable physics-informed neural networks (SPINN). NeurIPS 36.", doi: "10.48550/arXiv.2306.15969" },
  { cite: "Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021). Characterizing possible failure modes in physics-informed neural networks. NeurIPS 34.", doi: "10.48550/arXiv.2109.01050" },
  { cite: "Raissi, Yazdani & Karniadakis (2020). Hidden fluid mechanics: learning velocity and pressure fields from flow visualizations. Science 367(6481):1026-1030.", doi: "10.1126/science.aaw4741" },
  { cite: "Cai, Mao, Wang, Yin & Karniadakis (2021). Physics-informed neural networks (PINNs) for fluid mechanics: a review. Acta Mech. Sinica 37:1727-1738.", doi: "10.1007/s10409-021-01148-1" },
  { cite: "Grossmann, Komorowska, Latz & Schönlieb (2024). Can physics-informed neural networks beat the finite element method? IMA J. Appl. Math.", doi: "10.48550/arXiv.2302.04107" },
  { cite: "Wang, Wang & Perdikaris (2021). Learning the solution operator of parametric PDEs with physics-informed DeepONets. Science Advances 7(40):eabi8605.", doi: "10.1126/sciadv.abi8605" },
  { cite: "Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang (2021). Physics-informed machine learning. Nature Reviews Physics 3:422-440.", doi: "10.1038/s42254-021-00314-5" },
  { cite: "Tartakovsky, Ortiz Marrero, Perdikaris, Tartakovsky & Barajas-Solano (2020). Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems. Water Resources Research 56:e2019WR026731.", doi: "10.1029/2019WR026731" },
  { cite: "Rasht-Behesht, Huber, Shukla & Karniadakis (2022). Physics-informed neural networks (PINNs) for wave propagation and full waveform inversions. JGR Solid Earth 127:e2021JB023120.", doi: "10.1029/2021JB023120" },
  { cite: "Cuomo, Schiano Di Cola, Giampaolo, Rozza, Raissi & Piccialli (2022). Scientific machine learning through physics-informed neural networks: where we are and what's next. J. Sci. Comput. 92:88.", doi: "10.1007/s10915-022-01939-z" },
  { cite: "Two-stage Bayesian physics-informed neural network (TSBPINN) inversion of groundwater contaminant source parameters (position/intensity) with posterior uncertainty. ASCE J. Water Resources Planning and Management.", doi: "10.1061/JWRMD5.WRENG-7084" },
];

/** PINNs AS ESTIMATORS IN THE WILD (issue #46): the published record that the estimation-instrument framing is
 *  the field's own, transcribed from the adversarially verified research dossier
 *  (wip/web-review/estimation-reframe-plan-2026-07-10.md §5b: 20 claims, 3-0 verification votes each, primary
 *  sources only). Each entry maps the published use to the catalogue case that teaches the same mechanism. */
const ESTIMATORS: { en: string; es: string; caseLink: string; caseLabel: string; ref: number }[] = [
  {
    en: "The foundational PINN paper already defines TWO problem classes, and the second, 'data-driven discovery of partial differential equations', is exactly the estimation use: inferring unknown coefficients and fields from measurement data, with the physics acting as the regularizer that lets sparse, indirect observations determine them.",
    es: "El paper fundacional de las PINN ya define DOS clases de problemas, y la segunda, el 'descubrimiento de ecuaciones a partir de datos', es exactamente el uso de estimación: inferir coeficientes y campos desconocidos desde mediciones, con la física como regularizador que permite que observaciones dispersas e indirectas los determinen.",
    caseLink: "#/?case=ind-heat2d-inverse&view=compare", caseLabel: "ind-heat2d-inverse", ref: 9,
  },
  {
    en: "Hidden Fluid Mechanics (Science 2020) estimates VELOCITY and PRESSURE fields from flow-visualization images (a passive dye), because extracting them from images directly is otherwise not possible: and it does so while being agnostic to the geometry and the boundary/initial conditions that classical CFD inversion would demand. The paper's own words: quantitative information 'for which direct measurements may not be possible'.",
    es: "Hidden Fluid Mechanics (Science 2020) estima campos de VELOCIDAD y PRESIÓN desde imágenes de visualización de flujo (un tinte pasivo), porque extraerlos directamente de las imágenes no es posible de otro modo: y lo hace siendo agnóstico a la geometría y a las condiciones de borde/iniciales que la inversión CFD clásica exigiría. En palabras del propio paper: información cuantitativa 'para la que mediciones directas pueden no ser posibles'.",
    caseLink: "#/?case=poll-ocean-transport&view=field", caseLabel: "poll-ocean-transport", ref: 15,
  },
  {
    en: "The field's authoritative review (Nature Reviews Physics 2021) states as a Key Point that PINNs are 'effective and efficient for ill-posed and inverse problems', and that solving inverse problems with hidden physics classically is 'often prohibitively expensive'; its flagship figure infers the 3D flow over an espresso cup from Tomo-BOS imaging: velocity estimated from an indirectly measured scalar field, on real experimental data.",
    es: "La revisión de referencia del campo (Nature Reviews Physics 2021) establece como Punto Clave que las PINN son 'efectivas y eficientes para problemas mal planteados e inversos', y que resolver inversos con física oculta por vías clásicas es 'a menudo prohibitivamente caro'; su figura insignia infiere el flujo 3D sobre una taza de espresso desde imágenes Tomo-BOS: velocidad estimada desde un campo escalar medido indirectamente, con datos experimentales reales.",
    caseLink: "#/?case=env-soil-heat-real&view=diagnostics", caseLabel: "env-soil-heat-real", ref: 19,
  },
  {
    en: "For subsurface flow, physics-informed networks estimate the spatially distributed hydraulic conductivity from sparse head + conductivity measurements (Darcy constraint), and the UNSATURATED conductivity function from capillary pressure ONLY, assuming zero direct measurements 'because it is difficult to measure unsaturated conductivity in the field': the PINN as proxy instrument for an unmeasurable constitutive law, reported MORE accurate than the classical MAP inversion baseline.",
    es: "En flujo subterráneo, redes informadas por física estiman la conductividad hidráulica distribuida desde mediciones dispersas de carga + conductividad (restricción de Darcy), y la función de conductividad NO SATURADA solo desde presión capilar, asumiendo cero mediciones directas 'porque es difícil medir la conductividad no saturada en terreno': la PINN como instrumento proxy de una ley constitutiva no medible, reportada MÁS precisa que la inversión clásica MAP de referencia.",
    caseLink: "#/?case=poll-tailings-seepage&view=field", caseLabel: "poll-tailings-seepage · bench-darcy-operator", ref: 20,
  },
  {
    en: "The first full-waveform inversion for seismology with PINNs recovers subsurface VELOCITY MODELS from recorded wavefields: meshless, and robust even from deliberately poor starting models that stall classical gradient-based FWI. Its honest limits are stated in the paper and match this catalogue's stance: for the pure forward problem, spectral-element and finite-difference solvers remain more efficient and accurate, and inverted discontinuities come out smoothed.",
    es: "La primera inversión de forma de onda completa para sismología con PINN recupera MODELOS DE VELOCIDAD del subsuelo desde campos de onda registrados: sin malla, y robusta incluso desde modelos iniciales deliberadamente pobres que estancan la FWI clásica por gradiente. Sus límites honestos están en el propio paper y calzan con la postura de este catálogo: para el problema directo puro, los solucionadores espectrales y de diferencias finitas siguen siendo más eficientes y precisos, y las discontinuidades invertidas salen suavizadas.",
    caseLink: "#/?case=ind-helmholtz&view=training", caseLabel: "ind-helmholtz", ref: 21,
  },
  {
    en: "A two-stage Bayesian PINN inverts groundwater contaminant SOURCE parameters (position and intensity) from noisy concentration observations, with posterior distributions carrying the uncertainty: the estimate arrives WITH its error bars, exactly the mechanism the source-uq case teaches.",
    es: "Una PINN bayesiana de dos etapas invierte los parámetros de la FUENTE contaminante de aguas subterráneas (posición e intensidad) desde observaciones ruidosas de concentración, con distribuciones posteriores llevando la incertidumbre: la estimación llega CON sus barras de error, exactamente el mecanismo que enseña el caso source-uq.",
    caseLink: "#/?case=poll-source-uq-bpinn&view=field", caseLabel: "poll-source-uq-bpinn", ref: 23,
  },
  {
    en: "And the workflow argument, from the 2022 survey: the SAME code that solves the forward problem solves the inverse one with minimal modification: no separate adjoint derivation, no FEM-plus-optimization loop: which is why 'characterizing fluid flows from sensor data' is its canonical inverse example.",
    es: "Y el argumento de flujo de trabajo, de la revisión de 2022: el MISMO código que resuelve el problema directo resuelve el inverso con modificación mínima: sin derivar un adjunto aparte, sin bucle FEM+optimización: por eso 'caracterizar flujos desde datos de sensores' es su ejemplo inverso canónico.",
    caseLink: "#/?case=env-soil-heat-real&view=compare", caseLabel: "env-soil-heat-real", ref: 22,
  },
];

/** The classical → SOTA → candidate-novel LADDER per family (transcribed from the verified deep-research report,
 *  wip/web-review/sota-research-2026-06-26.md). Kept honest: the frontier method + a concrete PINN-Lab proposal +
 *  the confirmed limit. `ref` cites the frontier claim. */
const BEYOND: Record<string, { en: string; es: string; ref?: number }> = {
  "adaptive-sampling": {
    en: "Frontier: RAD/RAR-D (residual-based adaptive distribution) beat uniform + RAR-G at equal budget on stiff/steep problems [Wu 2023]. Candidate for PINN-Lab: couple RAD with causal weighting on Allen-Cahn. Honest limit: better sampling alone does not fix loss-conditioning or causality.",
    es: "Frontera: RAD/RAR-D (distribución adaptativa por residual) superan a uniforme + RAR-G a igual presupuesto en problemas rígidos/abruptos [Wu 2023]. Candidato para PINN-Lab: combinar RAD con pesado causal en Allen-Cahn. Límite honesto: mejor muestreo por sí solo no arregla el condicionamiento ni la causalidad.",
    ref: 1,
  },
  "causal-curriculum": {
    en: "Frontier: causal weighting was the FIRST method to make PINNs succeed on chaotic/turbulent systems (Lorenz, Kuramoto-Sivashinsky, 2D NS) [Wang 2024]: but only over a finite PRE-LYAPUNOV window (Lorenz shown t∈[0,20]; KS exceeds 10% error after t≈0.8). PINN-Lab's candidate-novel contribution is exactly this discipline: a Lyapunov-horizon-aware LEAVE-TIME metric (see the double pendulum) so the app never overclaims beyond the predictability window.",
    es: "Frontera: el pesado causal fue el PRIMER método en hacer que las PINN tengan éxito en sistemas caóticos/turbulentos (Lorenz, Kuramoto-Sivashinsky, NS 2D) [Wang 2024]: pero solo en una ventana finita PRE-LYAPUNOV (Lorenz mostrado t∈[0,20]; KS supera 10% de error tras t≈0.8). La contribución candidata-novel de PINN-Lab es exactamente esa disciplina: una métrica LEAVE-TIME consciente del horizonte de Lyapunov (ver el péndulo doble) para no sobre-prometer más allá de la ventana de predecibilidad.",
    ref: 2,
  },
  "loss-weighting": {
    en: "Frontier: Residual-Based Attention (RBA): a gradient-LESS per-point weighting (EMA of the residuals) that improves on adversarial Self-Adaptive PINNs at O(N) cost, no min-max [Anagnostopoulos 2024]. Candidate for PINN-Lab: RBA per-point weights + causal temporal weights together. Honest limit: NTK is a diagnosis, not THE single root cause: an ill-conditioned loss landscape is also implicated [Krishnapriyan 2021]; RBA can saturate.",
    es: "Frontera: Atención Basada en Residual (RBA): pesado por punto SIN gradiente (EMA de los residuales) que mejora a las Self-Adaptive PINN adversariales a costo O(N), sin min-max [Anagnostopoulos 2024]. Candidato para PINN-Lab: pesos RBA por punto + pesos causales temporales juntos. Límite honesto: NTK es un diagnóstico, no LA única causa raíz: también influye un paisaje de pérdida mal condicionado [Krishnapriyan 2021]; RBA puede saturar.",
    ref: 10,
  },
  architectures: {
    en: "Frontier: PirateNets: a gated residual MLP whose skip connections fix the pathological initialization that makes DEEP PINNs train worse than shallow ones, unlocking depth [Wang 2024]. Also physics-informed Kolmogorov-Arnold networks (PIKANs), learnable-activation networks with promising accuracy/interpretability [Toscano 2024]. Candidate for PINN-Lab: PirateNets or cheb-PIKAN as a drop-in for the oscillatory/stiff cases. Honest limit: PIKANs are heavier and not universally better yet.",
    es: "Frontera: PirateNets: un MLP residual con compuertas cuyas conexiones de salto arreglan la inicialización patológica que hace que las PINN PROFUNDAS entrenen peor que las someras, habilitando la profundidad [Wang 2024]. También las redes de Kolmogorov-Arnold físicas (PIKAN), con activaciones aprendibles y precisión/interpretabilidad prometedoras [Toscano 2024]. Candidato para PINN-Lab: PirateNets o cheb-PIKAN como reemplazo directo en los casos oscilatorios/rígidos. Límite honesto: las PIKAN son más pesadas y aún no universalmente mejores.",
    ref: 11,
  },
  "domain-decomposition": {
    en: "Frontier: XPINN generalized space-time decomposition [Jagtap 2020] and FBPINNs, which express the solution in an overlapping partition of unity of sub-networks and scale to large/multi-scale domains [Moseley 2023]. Candidate for PINN-Lab: FBPINN on a larger multi-scale barrier. Honest limit: the overhead pays off only when the domain is large or multi-scale.",
    es: "Frontera: la descomposición espacio-temporal generalizada XPINN [Jagtap 2020] y las FBPINN, que expresan la solución en una partición de la unidad solapada de sub-redes y escalan a dominios grandes/multi-escala [Moseley 2023]. Candidato para PINN-Lab: FBPINN en una barrera multi-escala mayor. Límite honesto: el sobrecosto rinde solo cuando el dominio es grande o multi-escala.",
    ref: 5,
  },
  "variational-scalable": {
    en: "Frontier: Separable PINNs (SPINN) factor the network into per-axis 1D subnets combined by a tensor product, cutting the forward/backward cost enough to reach >10^7 collocation points [Cho 2023]. Candidate for PINN-Lab: SPINN for a high-resolution or 3-D case. Honest limit: the tensor-product structure suits grid-like/separable domains best.",
    es: "Frontera: las PINN Separables (SPINN) factorizan la red en subredes 1D por eje combinadas por un producto tensorial, reduciendo el costo forward/backward lo suficiente para alcanzar >10^7 puntos de colocación [Cho 2023]. Candidato para PINN-Lab: SPINN para un caso de alta resolución o 3-D. Límite honesto: la estructura de producto tensorial encaja mejor en dominios tipo grilla/separables.",
    ref: 13,
  },
  "inverse-uq": {
    en: "This is where PINNs genuinely win: the HYBRID data+physics PINN. Adding a data term is not a hack: it is a fix for an OPTIMIZATION pathology: the PDE-residual gradients dominate (NTK K_rr ≫ K_uu), so a pure-physics net can settle on a low-residual but WRONG solution among the infinitely many satisfying the PDE; the data term supplies the missing gradient that anchors it to the realizable one. The seminal example is Hidden Fluid Mechanics [Raissi 2020], which reconstructs full velocity+pressure fields from sparse noisy visualizations + the Navier-Stokes residual: geometry/BC-agnostic, robust to noise. PINN-Lab's heat2d-inverse (sparse T sensors), soil-heat-real (REAL data) and uq-bpinn are exactly this rung. Honest scope: this is data ASSIMILATION / reconstruction (it needs data; it does not predict from nothing, and extrapolation beyond the data is unreliable): the strong claim that a hybrid PINN beats classical CFD on a general forward solve is NOT supported.",
    es: "Aquí es donde las PINN sí ganan: la PINN HÍBRIDA datos+física. Sumar un término de datos no es un truco: arregla una patología de OPTIMIZACIÓN: los gradientes del residual EDP dominan (NTK K_rr ≫ K_uu), así que una red física-pura puede quedarse en una solución de bajo residual pero INCORRECTA entre las infinitas que satisfacen la EDP; el término de datos aporta el gradiente que la ancla a la realizable. El ejemplo seminal es Hidden Fluid Mechanics [Raissi 2020], que reconstruye campos completos de velocidad+presión desde visualizaciones dispersas y ruidosas + el residual de Navier-Stokes: agnóstico a geometría/BC, robusto al ruido. heat2d-inverse (sensores T dispersos), soil-heat-real (datos REALES) y uq-bpinn son justo este peldaño. Alcance honesto: es ASIMILACIÓN / reconstrucción de datos (necesita datos; no predice desde la nada, y extrapolar más allá de los datos no es confiable): la afirmación fuerte de que una PINN híbrida gana a la CFD clásica en un problema directo general NO está respaldada.",
    ref: 15,
  },
  "operator-learning": {
    en: "Frontier: physics-informed DeepONet learns the solution OPERATOR (a whole family a↦u), and the physics penalty cuts its data demand: a trainable surrogate that answers a new instance in one forward pass [Wang 2021]. Candidate for PINN-Lab: a physics-informed DeepONet over the parametric families. Honest limit: operators are distribution-bounded: accuracy degrades out-of-distribution, so validate OOD against a classical reference (as the Darcy case does).",
    es: "Frontera: el DeepONet físico aprende el OPERADOR solución (toda una familia a↦u), y la penalización física reduce su demanda de datos: un surrogate entrenable que responde una instancia nueva en una pasada [Wang 2021]. Candidato para PINN-Lab: un DeepONet físico sobre las familias paramétricas. Límite honesto: los operadores están acotados por la distribución: la precisión cae fuera de distribución, así que valida OOD contra una referencia clásica (como hace el caso Darcy).",
    ref: 18,
  },
};

/** Short tab label from a long method-family title: cut at ":" or "(" (the elaboration), clamp at a word
 *  boundary. Keeps the SubTabs strip scannable ("Adaptive sampling", "Hybrid data + physics"). */
function mLabel(s: string): string {
  let t = s.split(/[:(]/)[0].trim();
  if (t.length > 26) {
    t = t.slice(0, 26);
    const sp = t.lastIndexOf(" ");
    if (sp > 12) t = t.slice(0, sp);
    t = t.replace(/[\s,;:.&]+$/, "") + "…";
  }
  return t;
}

export function Methodology() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";

  const scope = (
    <section className="panel scope-panel" style={{ marginBottom: 0 }}>
        <h3 style={{ marginTop: 0 }}>{es ? "Alcance honesto: dónde ganan (y no) las PINN": "Honest scope: where PINNs win (and don't)"}</h3>
        <p style={{ fontSize: 13.5 }}>
          {es
            ? "Para un problema DIRECTO bien planteado, un solver clásico FEM/FVM/espectral bien afinado suele ser más rápido y preciso: de hecho, la generación actual de PINN AÚN NO supera al método de elementos finitos [Grossmann 2024]; esto es el consenso, no un fallo. La frontera verificada confirma dos límites duros: las PINN física-pura FALLAN al endurecerse el régimen (error ~100% en convección-dominada [Krishnapriyan 2021]) y en geometrías CFD complejas (el escalón hacia atrás estanca). El arreglo práctico es la PINN HÍBRIDA datos+física (ver 'Problemas inversos y UQ' abajo): un término de datos ancla la optimización mal condicionada a la solución realizable: por eso las PINN ganan en ASIMILACIÓN/inversión (datos escasos/ruidosos, surrogates paramétricos, alta dimensión), no en el problema directo."
           : "For a single well-posed FORWARD solve, a tuned classical FEM/FVM/spectral solver is usually faster and more accurate: in fact the current generation of PINNs has NOT beaten the finite element method [Grossmann 2024]; this is the consensus, not a failure. The verified frontier confirms two hard limits: pure-physics PINNs FAIL as the regime hardens (~100% error on convection-dominated problems [Krishnapriyan 2021]) and on complex CFD geometry (the backward-facing step stalls). The practical fix is the HYBRID data+physics PINN (see 'Inverse problems & UQ' below): a data term anchors the ill-conditioned optimization to the realizable solution: which is why PINNs earn their keep on ASSIMILATION/inverse (sparse/noisy data, parametric surrogates, high-dimensional / mesh-impractical domains), not on the forward solve."}{" "}
          <a href="https://doi.org/10.48550/arXiv.2302.04107" target="_blank" rel="noreferrer noopener">Grossmann 2024</a>{" · "}
          <a href="https://doi.org/10.48550/arXiv.2109.01050" target="_blank" rel="noreferrer noopener">Krishnapriyan 2021</a>{" · "}
          <a href="https://doi.org/10.1126/science.aaw4741" target="_blank" rel="noreferrer noopener">HFM (Raissi 2020)</a>
        </p>
      </section>
  );

  const estimators = (
    <section className="panel" style={{ marginBottom: 0, borderColor: "var(--accent-2)" }}>
        <h3 style={{ marginTop: 0 }}>{es ? "PINNs como estimadores en el mundo real" : "PINNs as estimators in the wild"}</h3>
        <p style={{ fontSize: 13.5 }}>
          {es
            ? "La razón para usar una PINN no es resolver la EDP: es ESTIMAR algo que no se puede medir directamente, usando la EDP como instrumento. Este es el registro publicado (cada afirmación verificada contra su fuente primaria; ver la referencia junto a cada una), y el caso del catálogo que enseña el mismo mecanismo:"
            : "The reason to use a PINN is not to solve the PDE: it is to ESTIMATE something you cannot measure directly, using the PDE as the instrument. This is the published record (every claim verified against its primary source; see the reference beside each), and the catalogue case that teaches the same mechanism:"}
        </p>
        <ul style={{ fontSize: 13, lineHeight: 1.6 }}>
          {ESTIMATORS.map((x, i) => (
            <li key={i} style={{ margin: "8px 0" }}>
              {es ? x.es : x.en}{" "}
              <a href={x.caseLink} className="mono" style={{ fontSize: 12 }}>{x.caseLabel}</a>{" · "}
              <a href={`https://doi.org/${REFS[x.ref - 1]?.doi}`} target="_blank" rel="noreferrer noopener" style={{ fontSize: 12 }}>
                doi:{REFS[x.ref - 1]?.doi}
              </a>
            </li>
          ))}
        </ul>
      </section>
  );

  const methodPanel = (m: Method) => (
        <section className="panel" style={{ marginBottom: 0 }}>
          <h3 style={{ color: "var(--accent)", marginTop: 0 }}>{es ? m.es: m.en}</h3>
          <p style={{ fontSize: 14 }}>{es ? m.bodyEs: m.bodyEn}</p>
          {m.eq && <Eq tex={m.eq} />}
          <div className="row" style={{ gap: 16, marginTop: 6 }}>
            <span style={{ fontSize: 13 }}>
              <strong className="muted">{es ? "Casos: ": "Cases: "}</strong>
              <span className="mono">{m.cases}</span>
            </span>
          </div>
          {BEYOND[m.group] && (
            <div className="beyond-note">
              <strong>{es ? "Frontera SOTA + propuesta": "SOTA frontier + proposal"}:</strong>{" "}
              {es ? BEYOND[m.group]!.es: BEYOND[m.group]!.en}
              {BEYOND[m.group]!.ref != null && (
                <>{" "}
                  <a href={`https://doi.org/${REFS[BEYOND[m.group]!.ref! - 1]?.doi}`} target="_blank" rel="noreferrer noopener">
                    doi:{REFS[BEYOND[m.group]!.ref! - 1]?.doi}
                  </a>
                </>
              )}
            </div>
          )}
          {/* Per-section references (ADR-0016 §7.5): each section shows ONLY its own refs; there is NO
              bottom-of-page bibliography dump (the banned ReferenceList pattern was removed). */}
          <div className="refs">
            <span className="refs-label">{es ? "Referencias de esta sección" : "References for this section"}</span>
            <ul>
              <li>{REFS[m.ref - 1]?.cite}{" "}<a href={`https://doi.org/${REFS[m.ref - 1]?.doi}`} target="_blank" rel="noreferrer noopener">doi:{REFS[m.ref - 1]?.doi}</a></li>
              {BEYOND[m.group]?.ref != null && BEYOND[m.group]!.ref !== m.ref && (
                <li>{REFS[BEYOND[m.group]!.ref! - 1]?.cite}{" "}<a href={`https://doi.org/${REFS[BEYOND[m.group]!.ref! - 1]?.doi}`} target="_blank" rel="noreferrer noopener">doi:{REFS[BEYOND[m.group]!.ref! - 1]?.doi}</a></li>
              )}
            </ul>
          </div>
        </section>
  );

  const tabs = [
    { id: "scope", label: es ? "Alcance honesto" : "Honest scope", content: scope },
    { id: "estimators", label: es ? "Estimadores" : "Estimators", content: estimators },
    ...METHODS.map((m) => ({ id: m.group, label: mLabel(es ? m.es : m.en), content: methodPanel(m) })),
  ];

  return (
    <div className="prose" style={{ maxWidth: 1100 }}>
      <h1>{es ? "Metodología: métodos SOTA": "Methodology: SOTA methods"}</h1>
      <p className="muted">
        {es
          ? "Cada familia de métodos del estado del arte se EJERCE en al menos un caso de PINN-Lab (no solo se nombra). Cada pestaña es una familia: la idea, su formulación, los casos que la ejercitan, la referencia primaria revisada por pares y la frontera SOTA con una propuesta candidata y su límite honesto. La receta base Adam, L-BFGS se usa en todos."
         : "Each state-of-the-art method family is EXERCISED in at least one PINN-Lab case (not merely named). Each tab is a family: the idea, its formulation, the cases that exercise it, the primary peer-reviewed reference, and the SOTA frontier with a candidate-novel proposal and its honest limit. The Adam then L-BFGS base recipe is used everywhere."}
      </p>
      <SubTabs ariaLabel={es ? "Familias de métodos" : "Method families"} tabs={tabs} />
    </div>
  );
}
