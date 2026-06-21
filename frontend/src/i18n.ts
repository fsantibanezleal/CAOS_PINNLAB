// Lightweight i18n: chrome strings in EN/ES (long-form page content branches on lang inside the page components).
import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const en = {
  brand: "PINN-Lab",
  tagline: "Physics-Informed Neural Networks — a runnable catalogue",
  nav: {
    app: "App",
    introduction: "Introduction",
    methodology: "Methodology",
    implementation: "Implementation",
    experiments: "Experiments",
    benchmark: "Benchmark",
  },
  app: {
    selectCase: "Case",
    method: "Method",
    engine: "Engine",
    lane: "Lane",
    relL2: "Relative L2",
    onnxParity: "ONNX parity",
    governing: "Governing equations",
    live: "Live (ONNX)",
    replay: "Replay (baked)",
    loading: "Loading…",
    field: "Solution field",
    dataLabel: "Data",
  },
  honesty: { synthetic: "synthetic", "synthetic-illustrative": "synthetic (illustrative)", validated: "real-data validated", "validated-real": "real data (validated)" },
  arch: {
    title: "How it was built",
    open: "How it was built",
    close: "Close",
    svgFailed: "Diagram failed to load: {{message}}",
    tab: {
      overview: "Overview",
      web: "Web app",
      pipeline: "Offline pipeline",
      bridge: "Train → ONNX → web",
      gate: "Live vs precompute",
      methods: "Methods & honesty",
    },
    overview: {
      text: "PINN-Lab is an offline pipeline plus a static, deterministic-replay web app (archetype ADR-0057). The pipeline IS the product: it trains every physics-informed network offline, validates it, exports it to ONNX and bakes a compact artifact. The web app ships no training stack — it loads the baked artifacts and runs the exported network live in the browser. The map shows the full lifecycle: from a researched problem fiche, through the six-stage pipeline, to the committed artifacts and the deployed static site — what runs offline, what is precomputed, and what runs in the browser.",
    },
    web: {
      text: "The browser app is a static Vite + React + TypeScript SPA (HashRouter, zustand state, i18next EN/ES, KaTeX equations). Selecting a case fetches its manifest + baked field trace and paints a viridis heatmap with an output selector and time slider; for LIVE cases onnxruntime-web evaluates the exported .onnx at any resolution or cursor probe in real time. Nothing is computed on a server — the SPA is served as static files by GitHub Pages, so the web flow is: select → fetch manifest/trace/onnx → render → probe live.",
    },
    pipeline: {
      text: "Offline, the `pinnlab` engine runs six deterministic stages per case — preprocess → feature_extraction → train → infer → evaluate → export — a pure function of (case, seed). Two versioned data contracts bracket it: ingestion (observation tables for inverse cases, validated against schema + an explicit outlier policy) and artifact (the manifest + field trace the web reads). `train` is the heavy SOTA stage (DeepXDE, Adam → L-BFGS, optional RAR); `evaluate` is the leakage-safe test against an analytic / dataset / benchmark anchor.",
    },
    bridge: {
      text: "A trained PINN is heavy to make but tiny to evaluate — a few dense layers. The train stage exports the trained network to ONNX (opset 18, the modern dynamo exporter, self-contained) and verifies ONNX-vs-model parity < 1e-4, proving the browser evaluates exactly the validated network — hard constraints and Fourier features baked into the graph. onnxruntime-web then runs it client-side in well under a millisecond. This is the bridge that lets a heavy offline method become an interactive static page.",
    },
    gate: {
      text: "Each case is classified LIVE or PRECOMPUTE from MEASUREMENTS, never by hand: live iff the ONNX ≤ 4 MB AND one full-grid inference ≤ 120 ms AND the baked trace ≤ 1 MB. A LIVE case ships its .onnx and runs interactively in the browser; a PRECOMPUTE case ships only the replay trace. The verdict and the three measured numbers go into the manifest, and CI re-derives the lane from them so a mislabel cannot ship — the boundary between what is computed live and what is precomputed is honest and auditable.",
    },
    methods: {
      text: "18 cases across canonical benchmarks, mining / mineral-processing, pollution / environmental, industrial fluids & heat, and control — each exercising a real SOTA technique, each honestly labeled. The matrix maps cases to methods; the blocks summarize the method families and the honesty taxonomy.",
      constraints: {
        title: "Hard constraints & architectures",
        text: "Output-transform hard BC/IC (Poisson, heat, wave), Fourier-feature inputs (Helmholtz), SIREN, PFNN — boundary conditions satisfied exactly and captured inside the exported ONNX graph.",
      },
      adaptive: {
        title: "Adaptive sampling & decomposition",
        text: "RAR residual-adaptive resampling for sharp fronts (Burgers shock, Allen–Cahn interface) and FBPINN 2-channel domain decomposition for high-contrast media (soil barrier).",
      },
      inverse: {
        title: "Inverse problems",
        text: "Recover unknown fields/parameters from sparse observations: a 2nd network output for a conductivity field (2D inverse heat) and a trainable diffusivity scalar (real soil-heat) via PointSetBC + the ingestion contract.",
      },
      uq: {
        title: "Uncertainty quantification",
        text: "A Bayesian PINN as a K=5 bagged deep ensemble exported as one [mean, std] ONNX — epistemic uncertainty that grows where data is sparse (2σ calibration 100%).",
      },
      real: {
        title: "Real data",
        text: "env-soil-heat-real recovers soil thermal diffusivity (0.30 mm²/s) from NOAA USCRN temperatures with held-out-depth validation (RMSE 1.0 °C) — the one case trained on a measured dataset.",
      },
      honesty: {
        title: "Honesty taxonomy",
        text: "Every case is flagged synthetic (closed-form / MMS truth), synthetic-illustrative (a faithful reduced model, not fit to data) or validated-real (a real dataset) — never dressed up; CPU-limited accuracies are reported, not hidden.",
      },
    },
  },
};

const es: typeof en = {
  brand: "PINN-Lab",
  tagline: "Redes Neuronales Informadas por Física — un catálogo ejecutable",
  nav: {
    app: "App",
    introduction: "Introducción",
    methodology: "Metodología",
    implementation: "Implementación",
    experiments: "Experimentos",
    benchmark: "Benchmark",
  },
  app: {
    selectCase: "Caso",
    method: "Método",
    engine: "Motor",
    lane: "Lane",
    relL2: "L2 relativo",
    onnxParity: "Paridad ONNX",
    governing: "Ecuaciones de gobierno",
    live: "En vivo (ONNX)",
    replay: "Replay (horneado)",
    loading: "Cargando…",
    field: "Campo solución",
    dataLabel: "Datos",
  },
  honesty: { synthetic: "sintético", "synthetic-illustrative": "sintético (ilustrativo)", validated: "datos reales validados", "validated-real": "datos reales (validado)" },
  arch: {
    title: "Cómo se construyó",
    open: "Cómo se construyó",
    close: "Cerrar",
    svgFailed: "No se pudo cargar el diagrama: {{message}}",
    tab: {
      overview: "Visión general",
      web: "App web",
      pipeline: "Pipeline offline",
      bridge: "Train → ONNX → web",
      gate: "Live vs precómputo",
      methods: "Métodos y honestidad",
    },
    overview: {
      text: "PINN-Lab es un pipeline offline más una app web estática de replay determinista (arquetipo ADR-0057). El pipeline ES el producto: entrena offline cada red informada por física, la valida, la exporta a ONNX y hornea un artefacto compacto. La app web no incluye stack de entrenamiento — carga los artefactos horneados y ejecuta la red exportada en vivo en el navegador. El mapa muestra todo el ciclo de vida: desde la ficha de problema investigada, por el pipeline de seis etapas, hasta los artefactos versionados y el sitio estático desplegado — qué corre offline, qué se precomputa y qué corre en el navegador.",
    },
    web: {
      text: "La app del navegador es una SPA estática Vite + React + TypeScript (HashRouter, estado con zustand, i18next EN/ES, ecuaciones KaTeX). Seleccionar un caso descarga su manifiesto + traza de campo horneada y pinta un heatmap viridis con selector de salida y slider temporal; para los casos LIVE, onnxruntime-web evalúa el .onnx exportado a cualquier resolución o sonda de cursor en tiempo real. Nada se computa en un servidor — la SPA se sirve como archivos estáticos en GitHub Pages, así el flujo web es: seleccionar → descargar manifiesto/traza/onnx → renderizar → sondear en vivo.",
    },
    pipeline: {
      text: "Offline, el motor `pinnlab` corre seis etapas deterministas por caso — preprocess → feature_extraction → train → infer → evaluate → export — función pura de (caso, semilla). Dos contratos de datos versionados lo enmarcan: ingesta (tablas de observaciones para casos inversos, validadas contra esquema + una política explícita de outliers) y artefacto (el manifiesto + la traza de campo que lee la web). `train` es la etapa SOTA pesada (DeepXDE, Adam → L-BFGS, RAR opcional); `evaluate` es el test sin fuga contra un anclaje analítico / dataset / benchmark.",
    },
    bridge: {
      text: "Una PINN entrenada es cara de producir pero diminuta de evaluar — unas pocas capas densas. La etapa train exporta la red entrenada a ONNX (opset 18, exportador dynamo moderno, autocontenido) y verifica la paridad ONNX-vs-modelo < 1e-4, probando que el navegador evalúa exactamente la red validada — restricciones duras y features de Fourier horneadas en el grafo. onnxruntime-web la corre del lado cliente en bastante menos de un milisegundo. Este es el puente que convierte un método offline pesado en una página estática interactiva.",
    },
    gate: {
      text: "Cada caso se clasifica LIVE o PRECOMPUTE a partir de MEDICIONES, nunca a mano: live si y solo si el ONNX ≤ 4 MB Y una inferencia de grilla completa ≤ 120 ms Y la traza horneada ≤ 1 MB. Un caso LIVE envía su .onnx y corre interactivo en el navegador; un caso PRECOMPUTE envía solo la traza de replay. El veredicto y los tres números medidos van al manifiesto, y CI re-deriva el lane desde ellos para que una etiqueta errónea no pueda desplegarse — la frontera entre lo que se computa en vivo y lo precomputado es honesta y auditable.",
    },
    methods: {
      text: "18 casos en benchmarks canónicos, minería / procesamiento de minerales, contaminación / ambiental, fluidos & calor industrial, y control — cada uno ejercitando una técnica SOTA real, cada uno etiquetado con honestidad. La matriz mapea casos a métodos; los bloques resumen las familias de métodos y la taxonomía de honestidad.",
      constraints: {
        title: "Restricciones duras y arquitecturas",
        text: "BC/IC duras por output-transform (Poisson, calor, onda), entradas con features de Fourier (Helmholtz), SIREN, PFNN — condiciones de borde satisfechas exactamente y capturadas dentro del grafo ONNX exportado.",
      },
      adaptive: {
        title: "Muestreo adaptativo y descomposición",
        text: "RAR (residual-adaptive resampling) para frentes abruptos (choque de Burgers, interfaz de Allen–Cahn) y descomposición de dominio FBPINN de 2 canales para medios de alto contraste (barrera de suelo).",
      },
      inverse: {
        title: "Problemas inversos",
        text: "Recuperar campos/parámetros desconocidos desde observaciones dispersas: una 2ª salida de red para un campo de conductividad (calor inverso 2D) y un escalar de difusividad entrenable (soil-heat real) vía PointSetBC + el contrato de ingesta.",
      },
      uq: {
        title: "Cuantificación de incertidumbre",
        text: "Una PINN bayesiana como deep ensemble de K=5 con bagging exportado como un único ONNX [media, std] — incertidumbre epistémica que crece donde faltan datos (calibración 2σ 100%).",
      },
      real: {
        title: "Datos reales",
        text: "env-soil-heat-real recupera la difusividad térmica del suelo (0.30 mm²/s) desde temperaturas reales NOAA USCRN con validación en profundidad held-out (RMSE 1.0 °C) — el único caso entrenado sobre un dataset medido.",
      },
      honesty: {
        title: "Taxonomía de honestidad",
        text: "Cada caso se marca synthetic (verdad cerrada / MMS), synthetic-illustrative (un modelo reducido fiel, no ajustado a datos) o validated-real (un dataset real) — nunca disfrazado; las precisiones limitadas por CPU se reportan, no se ocultan.",
      },
    },
  },
};

i18n.use(initReactI18next).init({
  resources: { en: { t: en }, es: { t: es } },
  lng: localStorage.getItem("pinnlab.lang") || (navigator.language?.startsWith("es") ? "es" : "en"),
  fallbackLng: "en",
  defaultNS: "t",
  interpolation: { escapeValue: false },
});

export default i18n;
