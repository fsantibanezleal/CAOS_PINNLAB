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
  honesty: { synthetic: "synthetic", "synthetic-illustrative": "synthetic (illustrative)", validated: "real-data validated" },
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
  honesty: { synthetic: "sintético", "synthetic-illustrative": "sintético (ilustrativo)", validated: "datos reales validados" },
};

i18n.use(initReactI18next).init({
  resources: { en: { t: en }, es: { t: es } },
  lng: localStorage.getItem("pinnlab.lang") || (navigator.language?.startsWith("es") ? "es" : "en"),
  fallbackLng: "en",
  defaultNS: "t",
  interpolation: { escapeValue: false },
});

export default i18n;
