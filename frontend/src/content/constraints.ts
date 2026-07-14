/** CONSTRAINT ANATOMY (issue #44): what pins each case's solution down. A PDE alone has infinitely many
 *  solutions; these chips name the side conditions that make the problem well-posed (BC/IC), the tunable
 *  parameters, the inverse unknowns, the observed data, and the validation anchor. Sourced from each case
 *  module's definition (cases/<case>.py); shown color-coded in the context strip so the anatomy of
 *  well-posedness is taught, not just rendered inside the equation string. */

export type PinKind = "bc" | "ic" | "param" | "unknown" | "data" | "anchor";

export interface Pin {
  kind: PinKind;
  en: string;
  es: string;
}

export const PIN_LABELS: Record<PinKind, { en: string; es: string }> = {
  bc: { en: "BC", es: "CB" },
  ic: { en: "IC", es: "CI" },
  param: { en: "PARAM", es: "PARÁM" },
  unknown: { en: "UNKNOWN", es: "INCÓGNITA" },
  data: { en: "DATA", es: "DATOS" },
  anchor: { en: "ANCHOR", es: "ANCLA" },
};

export const CONSTRAINTS: Record<string, Pin[]> = {
  "bench-poisson2d": [
    { kind: "bc", en: "u = 0 on the boundary (imposed HARD)", es: "u = 0 en el borde (impuesta DURA)" },
    { kind: "param", en: "source mode k ∈ [1,3] is a network input", es: "modo de fuente k ∈ [1,3] es entrada de la red" },
    { kind: "anchor", en: "closed-form u* (any k)", es: "u* de forma cerrada (todo k)" },
  ],
  "bench-heat1d": [
    { kind: "ic", en: "u(x,0) = sin(πx) (hard)", es: "u(x,0) = sin(πx) (dura)" },
    { kind: "bc", en: "u(0,t) = u(1,t) = 0 (hard)", es: "u(0,t) = u(1,t) = 0 (dura)" },
    { kind: "param", en: "diffusivity α ∈ [0.1,1]", es: "difusividad α ∈ [0.1,1]" },
    { kind: "anchor", en: "analytic decay e^(-απ²t) sin(πx)", es: "decaimiento analítico e^(-απ²t) sin(πx)" },
  ],
  "bench-wave1d": [
    { kind: "ic", en: "u = sin(πx), u_t = 0 at t=0 (hard)", es: "u = sin(πx), u_t = 0 en t=0 (dura)" },
    { kind: "bc", en: "u(0,t) = u(1,t) = 0 (hard)", es: "u(0,t) = u(1,t) = 0 (dura)" },
    { kind: "param", en: "wave speed c ∈ [0.5,2]", es: "velocidad c ∈ [0.5,2]" },
    { kind: "anchor", en: "standing wave sin(πx)cos(cπt)", es: "onda estacionaria sin(πx)cos(cπt)" },
  ],
  "bench-burgers1d": [
    { kind: "ic", en: "traveling front at x₀ = -0.4 (hard)", es: "frente viajero en x₀ = -0.4 (dura)" },
    { kind: "bc", en: "saturated states at x = ±1 (hard)", es: "estados saturados en x = ±1 (dura)" },
    { kind: "param", en: "viscosity ν ∈ [0.02,0.08] sets the front width", es: "viscosidad ν ∈ [0.02,0.08] fija el ancho del frente" },
    { kind: "anchor", en: "Whitham traveling-shock closed form", es: "forma cerrada del choque viajero (Whitham)" },
  ],
  "bench-allencahn": [
    { kind: "ic", en: "u(x,0) = x²cos(πx) (hard ansatz)", es: "u(x,0) = x²cos(πx) (ansatz duro)" },
    { kind: "bc", en: "endpoint-matched at x = ±1 (hard)", es: "coincidencia en x = ±1 (dura)" },
    { kind: "anchor", en: "spectral reference (Raissi npz) + RAR sampling chases the layers", es: "referencia espectral (npz de Raissi) + muestreo RAR sigue las capas" },
  ],
  "bench-navier-cavity": [
    { kind: "bc", en: "no-slip walls + regularized moving lid (soft, 10x weight)", es: "paredes no-slip + tapa móvil regularizada (suave, peso 10x)" },
    { kind: "bc", en: "pressure gauge p(0,0) = 0", es: "gauge de presión p(0,0) = 0" },
    { kind: "anchor", en: "Ghia 1982 Re=100 centerlines", es: "líneas centrales Ghia 1982 Re=100" },
  ],
  "bench-darcy-operator": [
    { kind: "bc", en: "u = 0 on the boundary", es: "u = 0 en el borde" },
    { kind: "data", en: "training PAIRS a(x) → u(x): the operator is learned from data", es: "PARES de entrenamiento a(x) → u(x): el operador se aprende de datos" },
    { kind: "anchor", en: "finite-difference solve per held-out a(x)", es: "solución de diferencias finitas por cada a(x) reservado" },
  ],
  "ctrl-zero-source": [
    { kind: "bc", en: "u = 0 on the boundary (hard)", es: "u = 0 en el borde (dura)" },
    { kind: "param", en: "amplitude a (a=0 is the degenerate control: u ≡ 0)", es: "amplitud a (a=0 es el control degenerado: u ≡ 0)" },
    { kind: "anchor", en: "manufactured u* (any a)", es: "u* manufacturada (todo a)" },
  ],
  "dyn-double-pendulum": [
    { kind: "ic", en: "θᵢ(0) = 120°, θ̇ᵢ(0) = 0 (hard, θ₀ + t²N ansatz)", es: "θᵢ(0) = 120°, θ̇ᵢ(0) = 0 (dura, ansatz θ₀ + t²N)" },
    { kind: "anchor", en: "RK45 integrator rtol=atol=1e-10; leave-time is the honest metric", es: "integrador RK45 rtol=atol=1e-10; leave-time es la métrica honesta" },
  ],
  "ind-helmholtz": [
    { kind: "bc", en: "u = 0 on ∂Ω (soft, 100x weight)", es: "u = 0 en ∂Ω (suave, peso 100x)" },
    { kind: "param", en: "wavenumber fixed at k₀ = 6π (the spectral-bias stress)", es: "número de onda fijo k₀ = 6π (el estrés del sesgo espectral)" },
    { kind: "anchor", en: "analytic MMS + classical FDM standard", es: "MMS analítica + estándar FDM clásico" },
  ],
  "ind-hidden-velocity": [
    { kind: "unknown", en: "the WHOLE velocity field (u,v): never measured, the HFM hidden state", es: "TODO el campo de velocidad (u,v): nunca medido, el estado oculto de HFM" },
    { kind: "data", en: "~640 sparse noisy dye samples are the ONLY observations (no IC/BC on c)", es: "~640 muestras de tinte dispersas y ruidosas son las ÚNICAS observaciones (sin CI/CB de c)" },
    { kind: "param", en: "D = 0.02 known (HFM-style); incompressibility + declared STEADY flow as residuals", es: "D = 0.02 conocida (estilo HFM); incompresibilidad + flujo ESTACIONARIO declarado como residuales" },
    { kind: "anchor", en: "closed-form u* + stability-asserted FD dye truth; swept vs dead-zone error split", es: "u* cerrada + tinte FD con estabilidad verificada; error separado barrida vs zona muerta" },
  ],
  "ind-heat2d-inverse": [
    { kind: "bc", en: "T = 0 on the boundary (hard)", es: "T = 0 en el borde (dura)" },
    { kind: "unknown", en: "k(x,y) is a FIELD unknown: physics alone leaves it underdetermined (356%)", es: "k(x,y) es incógnita de CAMPO: la física sola la deja indeterminada (356%)" },
    { kind: "data", en: "~100 sparse noisy T sensors make it solvable (4%): see the identifiability curve", es: "~100 sensores T dispersos y ruidosos la hacen soluble (4%): ver la curva de identificabilidad" },
    { kind: "anchor", en: "manufactured (T*, k*, q) triple", es: "triple manufacturado (T*, k*, q)" },
  ],
  "env-soil-heat-real": [
    { kind: "bc", en: "REAL 5 cm + 100 cm sensor series are the Dirichlet boundaries", es: "las series REALES de 5 y 100 cm son los bordes Dirichlet" },
    { kind: "unknown", en: "one SCALAR diffusivity α (recovered: 0.30 mm²/s)", es: "UNA difusividad ESCALAR α (recuperada: 0.30 mm²/s)" },
    { kind: "data", en: "held-out 10/20/50 cm sensors validate OUT-OF-SAMPLE (~1 °C)", es: "los sensores reservados de 10/20/50 cm validan FUERA DE MUESTRA (~1 °C)" },
  ],
  "mine-comminution-pbe": [
    { kind: "ic", en: "Gaussian size distribution at s₀", es: "distribución gaussiana de tamaño en s₀" },
    { kind: "param", en: "grind rate g (network input)", es: "tasa de molienda g (entrada de la red)" },
    { kind: "anchor", en: "advected-diffused Gaussian (exact, any g)", es: "gaussiana advectada-difundida (exacta, todo g)" },
  ],
  "mine-flotation-kinetics": [
    { kind: "ic", en: "C(k,0) = 1", es: "C(k,0) = 1" },
    { kind: "param", en: "rate constant k ∈ [0.5,5] (network input)", es: "constante de tasa k ∈ [0.5,5] (entrada de la red)" },
    { kind: "anchor", en: "exact C* = e^(-kt)", es: "exacta C* = e^(-kt)" },
  ],
  "mine-heap-leach-rt": [
    { kind: "bc", en: "MMS-derived boundary/forcing per species", es: "borde/forzamiento derivados por MMS por especie" },
    { kind: "param", en: "time t is the Live scrubber", es: "el tiempo t es el scrubber del Live" },
    { kind: "anchor", en: "manufactured 2-species (cA*, cB*)", es: "manufacturadas 2 especies (cA*, cB*)" },
  ],
  "mine-thickener-settling": [
    { kind: "ic", en: "MMS descending-front profile", es: "perfil MMS de frente descendente" },
    { kind: "param", en: "descent rate R (network input)", es: "tasa de descenso R (entrada de la red)" },
    { kind: "anchor", en: "MMS exact through the genuine nonlinear flux + degenerate D", es: "MMS exacta a través del flujo no lineal genuino + D degenerada" },
  ],
  "poll-ocean-transport": [
    { kind: "ic", en: "Gaussian patch released at (x₀,y₀)", es: "parche gaussiano liberado en (x₀,y₀)" },
    { kind: "param", en: "time t is the Live scrubber; current v = (0.45,0.35)", es: "el tiempo t es el scrubber del Live; corriente v = (0.45,0.35)" },
    { kind: "anchor", en: "2-D Green's function (exact, not manufactured)", es: "función de Green 2-D (exacta, no manufacturada)" },
  ],
  "poll-soil-barrier": [
    { kind: "ic", en: "c(x,0) = 0 (hard)", es: "c(x,0) = 0 (dura)" },
    { kind: "bc", en: "inlet c = 1-e^(-t), outlet c = 0 (hard lift)", es: "entrada c = 1-e^(-t), salida c = 0 (lift duro)" },
    { kind: "param", en: "D(x) jumps 100x inside the barrier (the kink)", es: "D(x) salta 100x dentro de la barrera (el quiebre)" },
    { kind: "anchor", en: "layered series-resistance MMS", es: "MMS de resistencias en serie por capas" },
  ],
  "poll-source-uq-bpinn": [
    { kind: "bc", en: "c = 0 at both walls", es: "c = 0 en ambas paredes" },
    { kind: "data", en: "a handful of sparse noisy sensors: NOT the full IC: hence the σ band", es: "un puñado de sensores dispersos y ruidosos: NO la CI completa: de ahí la banda σ" },
    { kind: "anchor", en: "analytic diffusion mode + 2σ calibration", es: "modo de difusión analítico + calibración 2σ" },
  ],
  "poll-tailings-seepage": [
    { kind: "ic", en: "exact separable profile at t=0", es: "perfil separable exacto en t=0" },
    { kind: "bc", en: "head profile pinned at the column ends", es: "perfil de carga fijado en los extremos de la columna" },
    { kind: "param", en: "Gardner sorptive number α (network input)", es: "número sortivo de Gardner α (entrada de la red)" },
    { kind: "anchor", en: "Kirchhoff-transform exact (α-dependent)", es: "exacta por transformada de Kirchhoff (depende de α)" },
  ],
};
