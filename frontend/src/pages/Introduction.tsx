import { Equation, InlineMath } from "../components/Equation";
import { useUI } from "../store";

/** Introduction — "what PINN-Lab is", to ADR-0016 §9 depth: deep prose + the PINN loss in KaTeX + an honest scope. */
export function Introduction() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  return (
    <div className="prose" style={{ maxWidth: 1000 }}>
      <h1>{es ? "Qué es PINN-Lab" : "What PINN-Lab is"}</h1>

      <p>
        {es
          ? "Las Redes Neuronales Informadas por Física (PINNs) incrustan la ecuación diferencial que gobierna un sistema directamente en la función de pérdida de una red, vía diferenciación automática. En vez de ajustar datos, la red minimiza el RESIDUAL de la EDP en puntos de colocación del dominio, más las condiciones de borde/iniciales y los datos que haya:"
          : "Physics-Informed Neural Networks (PINNs) embed a system's governing differential equation directly into a network's loss via automatic differentiation. Instead of fitting data, the network minimises the PDE RESIDUAL at collocation points across the domain, plus the boundary/initial conditions and any data:"}
      </p>
      <Equation tex={String.raw`\mathcal{L}(\theta)=\underbrace{\lambda_r\!\sum_i \big|\mathcal{N}[u_\theta](\mathbf{x}_i)\big|^2}_{\text{PDE residual}}+\underbrace{\lambda_b\!\sum_j \big|\mathcal{B}[u_\theta](\mathbf{x}_j)\big|^2}_{\text{boundary/initial}}+\underbrace{\lambda_d\!\sum_k \big|u_\theta(\mathbf{x}_k)-u_k\big|^2}_{\text{data (optional)}}`} />
      <p>
        {es ? (
          <>
            donde <InlineMath tex={String.raw`\mathcal{N}`} /> es el operador diferencial (p. ej.
            <InlineMath tex={String.raw`\ \mathcal{N}[u]=u_t-\alpha u_{xx}`} />) y <InlineMath tex={String.raw`u_\theta`} />
            es la red. Las derivadas salen <em>exactas</em> por diferenciación automática, no por diferencias finitas —
            por eso el método es <strong>sin malla</strong>.
          </>
        ) : (
          <>
            where <InlineMath tex={String.raw`\mathcal{N}`} /> is the differential operator (e.g.
            <InlineMath tex={String.raw`\ \mathcal{N}[u]=u_t-\alpha u_{xx}`} />) and <InlineMath tex={String.raw`u_\theta`} />
            is the network. The derivatives are <em>exact</em> by automatic differentiation, not finite differences —
            that is what makes the method <strong>mesh-free</strong>.
          </>
        )}
      </p>
      <p>
        {es
          ? "Eso compra tres cosas que la numérica clásica no entrega juntas: solución sin malla de PDEs directas/inversas, descubrimiento de campos a partir de datos escasos/ruidosos (la PDE actúa como prior físico donde no hay sensores), y un único artefacto diferenciable que se exporta y corre en el navegador."
          : "That buys three things classical numerics can't give at once: mesh-free solution of forward/inverse PDEs, field discovery from sparse/noisy data (the PDE is a physics prior wherever there are no sensors), and a single differentiable artifact that exports and runs client-side."}
      </p>

      <h2>{es ? "Qué ES PINN-Lab" : "What PINN-Lab IS"}</h2>
      <ul>
        <li>
          {es
            ? "Un catálogo ejecutable de 19 casos de PDEs en cuatro grupos: benchmarks canónicos → minería/procesamiento mineral → polución/ambiental → fluidos/calor industriales. Cada caso se entrena offline con un motor SOTA, se valida contra una referencia analítica/numérica, se exporta a ONNX y se reproduce/infiere en el navegador."
            : "A runnable catalogue of 19 PDE cases in four groups: canonical benchmarks → mining/mineral-processing → pollution/environmental → industrial fluids/heat. Each case is trained offline by a SOTA engine, validated against an analytic or numerical reference, exported to ONNX, and replayed/inferred in the browser."}
        </li>
        <li>
          {es
            ? "Un instrumento de enseñanza y decisión: cada caso se presenta como un banco de trabajo idéntico — una barra de variantes (regímenes de parámetro) y cuatro sub-tabs (Field / Live / Charts / Context), con sus ecuaciones, el método SOTA que lo resuelve, una visualización que reacciona al cursor y a los controles, y un benchmark honesto contra la referencia."
            : "A teaching-and-decision instrument: each case is presented as an identical workbench — a variant bar (parameter regimes) and four sub-tabs (Field / Live / Charts / Context), with its governing equations, the SOTA method that solves it, a viz that reacts to the cursor and the controls, and an honest benchmark vs. the reference."}
        </li>
        <li>
          {es
            ? "Un catálogo de MÉTODOS: cada familia del estado del arte (hard constraints, muestreo adaptativo RAR, Fourier features/SIREN, descomposición de dominio, aprendizaje de operadores, inverso + UQ) se ejerce de verdad en al menos un caso, no solo se nombra."
            : "A catalogue of METHODS: each state-of-the-art family (hard constraints, RAR adaptive sampling, Fourier features/SIREN, domain decomposition, operator learning, inverse + UQ) is genuinely exercised in at least one case, not merely named."}
        </li>
      </ul>

      <h2>{es ? "Qué NO es" : "What it is NOT"}</h2>
      <ul>
        <li>
          {es
            ? "No reemplaza a FEM/FVM. Para un problema directo bien planteado un buen solver clásico suele ser más rápido y preciso. Las PINNs ganan en problemas inversos, asimilación de datos escasos, surrogates paramétricos (operadores) y dominios de alta dimensión."
            : "Not a replacement for FEM/FVM. For a single well-posed forward problem a good classical solver is usually faster and more accurate. PINNs win on inverse problems, sparse-data assimilation, parametric many-query surrogates (operators), and high-dimensional domains."}
        </li>
        <li>
          {es
            ? "No es un gemelo digital industrial. La mayoría de los casos de minería/polución se validan sobre datos simulados o anclas analíticas (MMS); cada uno lleva una etiqueta honesta sintético / ilustrativo-sintético / validado / real, y la página Benchmark publica el error real sin maquillar."
            : "Not an industrial digital twin. Most mining/pollution cases are validated on simulated data or analytic (MMS) anchors; each carries an honest synthetic / illustrative-synthetic / validated / real label, and the Benchmark page publishes the true, un-dressed-up error."}
        </li>
        <li>
          {es
            ? "No es un solver caja-negra. Todo es reproducible: el pipeline offline es determinista dado (caso, semilla), y el ONNX que corre tu navegador es exactamente la red validada (paridad < 1e-4)."
            : "Not a black-box solver. Everything is reproducible: the offline pipeline is deterministic given (case, seed), and the ONNX your browser runs is exactly the validated network (parity < 1e-4)."}
        </li>
      </ul>

      <p className="muted">
        {es
          ? "Empieza por la pestaña App: elige un caso, recorre sus variantes y pasa al tab Live para mover el parámetro físico y ver la red re-evaluarse en vivo. Methodology detalla los métodos; Implementation, la arquitectura; Benchmark, la honestidad."
          : "Start with the App tab: pick a case, walk its variants, and switch to the Live tab to move the physical parameter and watch the network re-evaluate live. Methodology details the methods; Implementation, the architecture; Benchmark, the honesty."}
      </p>
    </div>
  );
}
