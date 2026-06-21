import { useUI } from "../store";

export function Introduction() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  return (
    <div className="prose">
      <h1>{es ? "Qué es PINN-Lab" : "What PINN-Lab is"}</h1>
      <p>
        {es
          ? "Las Redes Neuronales Informadas por Física (PINNs) incrustan la ecuación diferencial que gobierna un sistema directamente en la función de pérdida de una red, vía diferenciación automática. Esto da tres cosas que la numérica clásica no entrega juntas: solución sin malla de PDEs directas/inversas, descubrimiento de campos a partir de datos escasos/ruidosos (la PDE actúa como prior físico donde no hay sensores), y un único artefacto diferenciable que se exporta y corre en el navegador."
          : "Physics-Informed Neural Networks (PINNs) embed a system's governing differential equation directly into a network's loss via automatic differentiation. That buys three things classical numerics can't give at once: mesh-free solution of forward/inverse PDEs, field discovery from sparse/noisy data (the PDE is a physics prior wherever there are no sensors), and a single differentiable artifact that exports and runs client-side."}
      </p>
      <h2>{es ? "Qué ES PINN-Lab" : "What PINN-Lab IS"}</h2>
      <ul>
        <li>
          {es
            ? "Un catálogo ejecutable de ~20 casos de PDEs: benchmarks canónicos → minería/procesamiento mineral → polución/ambiental → fluidos/calor industriales. Cada caso se entrena offline con un motor SOTA, se valida contra una referencia analítica/FEM, se exporta a ONNX y se reproduce/infiere en el navegador."
            : "A runnable catalogue of ~20 PDE cases: canonical benchmarks → mining/mineral-processing → pollution/environmental → industrial fluids/heat. Each case is trained offline by a SOTA engine, validated against an analytic/FEM reference, exported to ONNX, and replayed/inferred in the browser."}
        </li>
        <li>
          {es
            ? "Un instrumento de enseñanza y decisión: cada caso trae sus ecuaciones, el método SOTA que lo resuelve, una visualización que reacciona a los controles, y un benchmark contra una referencia."
            : "A teaching-and-decision instrument: every case carries its governing equations, the SOTA method that solves it, an interactive viz that reacts to controls, and a benchmark vs. a reference."}
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
            ? "No es un gemelo digital industrial. La mayoría de los casos de minería/polución se validan sobre datos simulados; cada uno lleva una etiqueta honesta sintético-o-validado."
            : "Not an industrial digital twin. Most mining/pollution cases are validated on simulated data; each carries an honest synthetic-or-validated label."}
        </li>
      </ul>
    </div>
  );
}
