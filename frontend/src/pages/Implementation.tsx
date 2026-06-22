import { Equation, InlineMath } from "../components/Equation";
import { SubTabs } from "../components/SubTabs";
import { useUI } from "../store";

/** Implementation page — the architecture and the three flows (offline precompute, web replay/live, authoring),
 *  to ADR-0016 §9 depth: sub-tabs + KaTeX, and ZERO internal repo paths in the rendered UI. */
export function Implementation() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";

  const overview = es ? (
    <>
      <p>
        PINN-Lab está partido en <strong>dos mundos</strong> unidos por un contrato de datos. El <em>mundo offline</em>
        (Python, pesado) entrena cada PINN con un motor del estado del arte, lo valida contra una referencia y hornea
        artefactos compactos. El <em>mundo web</em> (este SPA, liviano) <strong>nunca recomputa la física</strong>: solo
        carga esos artefactos y, cuando el modelo es lo bastante chico, corre la red exportada en tu navegador.
      </p>
      <p>Eso da <strong>tres carriles</strong> de ejecución:</p>
      <ul>
        <li><strong>Precómputo (offline):</strong> el entrenamiento real, una vez, determinista dado <InlineMath tex={String.raw`(\text{caso},\ \text{semilla})`} />.</li>
        <li><strong>Replay (web):</strong> el SPA dibuja el campo horneado y decimado — siempre disponible, sin red neuronal en runtime.</li>
        <li><strong>Live (web):</strong> el SPA evalúa el ONNX exportado con onnxruntime-web para barrer el parámetro en vivo.</li>
      </ul>
      <p className="muted">
        La separación es el punto: la verdad numérica se fija offline y se versiona; la web es un visor fiel, no una
        segunda implementación que pueda divergir.
      </p>
    </>
  ) : (
    <>
      <p>
        PINN-Lab is split into <strong>two worlds</strong> joined by a data contract. The <em>offline world</em>
        (Python, heavy) trains each PINN with a state-of-the-art engine, validates it against a reference and bakes
        compact artifacts. The <em>web world</em> (this SPA, light) <strong>never recomputes the physics</strong>: it
        only loads those artifacts and, when the model is small enough, runs the exported network in your browser.
      </p>
      <p>That gives <strong>three execution lanes</strong>:</p>
      <ul>
        <li><strong>Precompute (offline):</strong> the real training, once, deterministic given <InlineMath tex={String.raw`(\text{case},\ \text{seed})`} />.</li>
        <li><strong>Replay (web):</strong> the SPA draws the baked, decimated field — always available, no neural net at runtime.</li>
        <li><strong>Live (web):</strong> the SPA evaluates the exported ONNX with onnxruntime-web to sweep the parameter live.</li>
      </ul>
      <p className="muted">
        The separation is the point: numerical truth is fixed offline and versioned; the web is a faithful viewer, not a
        second implementation that can drift.
      </p>
    </>
  );

  const offline = es ? (
    <>
      <p>El pipeline offline es una cadena determinista de etapas; cada caso pasa por todas:</p>
      <ol>
        <li><strong>preprocess</strong> — valida la definición del caso e ingiere observaciones (problemas inversos).</li>
        <li><strong>feature / sampling</strong> — arma el plan de colocación (muestreo del dominio y, si aplica, refinamiento).</li>
        <li><strong>train</strong> — ajusta el PINN con el motor SOTA (Adam → L-BFGS, + RAR si el caso lo define), exporta a ONNX y verifica paridad.</li>
        <li><strong>infer</strong> — evalúa el campo en la grilla de cada variante (régimen de parámetro).</li>
        <li><strong>evaluate</strong> — calcula el <InlineMath tex={String.raw`L^2`} /> relativo contra la referencia analítica/FEM (la etapa de <em>test</em>).</li>
        <li><strong>export</strong> — escribe el artefacto de replay decimado + el manifest por variante.</li>
      </ol>
      <p>
        El error reportado es el <strong>L2 relativo</strong> contra el ancla de validación, sobre la grilla del campo:
      </p>
      <Equation tex={String.raw`\varepsilon_{\text{rel}}=\frac{\lVert u_\theta - u^*\rVert_2}{\lVert u^*\rVert_2},\qquad u^*=\text{referencia analítica/FEM}.`} />
      <p className="muted">
        Determinismo: misma semilla, mismo resultado. El motor pesado (DeepXDE/PhysicsNeMo) se usa AQUÍ, de verdad — no
        es decorativo.
      </p>
    </>
  ) : (
    <>
      <p>The offline pipeline is a deterministic chain of stages; every case passes through all of them:</p>
      <ol>
        <li><strong>preprocess</strong> — validate the case definition and ingest observations (inverse problems).</li>
        <li><strong>feature / sampling</strong> — build the collocation plan (domain sampling and, where applicable, refinement).</li>
        <li><strong>train</strong> — fit the PINN with the SOTA engine (Adam → L-BFGS, + RAR if the case defines it), export to ONNX and verify parity.</li>
        <li><strong>infer</strong> — evaluate the field on each variant's grid (parameter regime).</li>
        <li><strong>evaluate</strong> — compute the relative <InlineMath tex={String.raw`L^2`} /> against the analytic/FEM reference (the <em>test</em> stage).</li>
        <li><strong>export</strong> — write the decimated replay artifact + the per-variant manifest.</li>
      </ol>
      <p>The reported error is the <strong>relative L2</strong> against the validation anchor, over the field grid:</p>
      <Equation tex={String.raw`\varepsilon_{\text{rel}}=\frac{\lVert u_\theta - u^*\rVert_2}{\lVert u^*\rVert_2},\qquad u^*=\text{analytic/FEM reference}.`} />
      <p className="muted">
        Determinism: same seed, same result. The heavy engine (DeepXDE/PhysicsNeMo) is used HERE, for real — it is not
        decorative.
      </p>
    </>
  );

  const bridge = es ? (
    <>
      <p>
        El puente entre el entrenamiento y el navegador es la exportación a <strong>ONNX</strong>. La red entrenada se
        exporta (opset 18, ejes dinámicos para evaluar cualquier número de coordenadas) y se comprueba que el grafo ONNX
        reproduce <InlineMath tex={String.raw`\texttt{model.predict}`} /> sobre una muestra aleatoria del dominio:
      </p>
      <Equation tex={String.raw`\max_{\mathbf{x}}\big|\,u_{\text{ONNX}}(\mathbf{x})-u_{\text{torch}}(\mathbf{x})\,\big| < 10^{-4}\quad(\text{paridad}).`} />
      <p>
        Lo clave: las <strong>transformaciones de salida</strong> que imponen las condiciones de borde/iniciales de forma
        exacta (hard constraints) y las <strong>codificaciones de entrada</strong> (Fourier features, SIREN) son
        operaciones tensoriales puras, así que quedan <em>capturadas dentro del grafo ONNX</em>. Por eso el tab Live
        evalúa exactamente la misma función que se entrenó — no una aproximación.
      </p>
      <p>
        Un <strong>gate</strong> decide el carril por caso a partir de tres medidas — tamaño del ONNX, tiempo de
        inferencia en onnxruntime-web sobre la grilla completa, y bytes del artefacto de replay:
      </p>
      <Equation tex={String.raw`\text{lane}=\begin{cases}\textbf{live} & \text{onnx pequeño } \wedge\ t_{\text{infer}} \text{ bajo}\\[2pt]\textbf{precompute} & \text{en otro caso}\end{cases}`} />
    </>
  ) : (
    <>
      <p>
        The bridge from training to the browser is the <strong>ONNX</strong> export. The trained net is exported (opset
        18, dynamic axes so it can evaluate any number of coordinates) and the ONNX graph is checked to reproduce
        <InlineMath tex={String.raw`\texttt{model.predict}`} /> on a random in-domain sample:
      </p>
      <Equation tex={String.raw`\max_{\mathbf{x}}\big|\,u_{\text{ONNX}}(\mathbf{x})-u_{\text{torch}}(\mathbf{x})\,\big| < 10^{-4}\quad(\text{parity}).`} />
      <p>
        The crux: the <strong>output transforms</strong> that impose the boundary/initial conditions exactly (hard
        constraints) and the <strong>input encodings</strong> (Fourier features, SIREN) are pure tensor ops, so they are
        <em> captured inside the ONNX graph</em>. That is why the Live tab evaluates exactly the function that was
        trained — not an approximation.
      </p>
      <p>
        A <strong>gate</strong> picks each case's lane from three measurements — ONNX size, onnxruntime-web inference
        time over the full grid, and replay-artifact bytes:
      </p>
      <Equation tex={String.raw`\text{lane}=\begin{cases}\textbf{live} & \text{small onnx } \wedge\ t_{\text{infer}} \text{ low}\\[2pt]\textbf{precompute} & \text{otherwise}\end{cases}`} />
    </>
  );

  const webflow = es ? (
    <>
      <p>El SPA carga los artefactos en cascada, sin recomputar nada:</p>
      <ol>
        <li>Lee el <strong>índice</strong> (inventario de casos) y arma una pestaña por caso.</li>
        <li>Al abrir un caso, carga su <strong>manifest</strong>: ejes del campo, especificación de parámetros y la lista de variantes con sus métricas.</li>
        <li>Para la variante activa, descarga su <strong>traza</strong> (campo decimado) y la dibuja en el heatmap interactivo.</li>
        <li>En el tab <strong>Live</strong>, carga el <strong>ONNX</strong> una vez y lo re-evalúa cada vez que mueves un deslizador de parámetro.</li>
      </ol>
      <p>
        Cada caso se presenta como un <strong>banco de trabajo</strong> idéntico: una barra de variantes (chips de
        régimen + insignia del carril + nota bilingüe) y cuatro sub-tabs — <strong>Field</strong> (el campo +
        ecuaciones + lectura al cursor), <strong>Live</strong> (re-evaluación del ONNX), <strong>Charts</strong>
        (comparación de L2 por variante) y <strong>Context</strong> (el texto profundo del caso).
      </p>
    </>
  ) : (
    <>
      <p>The SPA loads the artifacts in a cascade, recomputing nothing:</p>
      <ol>
        <li>Read the <strong>index</strong> (case inventory) and build one tab per case.</li>
        <li>On opening a case, load its <strong>manifest</strong>: field axes, parameter spec and the list of variants with their metrics.</li>
        <li>For the active variant, fetch its <strong>trace</strong> (decimated field) and draw it in the interactive heatmap.</li>
        <li>In the <strong>Live</strong> tab, load the <strong>ONNX</strong> once and re-evaluate it whenever you move a parameter slider.</li>
      </ol>
      <p>
        Each case is presented as an identical <strong>workbench</strong>: a variant bar (regime chips + lane badge +
        bilingual note) and four sub-tabs — <strong>Field</strong> (the field + equations + cursor read-out),
        <strong> Live</strong> (ONNX re-evaluation), <strong>Charts</strong> (per-variant L2 comparison) and
        <strong> Context</strong> (the case's deep write-up).
      </p>
    </>
  );

  const design = es ? (
    <>
      <p>Autorar un caso nuevo sigue un flujo de diseño fijo, de la física a la web:</p>
      <ol>
        <li><strong>Definir el caso:</strong> EDP, dominio, entradas/salidas, ejes del campo y el ancla de validación (analítica/MMS/FEM).</li>
        <li><strong>Elegir el método SOTA</strong> que la física exige (hard constraints, RAR, Fourier features, descomposición de dominio, operador…).</li>
        <li><strong>Parametrizar:</strong> si existe una familia con forma cerrada, el parámetro físico se vuelve una <em>entrada de la red</em> y el caso bakea ≥6 variantes; si no, se publica como un benchmark de variante única, honestamente etiquetado.</li>
        <li><strong>Hornear y verificar:</strong> correr el pipeline, exigir un <InlineMath tex={String.raw`L^2`} /> relativo dentro de la banda esperada y paridad ONNX.</li>
        <li><strong>Escribir el Context</strong> bilingüe profundo y registrarlo, luego verificar la web con capturas antes de publicar.</li>
      </ol>
      <p className="muted">
        Honestidad por diseño: cada caso declara si su ancla es analítica, FEM o datos reales, y si los valores son
        ilustrativos-sintéticos o medidos. Nada se presenta como un gemelo digital calibrado si no lo es.
      </p>
    </>
  ) : (
    <>
      <p>Authoring a new case follows a fixed design flow, from physics to web:</p>
      <ol>
        <li><strong>Define the case:</strong> PDE, domain, inputs/outputs, field axes and the validation anchor (analytic/MMS/FEM).</li>
        <li><strong>Pick the SOTA method</strong> the physics demands (hard constraints, RAR, Fourier features, domain decomposition, operator…).</li>
        <li><strong>Parametrize:</strong> if a closed-form family exists, the physical parameter becomes a <em>network input</em> and the case bakes ≥6 variants; otherwise it ships as a single-variant benchmark, honestly labeled.</li>
        <li><strong>Bake and verify:</strong> run the pipeline, require a relative <InlineMath tex={String.raw`L^2`} /> within the expected band and ONNX parity.</li>
        <li><strong>Write the deep</strong> bilingual Context and register it, then screenshot-verify the web before publishing.</li>
      </ol>
      <p className="muted">
        Honesty by design: each case declares whether its anchor is analytic, FEM or real data, and whether values are
        illustrative-synthetic or measured. Nothing is presented as a calibrated digital twin if it is not one.
      </p>
    </>
  );

  return (
    <div className="prose" style={{ maxWidth: 1100 }}>
      <h1>{es ? "Implementación — arquitectura y flujos" : "Implementation — architecture & flows"}</h1>
      <p className="muted">
        {es
          ? "Cómo está construido PINN-Lab: dos mundos (offline pesado / web liviano) unidos por un contrato de datos, y los flujos de precómputo, web y diseño."
          : "How PINN-Lab is built: two worlds (heavy offline / light web) joined by a data contract, and the precompute, web and design flows."}
      </p>
      <SubTabs
        ariaLabel={es ? "Flujos de implementación" : "Implementation flows"}
        tabs={[
          { id: "overview", label: es ? "App y carriles" : "App & lanes", content: overview },
          { id: "offline", label: es ? "Precómputo offline" : "Offline precompute", content: offline },
          { id: "bridge", label: es ? "Puente → ONNX" : "Train → ONNX bridge", content: bridge },
          { id: "web", label: es ? "Flujo web" : "Web flow", content: webflow },
          { id: "design", label: es ? "Flujo de diseño" : "Design flow", content: design },
        ]}
      />
    </div>
  );
}
