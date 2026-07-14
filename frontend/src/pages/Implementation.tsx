import { Equation, InlineMath } from "../components/Equation";
import { SubTabs } from "../components/SubTabs";
import { useUI } from "../store";

/** Implementation page: the architecture and the three flows (offline precompute, web replay/live, authoring),
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
        <li><strong>Replay (web):</strong> el SPA dibuja el campo horneado y decimado: siempre disponible, sin red neuronal en runtime.</li>
        <li><strong>Live (web):</strong> el SPA evalúa el ONNX exportado con onnxruntime-web para barrer el parámetro en vivo.</li>
      </ul>
      <p className="muted">
        La separación es el punto: la verdad numérica se fija offline y se versiona; la web es un visor fiel, no una
        segunda implementación que pueda divergir.
      </p>
    </>
  ): (
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
        <li><strong>Replay (web):</strong> the SPA draws the baked, decimated field: always available, no neural net at runtime.</li>
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
        <li><strong>preprocess</strong>: valida la definición del caso e ingiere observaciones (problemas inversos).</li>
        <li><strong>feature / sampling</strong>: arma el plan de colocación (muestreo del dominio y, si aplica, refinamiento).</li>
        <li><strong>train</strong>: ajusta el PINN con el motor SOTA (Adam → L-BFGS, + RAR si el caso lo define), exporta a ONNX y verifica paridad.</li>
        <li><strong>infer</strong>: evalúa el campo en la grilla de cada variante (régimen de parámetro).</li>
        <li><strong>evaluate</strong>: calcula el <InlineMath tex={String.raw`L^2`} /> relativo contra la referencia analítica/numérica (la etapa de <em>test</em>).</li>
        <li><strong>export</strong>: escribe el artefacto de replay decimado + el manifest por variante.</li>
      </ol>
      <p>
        El error reportado es el <strong>L2 relativo</strong> contra el ancla de validación, sobre la grilla del campo:
      </p>
      <Equation tex={String.raw`\varepsilon_{\text{rel}}=\frac{\lVert u_\theta - u^*\rVert_2}{\lVert u^*\rVert_2},\qquad u^*=\text{referencia analítica/numérica}.`} />
      <p className="muted">
        Determinismo: misma semilla, mismo resultado. El motor pesado (DeepXDE/PhysicsNeMo) se usa AQUÍ, de verdad: no
        es decorativo.
      </p>
    </>
  ): (
    <>
      <p>The offline pipeline is a deterministic chain of stages; every case passes through all of them:</p>
      <ol>
        <li><strong>preprocess</strong>: validate the case definition and ingest observations (inverse problems).</li>
        <li><strong>feature / sampling</strong>: build the collocation plan (domain sampling and, where applicable, refinement).</li>
        <li><strong>train</strong>: fit the PINN with the SOTA engine (Adam → L-BFGS, + RAR if the case defines it), export to ONNX and verify parity.</li>
        <li><strong>infer</strong>: evaluate the field on each variant's grid (parameter regime).</li>
        <li><strong>evaluate</strong>: compute the relative <InlineMath tex={String.raw`L^2`} /> against the analytic/numerical reference (the <em>test</em> stage).</li>
        <li><strong>export</strong>: write the decimated replay artifact + the per-variant manifest.</li>
      </ol>
      <p>The reported error is the <strong>relative L2</strong> against the validation anchor, over the field grid:</p>
      <Equation tex={String.raw`\varepsilon_{\text{rel}}=\frac{\lVert u_\theta - u^*\rVert_2}{\lVert u^*\rVert_2},\qquad u^*=\text{analytic/numerical reference}.`} />
      <p className="muted">
        Determinism: same seed, same result. The heavy engine (DeepXDE/PhysicsNeMo) is used HERE, for real: it is not
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
        evalúa exactamente la misma función que se entrenó: no una aproximación.
      </p>
      <p>
        Un <strong>gate</strong> decide el carril por caso a partir de tres medidas: tamaño del ONNX, tiempo de
        inferencia en onnxruntime-web sobre la grilla completa, y bytes del artefacto de replay:
      </p>
      <Equation tex={String.raw`\text{lane}=\begin{cases}\textbf{live} & \text{onnx pequeño } \wedge\ t_{\text{infer}} \text{ bajo}\\[2pt]\textbf{precompute} & \text{en otro caso}\end{cases}`} />
    </>
  ): (
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
        trained: not an approximation.
      </p>
      <p>
        A <strong>gate</strong> picks each case's lane from three measurements: ONNX size, onnxruntime-web inference
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
        Cada caso es un <strong>banco de trabajo</strong> idéntico y de estructura estable. La columna izquierda
        elige el caso (buscador + menús de Dominio y Caso), muestra su tarjeta y qué fija la solución, y fija su
        <strong> régimen</strong> de parámetro. La <strong>ecuación gobernante</strong> encabeza el escenario a todo
        el ancho, siempre visible. Las pestañas, en orden fijo: <strong>Resultados</strong> (la pregunta y la
        respuesta computada + el veredicto + el gráfico clave), <strong>Contexto</strong> (resumen, teoría y método),
        <strong> Campo</strong>, <strong>Live</strong>, <strong>Comparar</strong>, <strong>Diagnóstico</strong> y
        <strong> Regímenes</strong>. Cada vista se dimensiona al escenario y llena la pantalla <em>sin scroll</em>.
      </p>
    </>
  ): (
    <>
      <p>The SPA loads the artifacts in a cascade, recomputing nothing:</p>
      <ol>
        <li>Read the <strong>index</strong> (case inventory) and populate the Domain and Case selectors.</li>
        <li>On selecting a case, load its <strong>manifest</strong>: field axes, parameter spec and the list of variants with their metrics.</li>
        <li>For the active variant, fetch its <strong>trace</strong> (decimated field) and draw it in the interactive, contain-fitted heatmap.</li>
        <li>In the <strong>Live</strong> tab, load the <strong>ONNX</strong> once and re-evaluate it whenever you move a parameter slider.</li>
      </ol>
      <p>
        Each case is an identical, <strong>structurally stable workbench</strong>. The left column selects the case
        (search + Domain and Case menus), shows its card and what pins the solution, and sets its
        <strong> regime</strong>. The <strong>governing equation</strong> heads the stage full-width, always visible.
        The tabs, in a fixed order: <strong>Results</strong> (the question, the computed answer, the verdict and the
        key graph), <strong>Context</strong> (summary, theory + method), <strong>Field</strong>, <strong>Live</strong>,
        <strong> Compare</strong>, <strong>Diagnostics</strong> and <strong>Regimes</strong>. Every view sizes itself
        to the stage and fills the screen with <em>no scrolling</em> (a contain-fit engine sizes each map/chart to
        the measured box, so nothing is hard-capped and the crosshair/markers stay truthful).
      </p>
    </>
  );

  const design = es ? (
    <>
      <p>Autorar un caso nuevo sigue un flujo de diseño fijo, de la física a la web:</p>
      <ol>
        <li><strong>Definir el caso:</strong> EDP, dominio, entradas/salidas, ejes del campo y el ancla de validación (analítica/MMS/numérica).</li>
        <li><strong>Elegir el método SOTA</strong> que la física exige (hard constraints, RAR, Fourier features, descomposición de dominio, operador…).</li>
        <li><strong>Parametrizar:</strong> si existe una familia con forma cerrada, el parámetro físico se vuelve una <em>entrada de la red</em> y el caso bakea ≥6 variantes; si no, se publica como un benchmark de variante única, honestamente etiquetado.</li>
        <li><strong>Hornear y verificar:</strong> correr el pipeline, exigir un <InlineMath tex={String.raw`L^2`} /> relativo dentro de la banda esperada y paridad ONNX.</li>
        <li><strong>Escribir el Context</strong> bilingüe profundo y registrarlo, luego verificar la web con capturas antes de publicar.</li>
      </ol>
      <p className="muted">
        Honestidad por diseño: cada caso declara si su ancla es analítica, numérica/benchmark o datos reales, y si los valores son
        ilustrativos-sintéticos o medidos. Nada se presenta como un gemelo digital calibrado si no lo es.
      </p>
    </>
  ): (
    <>
      <p>Authoring a new case follows a fixed design flow, from physics to web:</p>
      <ol>
        <li><strong>Define the case:</strong> PDE, domain, inputs/outputs, field axes and the validation anchor (analytic/MMS/numerical).</li>
        <li><strong>Pick the SOTA method</strong> the physics demands (hard constraints, RAR, Fourier features, domain decomposition, operator…).</li>
        <li><strong>Parametrize:</strong> if a closed-form family exists, the physical parameter becomes a <em>network input</em> and the case bakes ≥6 variants; otherwise it ships as a single-variant benchmark, honestly labeled.</li>
        <li><strong>Bake and verify:</strong> run the pipeline, require a relative <InlineMath tex={String.raw`L^2`} /> within the expected band and ONNX parity.</li>
        <li><strong>Write the deep</strong> bilingual Context and register it, then screenshot-verify the web before publishing.</li>
      </ol>
      <p className="muted">
        Honesty by design: each case declares whether its anchor is analytic, numerical/benchmark or real data, and whether values are
        illustrative-synthetic or measured. Nothing is presented as a calibrated digital twin if it is not one.
      </p>
    </>
  );

  const lessons = es ? (
    <>
      <p>
        Construir un catálogo honesto significó cometer errores reales y corregirlos con disciplina. Estos son los
        que dejaron una regla en el pipeline (no anécdotas: cada uno cambió el código o el proceso):
      </p>
      <ul>
        <li>
          <strong>Un solucionador FDM que divergió y casi se hornea.</strong> Para validar la cavidad de Navier
          se intentó un FDM propio: divergió a NaN. El guardia <InlineMath tex={String.raw`\varepsilon>0.06`} /> es
          <em> falso</em> para NaN, así que basura habría pasado el chequeo. Se revirtió y se validó contra las
          líneas centrales publicadas de Ghia. Regla: nunca hornear una referencia numérica sin sus chequeos de
          estabilidad, y guardar con <InlineMath tex={String.raw`\lnot(x<\text{tol})`} />, no con <InlineMath tex={String.raw`x>\text{tol}`} />.
        </li>
        <li>
          <strong>La trampa metaestable de Allen-Cahn.</strong> La PINN suave converge a un estado equivocado que
          <em> parece</em> convergido (95% de error, paredes de fase perdidas). La corrección: restricciones duras +
          muestreo adaptativo RAR que persigue las capas móviles (0.4%). La lección es el propio contenido del caso:
          en un directo difícil no falta información, falta el sesgo inductivo correcto.
        </li>
        <li>
          <strong>La corriente oculta y el supuesto que faltaba.</strong> El primer entrenamiento del caso insignia
          (velocidad desde tinte) daba 38-60% de error incluso donde el tinte barrió: la red gastaba su libertad en
          una velocidad variable en el tiempo que datos dispersos no pueden fijar. Declarar la corriente estacionaria
          (residuales <InlineMath tex={String.raw`u_t=v_t=0`} />) agregó todos los tiempos en un campo y bajó a 16%.
          Ese primer resultado se RECHAZÓ; el fracaso quedó documentado como contenido, no oculto.
        </li>
        <li>
          <strong>La bomba de cómputo.</strong> Los kits animados hacían autoplay de un rAF infinito que fijaba un
          núcleo de CPU. Regla permanente: animación pausada por defecto, corre UNA vez y se detiene, y se apaga
          cuando la pestaña se oculta (<InlineMath tex={String.raw`\texttt{visibilitychange}`} />).
        </li>
        <li>
          <strong>Un test que pisó los artefactos canónicos.</strong> Una prueba de humo del pipeline reescribió
          <InlineMath tex={String.raw`\texttt{data/derived}`} /> con una corrida CPU reducida, degradando la bakeada
          real. Ahora la suite entera está en un <em>sandbox</em> (los escritos van a un tmp); una prueba nunca
          escribe un artefacto canónico.
        </li>
      </ul>
    </>
  ): (
    <>
      <p>
        Building an honest catalogue meant making real mistakes and fixing them with discipline. These are the ones
        that left a rule in the pipeline (not anecdotes: each changed the code or the process):
      </p>
      <ul>
        <li>
          <strong>An FDM solver that diverged and nearly got baked.</strong> Validating the Navier cavity with a
          hand-rolled FDM diverged to NaN. The guard <InlineMath tex={String.raw`\varepsilon>0.06`} /> is <em>false</em>
          for NaN, so garbage would have passed the check. It was reverted and validated against the published Ghia
          centerlines instead. Rule: never bake a numerical reference without its stability checks, and guard with
          <InlineMath tex={String.raw`\ \lnot(x<\text{tol})`} />, not <InlineMath tex={String.raw`x>\text{tol}`} />.
        </li>
        <li>
          <strong>The Allen-Cahn metastable trap.</strong> The soft PINN converges to a wrong state that <em>looks</em>
          converged (95% error, phase walls lost). The fix: hard constraints + RAR adaptive sampling chasing the
          moving layers (0.4%). The lesson is the case's own content: a hard forward problem lacks the right inductive
          bias, not more information.
        </li>
        <li>
          <strong>The hidden current and the missing assumption.</strong> The flagship case's first training
          (velocity from dye) gave 38-60% error even where the dye swept: the net spent its freedom on a time-varying
          velocity that sparse data cannot pin. Declaring the current steady (residuals
          <InlineMath tex={String.raw`\ u_t=v_t=0`} />) aggregated all times into one field and dropped it to 16%.
          That first result was REJECTED; the failure is documented as content, not hidden.
        </li>
        <li>
          <strong>The compute bomb.</strong> The animated kits autoplayed an infinite rAF that pinned a CPU core.
          Standing rule: animation paused by default, runs ONCE and stops, and halts when the tab is hidden
          (<InlineMath tex={String.raw`\texttt{visibilitychange}`} />).
        </li>
        <li>
          <strong>A test that clobbered the canonical artifacts.</strong> A pipeline smoke test rewrote
          <InlineMath tex={String.raw`\ \texttt{data/derived}`} /> with a reduced CPU run, degrading the real bake.
          The whole suite is now <em>sandboxed</em> (writes go to a tmp dir); a test never writes a canonical artifact.
        </li>
      </ul>
    </>
  );

  return (
    <div className="prose" style={{ maxWidth: 1100 }}>
      <h1>{es ? "Implementación: arquitectura y flujos": "Implementation: architecture & flows"}</h1>
      <p className="muted">
        {es
          ? "Cómo está construido PINN-Lab: dos mundos (offline pesado / web liviano) unidos por un contrato de datos, los flujos de precómputo, web y diseño, y las lecciones de ingeniería que dejaron reglas en el pipeline."
         : "How PINN-Lab is built: two worlds (heavy offline / light web) joined by a data contract, the precompute, web and design flows, and the engineering lessons that left rules in the pipeline."}
      </p>
      <SubTabs
        ariaLabel={es ? "Flujos de implementación": "Implementation flows"}
        tabs={[
          { id: "overview", label: es ? "App y carriles": "App & lanes", content: overview },
          { id: "offline", label: es ? "Precómputo offline": "Offline precompute", content: offline },
          { id: "bridge", label: es ? "Puente → ONNX": "Train → ONNX bridge", content: bridge },
          { id: "web", label: es ? "Flujo web": "Web flow", content: webflow },
          { id: "design", label: es ? "Flujo de diseño": "Design flow", content: design },
          { id: "lessons", label: es ? "Lecciones de ingeniería": "Engineering lessons", content: lessons },
        ]}
      />
    </div>
  );
}
