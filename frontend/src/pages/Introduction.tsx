import { Equation, InlineMath } from "../components/Equation";
import { SubTabs } from "../components/SubTabs";
import { useUI } from "../store";

/** Introduction (ADR-0016 §6 SubTabs, §7 depth): the long non-comparative sections are chunked into sub-tabs
 *  (NN/g: tabs for a few long sections) instead of one long scroll. Deep prose + the PINN loss in KaTeX + an
 *  honest scope, bilingual EN/ES. */
export function Introduction() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";

  const whatItIs = (
    <>
      <div className="panel" style={{ margin: "0 0 16px", borderColor: "var(--accent)" }}>
        <h3 style={{ marginTop: 0 }}>{es ? "Cada caso responde una pregunta de ingeniería" : "Every case answers an engineering question"}</h3>
        <p>
          {es
            ? "Una EDP es un instrumento de medida: transporta mediciones baratas hasta la cantidad que se necesita y no puede medirse directamente. Ese es el contenido real de este catálogo: no la ecuación, sino la pregunta que cada caso responde y la estimación computada que entrega. Tres ejemplos, todos calculados por el pipeline offline (nunca afirmados):"
            : "A PDE is a measuring instrument: it transports cheap measurements to the quantity that is needed and cannot be measured directly. That is this catalogue's real content: not the equation, but the question each case answers and the computed estimate it delivers. Three examples, all computed by the offline pipeline (never asserted):"}
        </p>
        <ul>
          <li>{es ? "¿Cuál es la difusividad térmica de este suelo? No se puede medir in situ: dos series reales de temperatura de borde + la ecuación del calor la estiman (0.304 mm²/s), validada en profundidades que el optimizador nunca vio (~1 °C RMSE)." : "What is this soil's thermal diffusivity? It cannot be measured in situ: two real boundary temperature series + the heat equation estimate it (0.304 mm²/s), validated at depths the optimizer never saw (~1 degC RMSE)."}</li>
          <li>{es ? "¿Dónde aísla y dónde conduce esta placa? Un mapa de defectos desde ~100 temperaturas puntuales: 4% de error; con física sola y cero datos, 356% (irrecuperable)." : "Where is this plate insulating vs conducting? A defect map from ~100 point temperatures: 4% error; from physics alone with zero data, 356% (unrecoverable)."}</li>
          <li>{es ? "¿Cuándo llega el derrame al punto costero? Llegada a medio pico en t = 0.44 leída del campo transportado; cualquier otro punto está a una evaluación de distancia." : "When does the spill reach the coastal checkpoint? Half-peak arrival at t = 0.44 read off the transported field; any other point is one evaluation away."}</li>
        </ul>
        <p>
          {es
            ? "En el App, cada caso abre con la pregunta y la estimación antes que nada; el resto del banco de trabajo (Comparar, Campo, Live, Entrenamiento, Diagnóstico) muestra cómo se ganó ese número y cuándo desconfiar de él."
            : "In the App, every case opens with the question and the estimate before anything else; the rest of the workbench (Compare, Field, Live, Training, Diagnostics) shows how that number was earned and when to distrust it."}
        </p>
      </div>
      <h3>{es ? "Qué es PINN-Lab" : "What PINN-Lab is"}</h3>
      <ul>
        <li>{es ? "Un catálogo ejecutable de 21 casos de ecuaciones diferenciales en cuatro grupos: benchmarks canónicos, minería/procesamiento mineral, polución/ambiental, fluidos/calor industriales. Cada caso se entrena offline con un motor SOTA, se valida contra una referencia analítica/numérica, se exporta a ONNX y se reproduce/infiere en el navegador." : "A runnable catalogue of 21 differential-equation cases in four groups: canonical benchmarks, mining/mineral-processing, pollution/environmental, industrial fluids/heat. Each case is trained offline by a SOTA engine, validated against an analytic or numerical reference, exported to ONNX, and replayed/inferred in the browser."}</li>
        <li>{es ? "Un instrumento de enseñanza y decisión: cada caso es un banco de trabajo idéntico. La columna izquierda elige el caso y fija su régimen; la ecuación gobernante encabeza el escenario; y las pestañas son Resultados, Contexto, Campo, Live, Comparar, Diagnóstico y Regímenes. Cada vista llena la pantalla sin scroll." : "A teaching-and-decision instrument: each case is an identical workbench. The left column selects the case and sets its regime; the governing equation heads the stage; and the tabs are Results, Context, Field, Live, Compare, Diagnostics and Regimes. Every view fills the screen with no scrolling."}</li>
        <li>{es ? "Un catálogo de métodos: cada familia del estado del arte (hard constraints, muestreo adaptativo RAR, Fourier features/SIREN, descomposición de dominio, aprendizaje de operadores, inverso + UQ) se ejerce de verdad en al menos un caso, no solo se nombra." : "A catalogue of methods: each state-of-the-art family (hard constraints, RAR adaptive sampling, Fourier features/SIREN, domain decomposition, operator learning, inverse + UQ) is genuinely exercised in at least one case, not merely named."}</li>
      </ul>
      <h3>{es ? "Qué no es" : "What it is not"}</h3>
      <ul>
        <li>{es ? "No reemplaza a FEM/FVM. Para un problema directo bien planteado un buen solver clásico suele ser más rápido y preciso. Las PINNs ganan en problemas inversos, asimilación de datos escasos, surrogates paramétricos (operadores) y dominios de alta dimensión." : "Not a replacement for FEM/FVM. For a single well-posed forward problem a good classical solver is usually faster and more accurate. PINNs win on inverse problems, sparse-data assimilation, parametric many-query surrogates (operators), and high-dimensional domains."}</li>
        <li>{es ? "No es un gemelo digital industrial. La mayoría de los casos de minería/polución se validan sobre datos simulados o anclas analíticas (MMS); cada uno lleva una etiqueta honesta, y la página Benchmark publica el error real sin maquillar." : "Not an industrial digital twin. Most mining/pollution cases are validated on simulated data or analytic (MMS) anchors; each carries an honest label, and the Benchmark page publishes the true, un-dressed-up error."}</li>
        <li>{es ? "No es un solver caja-negra. Todo es reproducible: el pipeline offline es determinista dado (caso, semilla), y el ONNX que ejecuta el navegador es exactamente la red validada (paridad < 1e-4)." : "Not a black-box solver. Everything is reproducible: the offline pipeline is deterministic given (case, seed), and the ONNX the browser runs is exactly the validated network (parity < 1e-4)."}</li>
      </ul>
      <p className="muted">{es ? "Sigue: Metodología (los métodos), Implementación (la arquitectura), Experimentos (qué se ejecutó), Benchmark (la honestidad)." : "Continue: Methodology (the methods), Implementation (the architecture), Experiments (what was run), Benchmark (the honesty)."}</p>
    </>
  );

  const whyPdes = (
    <>
      <p>{es ? "Una ecuación diferencial parcial es una ley física escrita en forma local: la tasa de cambio en un punto queda determinada por su vecindad inmediata (flujos, gradientes, curvaturas). La conservación de masa, momento y energía toma esta forma. Resolverlas compra las tres predicciones sobre las que corre la ingeniería: el futuro de un sistema (el calor decayendo, un parche contaminante a la deriva), el interior no observado entre mediciones (reconstruir 10/20/50 cm de suelo desde sensores a 5 y 100 cm), y la respuesta a cambios de diseño (mover la viscosidad o la tasa de molienda y ver el régimen nuevo). Las formas cerradas existen solo para casos idealizados; los problemas reales exigen solucionadores numéricos, y solo por eso una nueva clase de solucionador como las PINN merece un catálogo." : "A partial differential equation is a physical law written locally: the rate of change at a point is determined by its immediate neighbourhood (fluxes, gradients, curvatures). Conservation of mass, momentum and energy all take this form. Solving them buys the three predictions engineering runs on: the future of a system (heat decaying, a pollutant patch drifting), the unseen interior between measurements (reconstructing the 10/20/50 cm soil depths from sensors at 5 and 100 cm), and the response to design changes (move the viscosity or the grind rate and see the new regime). Closed forms exist only for idealized cases; real problems demand numerical solvers, and only that makes a new solver class like PINNs worth a catalogue."}</p>
      <p>{es ? "Las Redes Neuronales Informadas por Física (PINNs) incrustan la ecuación diferencial que gobierna un sistema directamente en la función de pérdida de una red, vía diferenciación automática. En vez de ajustar datos, la red minimiza el residual de la EDP en puntos de colocación del dominio, más las condiciones de borde/iniciales y los datos que haya:" : "Physics-Informed Neural Networks (PINNs) embed a system's governing differential equation directly into a network's loss via automatic differentiation. Instead of fitting data, the network minimises the PDE residual at collocation points across the domain, plus the boundary/initial conditions and any data:"}</p>
      <Equation caption={es ? "La pérdida PINN: residual de la EDP + condiciones + datos (opcional)." : "The PINN loss: PDE residual + conditions + data (optional)."} tex={String.raw`\mathcal{L}(\theta)=\underbrace{\lambda_r\!\sum_i \big|\mathcal{N}[u_\theta](\mathbf{x}_i)\big|^2}_{\text{PDE residual}}+\underbrace{\lambda_b\!\sum_j \big|\mathcal{B}[u_\theta](\mathbf{x}_j)\big|^2}_{\text{boundary/initial}}+\underbrace{\lambda_d\!\sum_k \big|u_\theta(\mathbf{x}_k)-u_k\big|^2}_{\text{data (optional)}}`} />
      <p>{es ? (<>donde <InlineMath tex={String.raw`\mathcal{N}`} /> es el operador diferencial (p. ej. <InlineMath tex={String.raw`\ \mathcal{N}[u]=u_t-\alpha u_{xx}`} />) y <InlineMath tex={String.raw`u_\theta`} /> es la red. Las derivadas salen <em>exactas</em> por diferenciación automática, no por diferencias finitas: por eso el método es <strong>sin malla</strong>.</>) : (<>where <InlineMath tex={String.raw`\mathcal{N}`} /> is the differential operator (e.g. <InlineMath tex={String.raw`\ \mathcal{N}[u]=u_t-\alpha u_{xx}`} />) and <InlineMath tex={String.raw`u_\theta`} /> is the network. The derivatives are <em>exact</em> by automatic differentiation, not finite differences: that is what makes the method <strong>mesh-free</strong>.</>)}</p>
      <p>{es ? "Eso compra tres cosas que la numérica clásica no entrega juntas: solución sin malla de PDEs directas/inversas, descubrimiento de campos a partir de datos escasos/ruidosos (la PDE actúa como prior físico donde no hay sensores), y un único artefacto diferenciable que se exporta y se ejecuta en el navegador." : "That buys three things classical numerics can't give at once: mesh-free solution of forward/inverse PDEs, field discovery from sparse/noisy data (the PDE is a physics prior wherever there are no sensors), and a single differentiable artifact that exports and runs client-side."}</p>
    </>
  );

  const whatItReaches = (
    <>
      <p>{es ? "El punto que la optimización intenta alcanzar es la función donde todos los términos de la pérdida se anulan a la vez: residual de la EDP, condiciones de borde/iniciales y datos. Para un problema directo bien planteado, esa función de pérdida cero es la solución clásica. El intento ocurre por optimización no convexa, y ahí viven las patologías: la vista Entrenamiento muestra al carril ingenuo de Helmholtz clavado en ~100% con información matemática completa (la falla es de entrenamiento, no de capacidad). En la práctica el objetivo no es ganarle a FEM en un problema directo (ahí gana el clásico, y el caso poisson2d lo muestra), sino obtener la solución como objeto: sin malla, diferenciable en cualquier punto, paramétrica (una red = una familia entera), fusionable con datos dispersos, invertible para coeficientes ocultos y exportable (el ONNX ejecutándose en el navegador)." : "The point the optimization attempts to reach is the function where all loss terms vanish at once: the PDE residual, the boundary/initial conditions, and the data. For a well-posed forward problem, that zero-loss function is the classical solution. The attempt happens through non-convex optimization, and that is where the pathologies live: the Training view shows the naive Helmholtz lane stuck at ~100% with mathematically complete information (the failure is a training problem, not a capacity problem). In practice the target is not to beat FEM on a forward problem (the classical solver wins there, and the poisson2d case shows it), but to get the solution as an object: mesh-free, differentiable anywhere, parametric (one net = a whole family), fusable with sparse data, invertible for hidden coefficients, and exportable (the ONNX running in the browser)."}</p>
      <div className="panel" style={{ margin: "14px 0", borderColor: "var(--accent-2)" }}>
        <h3 style={{ marginTop: 0 }}>{es ? "El presupuesto de información: cuánto necesita una PINN para un resultado significativo" : "The information budget: how much a PINN needs for a significant result"}</h3>
        <ul>
          <li>{es ? "Directo bien planteado: física + condiciones de borde e iniciales son completas: cero datos (poisson 0.04%, heat1d 0.15%, burgers 0.08% vs el estándar, sin observaciones)." : "Forward, well-posed: physics + boundary + initial conditions are complete: zero data needed (poisson 0.04%, heat1d 0.15%, burgers 0.08% vs the standard, with no observations)."}</li>
          <li>{es ? "Pero información completa no es entrenabilidad: con la misma información, la Helmholtz ingenua da 121% y la allencahn suave 95%. En los directos difíciles falta el sesgo inductivo correcto (Fourier, restricciones duras, RAR), no más información." : "But information-complete is not trainable: with the same information, naive Helmholtz gives 121% and soft allencahn 95%. Hard forward problems lack the right inductive bias (Fourier, hard constraints, RAR), not more information."}</li>
          <li>{es ? "Inverso con incógnita escalar: poca data basta: dos series de borde reales recuperan una difusividad y predicen tres profundidades nunca vistas a ~1 °C." : "Inverse with a scalar unknown: little data suffices: two real boundary series recover one diffusivity and predict three held-out depths at ~1 degC."}</li>
          <li>{es ? "Inverso con incógnita de campo: los datos interiores son esenciales: 0 sensores = 356% (irrecuperable); ~100 sensores ruidosos = 4%. La curva completa de identificabilidad está computada en el Diagnóstico de heat2d-inverse." : "Inverse with a field unknown: interior data is essential: 0 sensors = 356% (unrecoverable); ~100 noisy sensors = 4%. The full identifiability curve is computed in heat2d-inverse's Diagnostics."}</li>
          <li>{es ? "El techo: ninguna cantidad de información vence al caos: el leave-time ~2 s del péndulo es un límite de Lyapunov, no un déficit de datos. Con datos escasos, el resultado honesto incluye la incertidumbre: σ del ensamble crece exactamente donde no hay sensores." : "The ceiling: no amount of information beats chaos: the pendulum's ~2 s leave-time is a Lyapunov limit, not a data deficit. Under scarce data the honest result includes the uncertainty: the ensemble's sigma grows exactly where sensors are absent."}</li>
        </ul>
      </div>
    </>
  );

  const winLose = (
    <>
      <p>{es ? "El catálogo no es una lista plana: es un arco honesto de dónde una PINN no es la herramienta a dónde sí lo es. Cada afirmación de abajo está demostrada con contenido computado (no afirmado) en el caso nombrado, en sus pestañas Comparar / Entrenamiento / Diagnóstico. La página Metodología desarrolla el porqué de cada método." : "The catalogue is not a flat list: it is an honest arc from where a PINN is not the tool to where it genuinely is. Every claim below is demonstrated with computed (not asserted) content in the named case, in its Compare / Training / Diagnostics tabs. The Methodology page develops the why behind each method."}</p>
      <ol>
        <li>{es ? "Directo fácil (poisson2d): la PINN iguala la forma cerrada (<0.1%), pero un solucionador clásico ya es exacto y más rápido: aquí la PINN no es la herramienta." : "Easy forward (poisson2d): the PINN matches the closed form (<0.1%), but a classical solver is already exact and faster: the PINN is not the tool here."}</li>
        <li>{es ? "Directo difícil (helmholtz): la PINN ingenua se queda en ~121% por sesgo espectral (la vista Entrenamiento la muestra sin aprender); las características de Fourier la llevan a 9%." : "Hard forward (helmholtz): the naive PINN sits at ~121% from spectral bias (the Training view shows it failing to learn); Fourier features bring it to 9%."}</li>
        <li>{es ? "Dinámica rígida (allencahn): la PINN suave colapsa a un estado metaestable (95%); restricciones duras + RAR siguen las capas (0.4%)." : "Stiff dynamics (allencahn): the soft PINN collapses to a metastable state (95%); hard constraints + RAR track the layers (0.4%)."}</li>
        <li>{es ? "Donde ganan (heat2d-inverse): con física pura y sin datos, k es irrecuperable (356%); con ~100 sensores dispersos, 4%. Los datos hacen soluble el inverso." : "Where they win (heat2d-inverse): pure physics with no data leaves k unrecoverable (356%); with ~100 sparse sensors, 4%. The data makes the inverse solvable."}</li>
        <li>{es ? "Datos reales (soil-heat-real): reconstruye temperaturas de suelo NOAA y valida fuera de muestra en profundidades nunca vistas (~1 °C RMSE)." : "Real data (soil-heat-real): reconstructs NOAA soil temperatures and validates out-of-sample at depths never seen (~1 degC RMSE)."}</li>
        <li>{es ? "Operadores (darcy): un FNO mapea un campo de permeabilidad nuevo a su presión en un paso (2.5% vs diferencias finitas): amortiza una familia entera." : "Operators (darcy): an FNO maps a new permeability field to its pressure in one pass (2.5% vs finite differences): amortizing a whole family."}</li>
        <li>{es ? "Incertidumbre (source-uq): un ensamble da media ± sigma; sigma crece exactamente donde faltan datos." : "Uncertainty (source-uq): an ensemble gives mean ± sigma; sigma grows exactly where data is sparse."}</li>
        <li>{es ? "El límite honesto (double-pendulum): la PINN sigue al integrador ~2 s y el caos la despega: ningún sustituto vence al exponente de Lyapunov." : "The honest limit (double-pendulum): the PINN tracks the integrator ~2 s and chaos peels it away: no surrogate beats the Lyapunov exponent."}</li>
      </ol>
    </>
  );

  return (
    <div className="prose" style={{ maxWidth: 1100 }}>
      <h1>{es ? "Qué es PINN-Lab" : "What PINN-Lab is"}</h1>
      <p className="muted">{es ? "Por qué importan las EDPs, qué intenta alcanzar una PINN, cuándo gana y cuándo pierde: el marco honesto del catálogo." : "Why PDEs matter, what a PINN attempts to reach, and when it wins and loses: the catalogue's honest frame."}</p>
      <SubTabs
        ariaLabel={es ? "Secciones de la introducción" : "Introduction sections"}
        tabs={[
          { id: "what", label: es ? "Qué es" : "What it is", content: whatItIs },
          { id: "why", label: es ? "Por qué las EDPs" : "Why PDEs", content: whyPdes },
          { id: "reaches", label: es ? "Qué alcanza" : "What it reaches", content: whatItReaches },
          { id: "winlose", label: es ? "Cuándo gana/pierde" : "When it wins/loses", content: winLose },
        ]}
      />
    </div>
  );
}
