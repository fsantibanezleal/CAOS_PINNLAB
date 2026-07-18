/** The situation layer (the owner's core critique, 2026-07-14): the app must establish what is being evaluated
 *  and why before showing any solver mechanics. Each case gets a concrete scenario with stakes: who is asking,
 *  what hangs on the answer, and what can actually be measured. The workbench leads with this; the PDE-vs-PINN
 *  comparison is evidence inside the investigation, not the show. All claims consistent with each case's baked
 *  content and honesty labels (synthetic scenarios are illustrative framings of the physics, and say so). */

export interface Scenario {
  /** the concrete situation, one or two sentences with stakes (who needs this and what happens if unknown) */
  situation_en: string;
  situation_es: string;
  /** what we can actually measure or know in this situation (the information at hand) */
  measured_en: string;
  measured_es: string;
}

export const SCENARIOS: Record<string, Scenario> = {
  "bench-heat1d": {
    situation_en: "A quenched steel bar leaves the bath and must not be handled until its core has cooled. Holding it too long wastes line time; too short burns someone. The plant needs the cooling time for every alloy it runs, not one lab measurement.",
    situation_es: "Una barra de acero templada sale del baño y no puede manipularse hasta que su núcleo se enfríe. Retenerla de más pierde tiempo de línea; de menos, quema a alguien. La planta necesita el tiempo de enfriamiento para cada aleación que procesa, no una medición de laboratorio.",
    measured_en: "The bar's surface is held at bath temperature (the boundary), the initial heat profile is known, and each alloy's diffusivity α is tabulated. Nothing inside the bar is measured.",
    measured_es: "La superficie queda a la temperatura del baño (el borde), el perfil inicial de calor se conoce, y la difusividad α de cada aleación está tabulada. Nada dentro de la barra se mide.",
  },
  "bench-poisson2d": {
    situation_en: "A membrane, an electrostatic potential, a steady temperature over a plate: the same elliptic balance appears everywhere a field settles under a fixed load. Before trusting any new solver on hard problems, it must prove itself where the exact answer is known.",
    situation_es: "Una membrana, un potencial electrostático, una temperatura estacionaria sobre una placa: el mismo balance elíptico aparece donde un campo se asienta bajo una carga fija. Antes de confiar en un solucionador nuevo en problemas difíciles, se le exige demostrar donde la respuesta exacta se conoce.",
    measured_en: "The load (source term) and the clamped edges are fully known: mathematically complete information, zero data needed. The closed-form solution exists to grade against.",
    measured_es: "La carga (término fuente) y los bordes fijos se conocen por completo: información matemáticamente completa, cero datos. Existe la solución cerrada para calificar.",
  },
  "bench-wave1d": {
    situation_en: "A tensioned cable or string is plucked: its oscillation period sets resonance risk for whatever it is attached to. Get the period wrong and a driven structure can be excited at exactly the wrong frequency.",
    situation_es: "Un cable o cuerda tensada se pulsa: su período de oscilación fija el riesgo de resonancia de lo que tenga conectado. Errar el período y una estructura forzada puede excitarse justo en la frecuencia equivocada.",
    measured_en: "The end anchors (fixed), the initial pluck shape, and the wave speed c from tension and density. The question is what motion follows.",
    measured_es: "Los anclajes (fijos), la forma inicial del pulso, y la velocidad de onda c desde tensión y densidad. La pregunta es qué movimiento sigue.",
  },
  "bench-burgers1d": {
    situation_en: "A steep front races down a line: traffic compressing into a jam, gas piling into a shock. The operator's question is when it arrives at a checkpoint: the warning time available before the front hits.",
    situation_es: "Un frente abrupto avanza por una línea: tráfico comprimiéndose en un taco, gas apilándose en un choque. La pregunta del operador es cuándo llega a un punto de control: el tiempo de aviso antes de que golpee.",
    measured_en: "The initial profile and the viscosity are known; the nonlinearity does the rest. The front position at every time follows from the solve.",
    measured_es: "El perfil inicial y la viscosidad se conocen; la no linealidad hace el resto. La posición del frente en cada instante sale de la solución.",
  },
  "bench-allencahn": {
    situation_en: "Two phases of a material separate and their walls creep as the microstructure coarsens: where the walls end up determines grain size, and grain size determines strength. Simulating this wrongly predicts the wrong material.",
    situation_es: "Dos fases de un material se separan y sus paredes migran mientras la microestructura se engrosa: dónde terminan las paredes determina el tamaño de grano, y el grano la resistencia. Simular esto mal predice el material equivocado.",
    measured_en: "The initial phase mixture and the interface energy are known. The trap: the equation has metastable states, and a lazy solver settles into the wrong one while looking converged.",
    measured_es: "La mezcla inicial de fases y la energía de interfaz se conocen. La trampa: la ecuación tiene estados metaestables, y un solucionador perezoso cae en el incorrecto pareciendo convergido.",
  },
  "dyn-double-pendulum": {
    situation_en: "A crane hook swinging under its load, a robot's double joint: chaotic mechanics. Any digital twin of such a machine has an expiry date, and deploying one without knowing that date is how surrogates fail silently in production.",
    situation_es: "Un gancho de grúa oscilando bajo su carga, la articulación doble de un robot: mecánica caótica. Cualquier gemelo digital de esa máquina tiene fecha de expiración, y desplegarlo sin conocer esa fecha es cómo los sustitutos fallan en silencio en producción.",
    measured_en: "The exact initial state and the equations of motion: perfect information. Chaos still wins past the divergence horizon; the case measures that horizon (~2 s here).",
    measured_es: "El estado inicial exacto y las ecuaciones de movimiento: información perfecta. El caos igual gana pasado el horizonte de divergencia; el caso mide ese horizonte (~2 s aquí).",
  },
  "bench-navier-cavity": {
    situation_en: "Every stirred tank, coating bath and cooling channel hides recirculation: where the vortex sits decides what gets mixed and what stagnates. The lid-driven cavity is the standardized version every CFD code is graded on.",
    situation_es: "Todo estanque agitado, baño de recubrimiento y canal de refrigeración esconde recirculación: dónde queda el vórtice decide qué se mezcla y qué se estanca. La cavidad de tapa móvil es la versión estandarizada con la que se califica todo código CFD.",
    measured_en: "The moving lid speed and the no-slip walls; Reynolds number set. Graded against the published Ghia benchmark centerlines.",
    measured_es: "La velocidad de la tapa y las paredes sin deslizamiento; número de Reynolds fijo. Calificado contra las líneas centrales publicadas de Ghia.",
  },
  "ind-helmholtz": {
    situation_en: "An acoustic source drives a cavity at a fixed frequency: what amplitude reaches a receiver across the room? Speaker placement, ultrasound focusing and EM antenna design all live on this answer, and it gets harder as frequency rises.",
    situation_es: "Una fuente acústica excita una cavidad a frecuencia fija: ¿qué amplitud llega a un receptor al otro lado? La ubicación de parlantes, el enfoque de ultrasonido y el diseño de antenas EM viven de esta respuesta, y se hace más difícil al subir la frecuencia.",
    measured_en: "The source, the walls, the frequency: complete information. The failure is not missing data but spectral bias: a plain network cannot even represent the oscillation (visible in Training).",
    measured_es: "La fuente, las paredes, la frecuencia: información completa. La falla no es falta de datos sino sesgo espectral: una red simple ni siquiera representa la oscilación (visible en Entrenamiento).",
  },
  "ind-heat2d-inverse": {
    situation_en: "A plate somewhere in service is insulating where it should conduct: a delamination, a void, moisture. It cannot be cut open. Maintenance needs a map of the defect from the outside, from a scatter of point temperature readings.",
    situation_es: "Una placa en servicio aísla donde debería conducir: una delaminación, un vacío, humedad. No se puede abrir. Mantenimiento necesita un mapa del defecto desde afuera, desde un puñado de lecturas puntuales de temperatura.",
    measured_en: "~100 noisy temperature points and the known heating. The unknown is an entire field k(x,y): without the physics prior, 100 points cannot determine it; without the data, nothing can (356%, computed).",
    measured_es: "~100 puntos de temperatura con ruido y el calentamiento conocido. La incógnita es un campo entero k(x,y): sin el prior físico, 100 puntos no lo determinan; sin los datos, nada puede (356%, computado).",
  },
  "ind-hidden-velocity": {
    situation_en: "A harbor current, blood in a vessel, gas in a furnace: a velocity probe cannot be placed everywhere, but what the flow carries can be seen: dye, contrast agent, smoke. The whole current field must come from those images alone (the Science 2020 Hidden Fluid Mechanics setting).",
    situation_es: "Una corriente de puerto, sangre en un vaso, gas en un horno: no puede ponerse una sonda de velocidad en todas partes, pero puede verse lo que el flujo arrastra: tinte, contraste, humo. Todo el campo de corriente debe salir de esas imágenes solamente (el escenario de Hidden Fluid Mechanics, Science 2020).",
    measured_en: "~640 sparse noisy dye samples over time. No velocity data anywhere, no boundary or initial conditions assumed: the transport physics plus a declared steady-current assumption do all the bridging.",
    measured_es: "~640 muestras dispersas y ruidosas de tinte en el tiempo. Ningún dato de velocidad, sin condiciones de borde o iniciales asumidas: la física del transporte más un supuesto declarado de corriente estacionaria hacen todo el puente.",
  },
  "poll-ocean-transport": {
    situation_en: "A spill is drifting toward the coast. The emergency team gets one question: when does it reach the water intake, and how concentrated? Every response decision: closures, booms, warnings: hangs on that arrival time.",
    situation_es: "Un derrame deriva hacia la costa. El equipo de emergencia recibe una pregunta: ¿cuándo llega a la toma de agua, y cuán concentrado? Cada decisión de respuesta: cierres, barreras, avisos: cuelga de ese tiempo de llegada.",
    measured_en: "The release point, the current, and the eddy diffusion. The exact drifting-spreading solution exists here, so the transport answer is exactly gradeable.",
    measured_es: "El punto de vertido, la corriente y la difusión turbulenta. La solución exacta de deriva y esparcimiento existe aquí, así que la respuesta de transporte es calificable exactamente.",
  },
  "poll-soil-barrier": {
    situation_en: "A contaminated site got a cutoff wall. The regulator's question is blunt: how many years does the wall actually buy downstream? Design the wall on a wrong delay and the plume arrives while everyone believes it is contained.",
    situation_es: "Un sitio contaminado recibió un muro pantalla. La pregunta del regulador es directa: ¿cuántos años compra realmente el muro aguas abajo? Diseñar el muro con un retraso errado y el penacho llega mientras todos lo creen contenido.",
    measured_en: "The soil and wall permeabilities and the source history. The sharp material jump at the wall is exactly what breaks smooth single-network solvers (hence the domain decomposition).",
    measured_es: "Las permeabilidades del suelo y del muro y la historia de la fuente. El salto brusco de material en el muro es justo lo que rompe a los solucionadores suaves de una sola red (de ahí la descomposición de dominio).",
  },
  "poll-tailings-seepage": {
    situation_en: "A tailings deposit's stability depends on suction in the unsaturated zone: lose suction and strength goes with it. Dam-safety screening must ask the same question across every material the deposit might contain: a family, not one case.",
    situation_es: "La estabilidad de un depósito de relaves depende de la succión en la zona no saturada: se pierde la succión y la resistencia se va con ella. El cribado de seguridad debe hacer la misma pregunta para cada material posible del depósito: una familia, no un caso.",
    measured_en: "Surface infiltration, the water table, and each candidate material's sorptive number α: the parametric axis one network carries whole.",
    measured_es: "La infiltración superficial, el nivel freático, y el número sortivo α de cada material candidato: el eje paramétrico que una sola red lleva completo.",
  },
  "poll-source-uq-bpinn": {
    situation_en: "Someone is discharging into the channel upstream: sensors are few and noisy, and the compliance ruling ('is the limit exceeded at the checkpoint?') will be challenged. An estimate without error bars is legally and scientifically worthless here.",
    situation_es: "Alguien descarga al canal aguas arriba: los sensores son pocos y ruidosos, y el veredicto de cumplimiento ('¿se supera el límite en el punto de control?') será impugnado. Una estimación sin barras de error es legal y científicamente inútil aquí.",
    measured_en: "Sparse noisy concentration readings plus the transport physics. The ensemble's sigma marks exactly where the answer should not be trusted: where sensors are absent.",
    measured_es: "Lecturas dispersas y ruidosas de concentración más la física del transporte. La sigma del ensamble marca exactamente dónde no confiar en la respuesta: donde no hay sensores.",
  },
  "env-soil-heat-real": {
    situation_en: "Ground-source heat pumps, buried cables and permafrost models all need one soil property: thermal diffusivity. There is no instrument that can be pushed into the ground to read it. This case recovers it from real NOAA station temperatures: the one case here on real data.",
    situation_es: "Las bombas de calor geotérmicas, los cables enterrados y los modelos de permafrost necesitan una propiedad del suelo: la difusividad térmica. No existe instrumento que se entierre y la lea. Este caso la recupera desde temperaturas reales de una estación NOAA: el único caso aquí con datos reales.",
    measured_en: "Two real boundary temperature series (5 cm and 100 cm). Validation is out-of-sample: three interior depths the optimizer never saw, reconstructed to ~1 degC.",
    measured_es: "Dos series reales de temperatura de borde (5 y 100 cm). La validación es fuera de muestra: tres profundidades interiores que el optimizador nunca vio, reconstruidas a ~1 °C.",
  },
  "mine-thickener-settling": {
    situation_en: "Every concentrator ends in thickeners: size them wrong and the plant either overflows fines to the dam or starves the filters. Sizing runs on settling curves: how fast the mudline falls for this ore at this flocculant dose.",
    situation_es: "Toda concentradora termina en espesadores: dimensionarlos mal y la planta rebosa finos a la presa o mata de hambre a los filtros. El dimensionamiento se basa en curvas de sedimentación: qué tan rápido cae la línea de lodo para este mineral con esta dosis de floculante.",
    measured_en: "The initial slurry and the flux law with its rate parameter R: the family axis. One network carries the whole design chart of settling curves.",
    measured_es: "La pulpa inicial y la ley de flujo con su parámetro R: el eje de la familia. Una red lleva la carta de diseño completa de curvas de sedimentación.",
  },
  "mine-flotation-kinetics": {
    situation_en: "Flotation circuit design asks one thing of each cell: what recovery in what residence time, for ores whose rate constant k varies day to day. The design chart R(k, t) is what gets a circuit sized.",
    situation_es: "El diseño de circuitos de flotación pide una cosa por celda: qué recuperación en qué tiempo de residencia, para minerales cuyo k varía día a día. La carta de diseño R(k, t) es lo que dimensiona un circuito.",
    measured_en: "First-order kinetics with k as a network input: the whole ore family in one net, recovery and t90 readable for any k without re-solving.",
    measured_es: "Cinética de primer orden con k como entrada de la red: toda la familia de minerales en una red, recuperación y t90 legibles para cualquier k sin re-resolver.",
  },
  "mine-heap-leach-rt": {
    situation_en: "A heap leach only pays when the reagent front actually reaches the bottom: irrigate too fast and reagent is wasted, too slow and recovery is months late. This case is the catalogue's coupled-PINN stress test: one network solving two nonlinear advection-diffusion-reaction PDEs at once, validated against an exact manufactured (MMS) solution.",
    situation_es: "Una pila de lixiviación solo paga cuando el frente de reactivo llega realmente al fondo: regar muy rápido malgasta reactivo, muy lento atrasa la recuperación por meses. Este caso es el test de estrés de PINN acoplada del catálogo: una sola red resolviendo dos EDPs no lineales de advección-difusión-reacción a la vez, validada contra una solución exacta manufacturada (MMS).",
    measured_en: "Irrigation at the top, the heap's transport and reaction parameters. The 2-D two-species reactive-transport field over time c(x,z;t), scored against an exact manufactured (MMS) reference per snapshot.",
    measured_es: "El riego en la cima, los parámetros de transporte y reacción de la pila. El campo 2-D de transporte reactivo de dos especies en el tiempo c(x,z;t), calificado contra una referencia exacta manufacturada (MMS) por instantánea.",
  },
  "mine-comminution-pbe": {
    situation_en: "Grinding is the single largest energy cost of a concentrator, and it is controlled by one curve: the passing fraction below target size versus grind time. Overgrinding burns power; undergrinding loses metal downstream in flotation.",
    situation_es: "La molienda es el mayor costo energético de una concentradora, y la controla una curva: la fracción pasante bajo el tamaño objetivo versus el tiempo de molienda. Sobremoler quema energía; submoler pierde metal en la flotación siguiente.",
    measured_en: "The feed size distribution and the size-transport (drift-diffusion) reduction of the breakage operator with grind rate g as the family axis: the operating chart in one network.",
    measured_es: "La distribución de tamaños de alimentación y la reducción de transporte de tamaño (deriva-difusión) del operador de rotura con la tasa g como eje de familia: la carta de operación en una red.",
  },
  "ctrl-zero-source": {
    situation_en: "Every measuring instrument gets a null test: feed it a case whose true answer is exactly zero and check it reads zero. An estimator that invents structure from nothing cannot be trusted on anything else. Here that null test is the a=0 limit of a manufactured Poisson family (amplitude a as a network input) that also verifies the solver across a range of loads.",
    situation_es: "Todo instrumento de medida recibe una prueba nula: se le da un caso cuya respuesta verdadera es exactamente cero y se verifica que lee cero. Un estimador que inventa estructura desde la nada no es confiable en nada más. Aquí esa prueba nula es el límite a=0 de una familia de Poisson manufacturada (la amplitud a como entrada de la red) que además verifica el solucionador en un rango de cargas.",
    measured_en: "The truth is the exact manufactured solution u*(x,y;a) over six amplitudes (a = 0 to 1). Only a=0 is the zero-source null control (field essentially flat zero); for a >= 0.2 the network recovers the two-mode field to relative-L2 <= 0.15% (seed 42).",
    measured_es: "La verdad es la solución exacta manufacturada u*(x,y;a) sobre seis amplitudes (a = 0 a 1). Solo a=0 es el control nulo de fuente cero (campo esencialmente plano cero); para a >= 0.2 la red recupera el campo de dos modos con L2 relativo <= 0.15% (semilla 42).",
  },
  "bench-darcy-operator": {
    situation_en: "Groundwater screening needs the pressure response of thousands of candidate geology maps: one classical solve each is the bottleneck. An operator surrogate answers each new map in a single forward pass.",
    situation_es: "El cribado de aguas subterráneas necesita la respuesta de presión de miles de mapas geológicos candidatos: una resolución clásica por mapa es el cuello de botella. Un surrogate de operador responde cada mapa nuevo en una sola pasada.",
    measured_en: "A training set of permeability-pressure pairs from a classical solver; graded on held-out maps it never saw (5.5% mean relative-L2 across 64 held-out maps vs finite differences).",
    measured_es: "Un set de entrenamiento de pares permeabilidad-presión de un solucionador clásico; calificado en mapas retenidos que nunca vio (5.5% de L2 relativo promedio en 64 mapas retenidos vs diferencias finitas).",
  },
};
