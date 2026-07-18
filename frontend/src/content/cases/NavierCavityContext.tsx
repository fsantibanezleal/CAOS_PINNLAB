import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-navier-cavity (2D lid-driven cavity, steady incompressible Navier-Stokes,
 *  coupled u-v-p primitive-variable PINN, validated against the Ghia 1982 Re=100 centerlines: single benchmark). */
export function NavierCavityContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: la cavidad con tapa móvil: Navier-Stokes incompresible estacionario</h2>
      <p>
        <strong>El problema.</strong> La <em>cavidad con tapa móvil</em> (lid-driven cavity) es el banco de pruebas
        canónico de la dinámica de fluidos incompresible: un cuadrado de fluido <InlineMath tex={String.raw`(0,1)^2`} />
        cuyas tres paredes están fijas y cuya <strong>tapa superior</strong> se desliza horizontalmente. El arrastre
        viscoso de la tapa pone a girar todo el fluido, formando un <strong>vórtice primario</strong> en el centro y
        pequeños <strong>remolinos de esquina</strong> contrarrotantes en las esquinas inferiores. No hay fuentes, ni
        entrada ni salida: toda la estructura nace de la condición de no-deslizamiento contra una pared móvil. Es
        deceptivamente simple de enunciar y notoriamente difícil de resolver con precisión, por lo que es el problema de
        validación estándar desde hace décadas.
      </p>
      <p>
        Aquí lo resolvemos con un <strong>PINN de variables primitivas</strong> acoplado: una sola red predice las tres
        incógnitas <InlineMath tex={String.raw`(u,v,p)`} /> a la vez, y la pérdida impone los <em>tres</em> residuos de
        Navier-Stokes más todas las condiciones de borde. Lo validamos contra el <strong>benchmark de Ghia, Ghia y Shin
        (1982)</strong> a <InlineMath tex={String.raw`\mathrm{Re}=100`} />: los perfiles de velocidad sobre las líneas
        centrales que toda solución de cavidad debe reproducir.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> la cavidad cuadrada <InlineMath tex={String.raw`(x,y)\in(0,1)^2`} />, grilla del campo <InlineMath tex={String.raw`101\times101`} />.</li>
        <li><strong>Incógnitas (3 salidas acopladas):</strong> velocidad horizontal <InlineMath tex={String.raw`u(x,y)`} />, velocidad vertical <InlineMath tex={String.raw`v(x,y)`} /> y presión <InlineMath tex={String.raw`p(x,y)`} />.</li>
        <li><strong>Viscosidad / número de Reynolds:</strong> <InlineMath tex={String.raw`\nu=0.01`} />, de modo que <InlineMath tex={String.raw`\mathrm{Re}=UL/\nu=100`} /> (régimen laminar, un vórtice dominante).</li>
        <li><strong>Densidad:</strong> <InlineMath tex={String.raw`\rho=1`} /> (adimensional).</li>
        <li><strong>Tapa (en <InlineMath tex={String.raw`y=1`} />):</strong> un perfil <em>regularizado</em> <InlineMath tex={String.raw`u=16\,x^2(1-x)^2`} />, <InlineMath tex={String.raw`v=0`} />, que se anula en las dos esquinas superiores para domesticar la singularidad de velocidad.</li>
        <li><strong>Paredes restantes:</strong> no-deslizamiento <InlineMath tex={String.raw`u=v=0`} />.</li>
        <li><strong>Calibre de presión (gauge):</strong> la presión incompresible solo está definida salvo una constante; la fijamos con <InlineMath tex={String.raw`p(0,0)=0`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Las ecuaciones de Navier-Stokes incompresibles estacionarias acoplan la cantidad de movimiento (en
        <InlineMath tex={String.raw`x`} /> y en <InlineMath tex={String.raw`y`} />) con la incompresibilidad:
      </p>
      <Equation tex={String.raw`\begin{aligned}
u\,u_x + v\,u_y &= -\tfrac{1}{\rho}\,p_x + \nu\,(u_{xx}+u_{yy}),\\[2pt]
u\,v_x + v\,v_y &= -\tfrac{1}{\rho}\,p_y + \nu\,(v_{xx}+v_{yy}),\\[2pt]
u_x + v_y &= 0,
\end{aligned}`} />
      <p>
        en <InlineMath tex={String.raw`(0,1)^2`} />. Los dos primeros son el balance de momento (advección
        <InlineMath tex={String.raw`(\mathbf{u}\cdot\nabla)\mathbf{u}`} /> = gradiente de presión + difusión viscosa); el
        tercero, la <strong>continuidad</strong> <InlineMath tex={String.raw`\nabla\cdot\mathbf{u}=0`} />, es el que hace
        que el problema sea verdaderamente acoplado: la presión <InlineMath tex={String.raw`p`} /> no tiene ecuación
        propia, actúa como el <em>multiplicador de Lagrange</em> que impone divergencia nula. Las condiciones de borde
        son
      </p>
      <Equation tex={String.raw`\mathbf{u}\big|_{y=1}=\big(16\,x^2(1-x)^2,\,0\big),\qquad \mathbf{u}\big|_{\text{otras paredes}}=\mathbf 0,\qquad p(0,0)=0.`} />
      <p>
        El número de Reynolds <InlineMath tex={String.raw`\mathrm{Re}=UL/\nu`} /> es el único parámetro físico: mide la
        importancia relativa de la inercia frente a la viscosidad. A <InlineMath tex={String.raw`\mathrm{Re}=100`} /> el
        flujo es suave y estacionario, con un único vórtice grande; a Reynolds altos aparecen capas límite finas y
        remolinos secundarios cada vez más intensos. <strong>No existe solución en forma cerrada</strong> de la cavidad
        para ningún <InlineMath tex={String.raw`\mathrm{Re}`} />: el ancla de validación es una <em>referencia
        numérica</em>.
      </p>

      <h3>El método: PINN de variables primitivas acoplado, con pesos de pérdida</h3>
      <p>
        Este es el caso más exigente del catálogo: <strong>tres salidas acopladas, tres residuos</strong>, un calibre de
        presión y la singularidad de las esquinas de la tapa. La función de pérdida suma los tres residuos de la PDE más
        cada condición de borde:
      </p>
      <Equation tex={String.raw`\mathcal L = \underbrace{\|r_x\|^2+\|r_y\|^2+\|\nabla\cdot\mathbf u\|^2}_{\text{residuos PDE}}\;+\;\lambda\!\!\sum_{\text{borde}}\!\|\,\mathbf u-\mathbf u_{\text{bc}}\,\|^2\;+\;\lambda\,|p(0,0)|^2.`} />
      <ul>
        <li><strong>Salidas acopladas:</strong> una sola FNN <InlineMath tex={String.raw`\mathcal N_\theta(x,y)\to(u,v,p)`} /> aprende los tres campos a la vez; la diferenciación automática entrega <InlineMath tex={String.raw`u_x,v_y,p_x,\dots`} /> para armar los residuos.</li>
        <li><strong>Arquitectura y entrenamiento:</strong> una red <InlineMath tex={String.raw`2\to[64]\times5\to3`} /> tanh/Glorot, optimizada con Adam (20 000 pasos, tasa de aprendizaje <InlineMath tex={String.raw`10^{-3}`} />) y luego un pulido L-BFGS, sobre 2601 puntos de colocación interiores + 400 de borde.</li>
        <li><strong>Tapa regularizada:</strong> el perfil <InlineMath tex={String.raw`16\,x^2(1-x)^2`} /> se anula con derivada nula en las esquinas, eliminando la discontinuidad <InlineMath tex={String.raw`u=1`} /> vs <InlineMath tex={String.raw`u=0`} /> que de otro modo arruinaría la convergencia.</li>
        <li><strong>Pesos de pérdida (BC y gauge x10):</strong> las condiciones de borde y el calibre de presión se sobre-ponderan <InlineMath tex={String.raw`\lambda=10`} /> respecto a los residuos interiores, con el vector exacto <InlineMath tex={String.raw`[1,1,1,10,10,10,10,10]`} /> (los tres residuos en 1; tapa-<InlineMath tex={String.raw`u`} />, tapa-<InlineMath tex={String.raw`v`} />, pared-<InlineMath tex={String.raw`u`} />, pared-<InlineMath tex={String.raw`v`} /> y el calibre de presión en 10), para que la red no los ignore: el truco práctico clave en PINNs multi-salida.</li>
        <li><strong>Pulido L-BFGS:</strong> tras Adam, un paso de L-BFGS afina el mínimo.</li>
      </ul>
      <p>
        Cabe ser <strong>honesto</strong> sobre el techo de precisión. En el lane CPU este PINN suave de variables
        primitivas obtiene un relativo-L2 de <strong>0.1675 (~17%)</strong> frente a las líneas centrales de Ghia
        (<InlineMath tex={String.raw`u`} /> central <strong>0.117</strong>,{" "}
        <InlineMath tex={String.raw`v`} /> central <strong>0.218</strong>). La red captura el vórtice primario y la
        estructura cualitativa de remolinos de esquina, pero el ajuste de las líneas centrales, <strong>en especial la
        velocidad vertical <InlineMath tex={String.raw`v`} /> en 0.22, es grueso</strong>, porque el lane CPU no puede
        costear la cantidad de iteraciones que una cavidad nítida a{" "}
        <InlineMath tex={String.raw`\mathrm{Re}=100`} /> necesita. Esto es <strong>fidelidad reducida, no una solución
        de precisión de publicación</strong>, y la brecha con Ghia se reporta como el número principal, no como una nota
        al pie. Formulaciones más fuertes (función de corriente-vorticidad, restricciones duras de incompresibilidad, o
        una pasada en GPU sobre <strong>PhysicsNeMo</strong>) la reducen; se <em>citan</em> como horizonte, no se
        reclaman aquí.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> cavidad 2D, estacionaria, incompresible, a <InlineMath tex={String.raw`\mathrm{Re}=100`} />
        fijo, con tapa regularizada y validación contra las líneas centrales de Ghia 1982. El campo se etiqueta
        <strong> sintético-ilustrativo</strong>: es una solución numérica de fidelidad reducida (lane CPU), no un dato
        experimental ni una referencia de fidelidad espectral.
      </p>
      <p>
        <strong>Fuera de alcance: y por qué este caso es de variante única.</strong> Sería tentador barrer el número de
        Reynolds como si fuera un parámetro de red (al estilo de los casos paramétricos del catálogo). No se hace, y la
        razón es de <em>honestidad</em>: la cavidad <strong>no tiene solución analítica para ningún
        <InlineMath tex={String.raw`\mathrm{Re}`} /></strong>, así que no existe un ancla en forma cerrada que pueda
        probarse por sustitución para toda una familia. La única referencia confiable es la tabla de Ghia, definida en un
        conjunto <em>discreto</em> de Reynolds (100, 400, 1000, …), y cada Reynolds más alto requiere su propia red y se
        vuelve marcadamente más difícil de entrenar en CPU. Inventar un barrido continuo de Reynolds sería fabricar
        regímenes sin ancla verificable: exactamente el modo de fallo "de juguete" que evitamos. Por eso este caso se
        publica como un <strong>benchmark único, riguroso, a <InlineMath tex={String.raw`\mathrm{Re}=100`} /></strong>.
        También queda fuera: turbulencia (3D / Reynolds altos), flujo transitorio (este es estacionario), y geometrías
        distintas del cuadrado.
      </p>

      <h3>Qué muestra el benchmark</h3>
      <p>
        Desde reposo, el arrastre de la tapa organiza un <strong>vórtice primario</strong> que llena la cavidad, con su
        centro desplazado hacia la esquina superior derecha (en la dirección del deslizamiento) a
        <InlineMath tex={String.raw`\mathrm{Re}=100`} />. En las dos esquinas inferiores aparecen <strong>remolinos
        secundarios</strong> contrarrotantes, débiles pero presentes. El campo de presión muestra un mínimo en el núcleo
        del vórtice y máximos donde la corriente impacta las paredes. La validación cuantitativa es directa: el perfil
        <InlineMath tex={String.raw`u(0.5,y)`} /> sobre la línea central vertical y <InlineMath tex={String.raw`v(x,0.5)`} />
        sobre la horizontal deben caer sobre los puntos digitalizados de Ghia 1982; la métrica reportada es el
        relativo-L2 contra esos puntos, <strong>0.1675</strong> (promedio de ambas líneas centrales), y el modelo ONNX
        exportado coincide con la red entrenada con una paridad máxima absoluta de{" "}
        <InlineMath tex={String.raw`1.237\times10^{-6}`} />.
      </p>

      <h3>Cómo leer y usar la viz</h3>
      <p>
        El <strong>heatmap</strong> muestra el campo seleccionado sobre la cavidad <InlineMath tex={String.raw`(x,y)`} />.
        Cambia entre <InlineMath tex={String.raw`u`} /> (rápido positivo bajo la tapa, negativo en el retorno inferior),
        <InlineMath tex={String.raw`v`} /> (sube por la pared derecha, baja por la izquierda) y la presión
        <InlineMath tex={String.raw`p`} /> (núcleo de baja presión = centro del vórtice). Al pasar el cursor para leer el
        valor exacto en cualquier punto. Usa el <strong>corte de línea</strong> sobre <InlineMath tex={String.raw`x=0.5`} />
        para ver el perfil <InlineMath tex={String.raw`u(0.5,y)`} /> en forma de S: el signo cambia donde la corriente
        de retorno se cruza con el flujo arrastrado por la tapa: y compáralo mentalmente con la curva de Ghia. Como es
        un benchmark de parámetro fijo, el tab <strong>Live</strong> re-evalúa la red entrenada (la misma física a
        <InlineMath tex={String.raw`\mathrm{Re}=100`} />), sin deslizador de parámetro.
      </p>
    </>
  ): (
    <>
      <h2>The problem: the lid-driven cavity: steady incompressible Navier-Stokes</h2>
      <p>
        <strong>The problem.</strong> The <em>lid-driven cavity</em> is the canonical test bed of incompressible fluid
        dynamics: a square of fluid <InlineMath tex={String.raw`(0,1)^2`} /> whose three walls are fixed and whose
        <strong> top lid</strong> slides horizontally. The viscous drag from the lid sets the whole fluid spinning,
        forming a <strong>primary vortex</strong> at the center and small counter-rotating <strong>corner eddies</strong>
        in the lower corners. There are no sources, no inflow and no outflow: the entire structure is born from the
        no-slip condition against a moving wall. It is deceptively simple to state and notoriously hard to solve
        accurately, which is exactly why it has been the standard validation problem for decades.
      </p>
      <p>
        Here we solve it with a coupled <strong>primitive-variable PINN</strong>: a single network predicts the three
        unknowns <InlineMath tex={String.raw`(u,v,p)`} /> at once, and the loss enforces <em>all three</em> Navier-Stokes
        residuals plus every boundary condition. We validate against the <strong>Ghia, Ghia &amp; Shin (1982)</strong>
        benchmark at <InlineMath tex={String.raw`\mathrm{Re}=100`} />: the centerline velocity profiles that any cavity
        solution must reproduce.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> the square cavity <InlineMath tex={String.raw`(x,y)\in(0,1)^2`} />, a <InlineMath tex={String.raw`101\times101`} /> field grid.</li>
        <li><strong>Unknowns (3 coupled outputs):</strong> horizontal velocity <InlineMath tex={String.raw`u(x,y)`} />, vertical velocity <InlineMath tex={String.raw`v(x,y)`} /> and pressure <InlineMath tex={String.raw`p(x,y)`} />.</li>
        <li><strong>Viscosity / Reynolds number:</strong> <InlineMath tex={String.raw`\nu=0.01`} />, so that <InlineMath tex={String.raw`\mathrm{Re}=UL/\nu=100`} /> (laminar regime, a single dominant vortex).</li>
        <li><strong>Density:</strong> <InlineMath tex={String.raw`\rho=1`} /> (non-dimensional).</li>
        <li><strong>Lid (at <InlineMath tex={String.raw`y=1`} />):</strong> a <em>regularized</em> profile <InlineMath tex={String.raw`u=16\,x^2(1-x)^2`} />, <InlineMath tex={String.raw`v=0`} />, vanishing at the two top corners to tame the velocity singularity.</li>
        <li><strong>Remaining walls:</strong> no-slip <InlineMath tex={String.raw`u=v=0`} />.</li>
        <li><strong>Pressure gauge:</strong> incompressible pressure is defined only up to a constant; we pin it with <InlineMath tex={String.raw`p(0,0)=0`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        The steady incompressible Navier-Stokes equations couple the momentum balance (in <InlineMath tex={String.raw`x`} />
        and in <InlineMath tex={String.raw`y`} />) with incompressibility:
      </p>
      <Equation tex={String.raw`\begin{aligned}
u\,u_x + v\,u_y &= -\tfrac{1}{\rho}\,p_x + \nu\,(u_{xx}+u_{yy}),\\[2pt]
u\,v_x + v\,v_y &= -\tfrac{1}{\rho}\,p_y + \nu\,(v_{xx}+v_{yy}),\\[2pt]
u_x + v_y &= 0,
\end{aligned}`} />
      <p>
        on <InlineMath tex={String.raw`(0,1)^2`} />. The first two are the momentum balance (advection
        <InlineMath tex={String.raw`(\mathbf{u}\cdot\nabla)\mathbf{u}`} /> = pressure gradient + viscous diffusion); the
        third, <strong>continuity</strong> <InlineMath tex={String.raw`\nabla\cdot\mathbf{u}=0`} />, is what makes the
        problem truly coupled: the pressure <InlineMath tex={String.raw`p`} /> has no equation of its own: it acts as the
        <em> Lagrange multiplier</em> that enforces zero divergence. The boundary conditions are
      </p>
      <Equation tex={String.raw`\mathbf{u}\big|_{y=1}=\big(16\,x^2(1-x)^2,\,0\big),\qquad \mathbf{u}\big|_{\text{other walls}}=\mathbf 0,\qquad p(0,0)=0.`} />
      <p>
        The Reynolds number <InlineMath tex={String.raw`\mathrm{Re}=UL/\nu`} /> is the single physical parameter: it
        measures inertia relative to viscosity. At <InlineMath tex={String.raw`\mathrm{Re}=100`} /> the flow is smooth and
        steady, with one large vortex; at higher Reynolds, thin boundary layers and increasingly intense secondary eddies
        appear. <strong>There is no closed-form solution</strong> of the cavity for any
        <InlineMath tex={String.raw`\mathrm{Re}`} />: the validation anchor is a <em>numerical reference</em>.
      </p>

      <h3>The method: coupled primitive-variable PINN with loss weighting</h3>
      <p>
        This is the most demanding case in the catalogue: <strong>three coupled outputs, three residuals</strong>, a
        pressure gauge, and the lid-corner singularity. The loss sums the three PDE residuals plus every boundary
        condition:
      </p>
      <Equation tex={String.raw`\mathcal L = \underbrace{\|r_x\|^2+\|r_y\|^2+\|\nabla\cdot\mathbf u\|^2}_{\text{PDE residuals}}\;+\;\lambda\!\!\sum_{\text{boundary}}\!\|\,\mathbf u-\mathbf u_{\text{bc}}\,\|^2\;+\;\lambda\,|p(0,0)|^2.`} />
      <ul>
        <li><strong>Coupled outputs:</strong> a single FNN <InlineMath tex={String.raw`\mathcal N_\theta(x,y)\to(u,v,p)`} /> learns all three fields at once; automatic differentiation supplies <InlineMath tex={String.raw`u_x,v_y,p_x,\dots`} /> to assemble the residuals.</li>
        <li><strong>Architecture and training:</strong> a <InlineMath tex={String.raw`2\to[64]\times5\to3`} /> tanh/Glorot network, optimized with Adam (20,000 steps, learning rate <InlineMath tex={String.raw`10^{-3}`} />) then an L-BFGS polish, over 2601 interior + 400 boundary collocation points.</li>
        <li><strong>Regularized lid:</strong> the profile <InlineMath tex={String.raw`16\,x^2(1-x)^2`} /> vanishes with zero derivative at the corners, removing the <InlineMath tex={String.raw`u=1`} /> vs <InlineMath tex={String.raw`u=0`} /> discontinuity that would otherwise wreck convergence.</li>
        <li><strong>Loss weighting (BC and gauge x10):</strong> the boundary conditions and the pressure gauge are up-weighted <InlineMath tex={String.raw`\lambda=10`} /> relative to the interior residuals, the exact vector being <InlineMath tex={String.raw`[1,1,1,10,10,10,10,10]`} /> (three residuals at 1; lid-<InlineMath tex={String.raw`u`} />, lid-<InlineMath tex={String.raw`v`} />, wall-<InlineMath tex={String.raw`u`} />, wall-<InlineMath tex={String.raw`v`} /> and the pressure gauge at 10), so the network does not ignore them: the key practical trick in multi-output PINNs.</li>
        <li><strong>L-BFGS polish:</strong> after Adam, an L-BFGS step sharpens the minimum.</li>
      </ul>
      <p>
        It is worth being <strong>honest</strong> about the accuracy ceiling. On the CPU lane this soft
        primitive-variable PINN scores a relative-L2 of <strong>0.1675 (~17%)</strong> against the Ghia centerlines
        (<InlineMath tex={String.raw`u`} />-centerline <strong>0.117</strong>,{" "}
        <InlineMath tex={String.raw`v`} />-centerline <strong>0.218</strong>). The network captures the primary vortex
        and the qualitative corner-eddy structure, but the centerline match, <strong>especially the vertical velocity{" "}
        <InlineMath tex={String.raw`v`} /> at 0.22, is coarse</strong>, because the CPU lane cannot afford the iteration
        count a sharp <InlineMath tex={String.raw`\mathrm{Re}=100`} /> cavity needs. This is <strong>reduced fidelity,
        not a publication-grade cavity solve</strong>, and the gap to Ghia is reported as the headline number, not a
        footnote. Stronger formulations (stream-function-vorticity, hard incompressibility constraints, or a GPU pass
        on <strong>PhysicsNeMo</strong>) would tighten it; they are <em>cited</em> as the horizon, not claimed here.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> a 2D, steady, incompressible cavity at fixed <InlineMath tex={String.raw`\mathrm{Re}=100`} />,
        with a regularized lid and validation against the Ghia 1982 centerlines. The field is labeled
        <strong> synthetic-illustrative</strong>: it is a reduced-fidelity numerical solution (CPU lane), not experimental
        data nor a spectral-fidelity reference.
      </p>
      <p>
        <strong>Out of scope: and why this case is single-variant.</strong> It would be tempting to sweep the Reynolds
        number as if it were a network parameter (in the style of the catalogue's parametric cases). We do not, and the
        reason is one of <em>honesty</em>: the cavity <strong>has no analytic solution for any
        <InlineMath tex={String.raw`\mathrm{Re}`} /></strong>, so there is no closed-form anchor that can be proven by
        substitution for an entire family. The only reliable reference is the Ghia table, defined on a <em>discrete</em>
        set of Reynolds numbers (100, 400, 1000, …), and each higher Reynolds needs its own network and becomes markedly
        harder to train on CPU. Inventing a continuous Reynolds sweep would be fabricating regimes with no verifiable
        anchor: exactly the "toy" failure mode we avoid. That is why this case ships as a <strong>single, rigorous
        benchmark at <InlineMath tex={String.raw`\mathrm{Re}=100`} /></strong>. Also out of scope: turbulence (3D / high
        Reynolds), transient flow (this is steady), and geometries other than the square.
      </p>

      <h3>What the benchmark shows</h3>
      <p>
        From rest, the lid drag organizes a <strong>primary vortex</strong> that fills the cavity, its center shifted
        toward the upper-right corner (the direction of the slide) at <InlineMath tex={String.raw`\mathrm{Re}=100`} />. In
        the two lower corners, weak but present counter-rotating <strong>secondary eddies</strong> appear. The pressure
        field shows a minimum at the vortex core and maxima where the stream impinges on the walls. Quantitative
        validation is direct: the profile <InlineMath tex={String.raw`u(0.5,y)`} /> along the vertical centerline and
        <InlineMath tex={String.raw`v(x,0.5)`} /> along the horizontal one must fall on the digitized points of Ghia 1982;
        the reported metric is the relative-L2 against those points, <strong>0.1675</strong> (mean of the two
        centerlines), and the exported ONNX model matches the trained network to{" "}
        <InlineMath tex={String.raw`1.237\times10^{-6}`} /> max-abs parity.
      </p>

      <h3>How to read &amp; use the viz</h3>
      <p>
        The <strong>heatmap</strong> shows the selected field over the cavity <InlineMath tex={String.raw`(x,y)`} />.
        Switch between <InlineMath tex={String.raw`u`} /> (fast positive under the lid, negative on the lower return),
        <InlineMath tex={String.raw`v`} /> (rising along the right wall, falling on the left) and pressure
        <InlineMath tex={String.raw`p`} /> (low-pressure core = vortex center). Hover to read the exact value at any point.
        Use the <strong>line-cut</strong> at <InlineMath tex={String.raw`x=0.5`} /> to see the S-shaped profile
        <InlineMath tex={String.raw`u(0.5,y)`} />: the sign changes where the return flow meets the lid-driven stream: 
        and compare it mentally with the Ghia curve. Since it is a fixed-parameter benchmark, the <strong>Live</strong>
        tab re-evaluates the trained network (the same physics at <InlineMath tex={String.raw`\mathrm{Re}=100`} />), with
        no parameter slider.
      </p>
    </>
  );
}
