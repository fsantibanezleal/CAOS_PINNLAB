import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-poisson2d, mirroring CAOS_SIMLAB's S0XDesc structure:
 *  The problem to Components & variables to Formalization to Scope & assumptions to What each variant shows to 
 *  How to read & use the viz. */
export function PoissonContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: la ecuación de Poisson con un modo de fuente ajustable</h2>
      <p>
        <strong>El problema.</strong> La ecuación de Poisson <InlineMath tex={String.raw`-\nabla^2 u = f`} /> es la EDP
        elíptica canónica: gobierna el potencial electrostático de una densidad de carga, la deflexión de una membrana
        bajo carga, la presión en flujo incompresible y la difusión en estado estacionario. Aquí la resolvemos en el
        cuadrado unitario <InlineMath tex={String.raw`(0,1)^2`} /> con condición de Dirichlet homogénea
        (<InlineMath tex={String.raw`u=0`} /> en el borde) y una <strong>fuente paramétrica</strong> cuyo modo
        espacial <InlineMath tex={String.raw`k`} /> se puede variar de forma continua. Una sola red entrenada
        <em> aprende toda la familia</em> <InlineMath tex={String.raw`u(x,y;k)`} />: en el tab <strong>Live</strong>,
        mover el deslizador de <InlineMath tex={String.raw`k`} /> re-evalúa el campo en el navegador sin reentrenar.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> el cuadrado unitario <InlineMath tex={String.raw`\Omega=(0,1)^2`} />, con la grilla del campo de <InlineMath tex={String.raw`101\times101`} />.</li>
        <li><strong>Incógnita:</strong> el campo escalar <InlineMath tex={String.raw`u(x,y)`} /> (potencial / deflexión / presión), nulo en todo el borde.</li>
        <li><strong>Parámetro de control:</strong> el <em>modo de fuente</em> <InlineMath tex={String.raw`k\in[1,3]`} />: un input de la red, no una constante. Controla cuántos lóbulos tiene el campo.</li>
        <li><strong>Fuente:</strong> <InlineMath tex={String.raw`f(x,y;k)=-\nabla^2 u^*`} />, la forzante de la solución manufacturada de abajo (forma cerrada).</li>
        <li><strong>Métrica:</strong> el error relativo-L2 del campo de la PINN contra la solución exacta, por régimen.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Usamos el <strong>método de soluciones manufacturadas</strong> (MMS) para tener una referencia exacta válida
        para CUALQUIER <InlineMath tex={String.raw`k`} /> continuo. Elegimos una solución que se anula en el borde por
        construcción y derivamos su forzante:
      </p>
      <Equation tex={String.raw`u^*(x,y;k)=g(x;k)\,g(y;k),\qquad g(t;k)=t(1-t)\,\sin(k\pi t),`} />
      <p>
        El factor <InlineMath tex={String.raw`t(1-t)`} /> garantiza <InlineMath tex={String.raw`g(0)=g(1)=0`} />, de
        modo que <InlineMath tex={String.raw`u^*=0`} /> en todo <InlineMath tex={String.raw`\partial\Omega`} /> para
        todo <InlineMath tex={String.raw`k`} /> (entero o no). Como <InlineMath tex={String.raw`u^*`} /> es separable,
        su laplaciano y la forzante son cerrados:
      </p>
      <Equation tex={String.raw`f=-\nabla^2 u^*=-\big(g''(x)\,g(y)+g(x)\,g''(y)\big).`} />
      <p>
        La red <InlineMath tex={String.raw`u_\theta(x,y,k)`} /> minimiza el residual de la EDP en puntos de colocación
        sobre <InlineMath tex={String.raw`(0,1)^2\times[1,3]`} />. La condición de borde se impone <strong>de forma
        exacta</strong> por una <em>restricción dura</em> (output transform), sin término de pérdida de borde:
      </p>
      <Equation tex={String.raw`u_\theta=x(1-x)\,y(1-y)\,\mathcal{N}_\theta(x,y,k)\ \Rightarrow\ u_\theta|_{\partial\Omega}=0\ \text{exactamente.}`} />

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> Poisson lineal en 2-D, Dirichlet homogéneo, fuente sinusoidal de modo continuo
        <InlineMath tex={String.raw`k\in[1,3]`} />, una red FNN tanh entrenada Adam to L-BFGS. La restricción dura
        sobrevive intacta a la exportación a ONNX (es álgebra tensorial pura), por eso el tab Live re-evalúa el campo
        exacto en el navegador; la salida ONNX coincide con la red offline a 1.2e-7 (máx abs). <strong>Fuera de alcance:</strong> coeficientes variables/anisótropos, bordes no
        rectangulares, Neumann/Robin, y modos <InlineMath tex={String.raw`k>3`} /> (el sesgo espectral de una red suave
        encarece las frecuencias altas: el régimen <InlineMath tex={String.raw`k=3`} /> ya es el test de estrés).
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> El barrido de modo recorre el sesgo espectral: <em>k=1</em> es el
        modo fundamental (un único lóbulo); <em>k=1.5</em> empieza a subdividir; <em>k=2</em> da un patrón de 2×2
        lóbulos; <em>k=2.25</em> y <em>k=2.5</em> son modos no enteros (siguen anulándose en el borde por el MMS) que
        afinan la oscilación; <em>k=3</em> es el patrón 3×3, el régimen más exigente para una red suave. Que la L2 relativa se
        mantenga ≤ 0.16% en los seis demuestra que UNA red cubre toda la familia.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> pinta <InlineMath tex={String.raw`u`} />
        en viridis; pasa el cursor para leer el valor exacto en cualquier punto (aparecen la cruz, las coordenadas y
        <InlineMath tex={String.raw`u`} />) y mira los dos <strong>perfiles de corte</strong> (u a lo largo de x y de y
        por el cursor). Los <strong>chips de régimen</strong> cargan cada modo pre-precalculado; el tab <strong>Charts</strong>
        compara el L2 de todos (clic para cargar). En el tab <strong>Live</strong>, mueve el deslizador de
        <InlineMath tex={String.raw`k`} /> y el campo se <em>recalcula en vivo</em> en tu navegador vía onnxruntime-web
      : la misma red entrenada offline, ahora paramétrica.
      </p>
    </>
  ): (
    <>
      <h2>The problem: the Poisson equation with a tunable source mode</h2>
      <p>
        <strong>The problem.</strong> The Poisson equation <InlineMath tex={String.raw`-\nabla^2 u = f`} /> is the
        canonical elliptic PDE: it governs the electrostatic potential of a charge density, the deflection of a loaded
        membrane, the pressure in incompressible flow, and steady-state diffusion. Here we solve it on the unit square
        <InlineMath tex={String.raw`(0,1)^2`} /> with a homogeneous Dirichlet condition (<InlineMath tex={String.raw`u=0`} />
        on the boundary) and a <strong>parametric source</strong> whose spatial mode <InlineMath tex={String.raw`k`} />
        can be varied continuously. A single trained network <em>learns the whole family</em>
        <InlineMath tex={String.raw`u(x,y;k)`} />: in the <strong>Live</strong> tab, moving the
        <InlineMath tex={String.raw`k`} /> slider re-evaluates the field in the browser with no retraining.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> the unit square <InlineMath tex={String.raw`\Omega=(0,1)^2`} />, with a <InlineMath tex={String.raw`101\times101`} /> field grid.</li>
        <li><strong>Unknown:</strong> the scalar field <InlineMath tex={String.raw`u(x,y)`} /> (potential / deflection / pressure), zero on the whole boundary.</li>
        <li><strong>Control parameter:</strong> the <em>source mode</em> <InlineMath tex={String.raw`k\in[1,3]`} />: a network input, not a constant. It sets how many lobes the field has.</li>
        <li><strong>Source:</strong> <InlineMath tex={String.raw`f(x,y;k)=-\nabla^2 u^*`} />, the forcing of the manufactured solution below (closed form).</li>
        <li><strong>Metric:</strong> the relative-L2 error of the PINN field vs the exact solution, per regime.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        We use the <strong>method of manufactured solutions</strong> (MMS) to obtain an exact reference valid for ANY
        continuous <InlineMath tex={String.raw`k`} />. We pick a solution that vanishes on the boundary by construction
        and derive its forcing:
      </p>
      <Equation tex={String.raw`u^*(x,y;k)=g(x;k)\,g(y;k),\qquad g(t;k)=t(1-t)\,\sin(k\pi t),`} />
      <p>
        The factor <InlineMath tex={String.raw`t(1-t)`} /> guarantees <InlineMath tex={String.raw`g(0)=g(1)=0`} />, so
        <InlineMath tex={String.raw`u^*=0`} /> on all of <InlineMath tex={String.raw`\partial\Omega`} /> for every
        <InlineMath tex={String.raw`k`} /> (integer or not). Because <InlineMath tex={String.raw`u^*`} /> is separable,
        its Laplacian and the forcing are closed-form:
      </p>
      <Equation tex={String.raw`f=-\nabla^2 u^*=-\big(g''(x)\,g(y)+g(x)\,g''(y)\big).`} />
      <p>
        The network <InlineMath tex={String.raw`u_\theta(x,y,k)`} /> minimises the PDE residual at collocation points
        over <InlineMath tex={String.raw`(0,1)^2\times[1,3]`} />. The boundary condition is imposed <strong>exactly</strong>
        by a <em>hard constraint</em> (output transform), with no boundary loss term:
      </p>
      <Equation tex={String.raw`u_\theta=x(1-x)\,y(1-y)\,\mathcal{N}_\theta(x,y,k)\ \Rightarrow\ u_\theta|_{\partial\Omega}=0\ \text{exactly.}`} />

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> linear 2-D Poisson, homogeneous Dirichlet, a sinusoidal source of continuous mode
        <InlineMath tex={String.raw`k\in[1,3]`} />, a tanh FNN trained Adam to L-BFGS. The hard constraint survives the
        ONNX export intact (it is pure tensor algebra), which is why the Live tab re-evaluates the exact field in the
        browser; the ONNX output matches the offline net to 1.2e-7 (max abs). <strong>Out of scope:</strong> variable/anisotropic coefficients, non-rectangular domains, Neumann/Robin
        conditions, and modes <InlineMath tex={String.raw`k>3`} /> (a smooth network's spectral bias makes high
        frequencies costly: the <InlineMath tex={String.raw`k=3`} /> regime is already the stress test).
      </p>

      <p>
        <strong>What each variant shows.</strong> The mode sweep walks the spectral bias: <em>k=1</em> is the
        fundamental mode (a single lobe); <em>k=1.5</em> begins to subdivide; <em>k=2</em> gives a 2×2 lobe pattern;
        <em>k=2.25</em> and <em>k=2.5</em> are off-integer modes (still boundary-vanishing thanks to the MMS) that
        sharpen the oscillation; <em>k=3</em> is the 3×3 pattern, the hardest regime for a smooth net. Keeping the relative-L2
        ≤ 0.16% across all six shows ONE network covers the whole family.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> paints <InlineMath tex={String.raw`u`} />
        in viridis; hover to read the exact value at any point (a crosshair, the coordinates and
        <InlineMath tex={String.raw`u`} /> appear) and watch the two <strong>line-cut profiles</strong> (u along x and
        along y through the cursor). The <strong>regime chips</strong> load each pre-baked mode; the <strong>Charts</strong>
        tab compares everyone's L2 (click to load). In the <strong>Live</strong> tab, move the
        <InlineMath tex={String.raw`k`} /> slider and the field <em>recomputes live</em> in your browser via
        onnxruntime-web: the same network trained offline, now parametric.
      </p>
    </>
  );
}
