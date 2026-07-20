import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for ind-helmholtz (2D high-wavenumber Helmholtz, Fourier-feature spectral-bias benchmark). */
export function HelmholtzContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>una onda estacionaria de alta frecuencia: la ecuación de Helmholtz 2D</h2>
      <p>
        La ecuación de Helmholtz
        <InlineMath tex={String.raw`\nabla^2 u + k_0^2\,u = -f`} /> es la forma en el <em>dominio de la frecuencia</em>
        de la ecuación de onda: describe el estado estacionario de un campo que vibra a una sola frecuencia (acústica en
        una sala, electromagnetismo de microondas, vibración de una membrana). El número de onda
        <InlineMath tex={String.raw`k_0`} /> fija <strong>cuántas oscilaciones</strong> caben en el dominio. Con
        <InlineMath tex={String.raw`k_0=2\pi n`} /> y <InlineMath tex={String.raw`n=3`} />, la solución es un patrón
        cuadriculado de máximos y mínimos alternados: y precisamente esa <strong>alta frecuencia espacial</strong> es lo
        que hace fracasar a un PINN ingenuo. Este caso muestra la técnica que lo resuelve: un embedding de
        <strong> características de Fourier</strong>.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> el cuadrado <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, grilla del campo <InlineMath tex={String.raw`121\times121`} />.</li>
        <li><strong>Incógnita:</strong> la amplitud estacionaria <InlineMath tex={String.raw`u(x,y)`} /> (p. ej. presión acústica o desplazamiento de una membrana).</li>
        <li><strong>Número de onda:</strong> <InlineMath tex={String.raw`k_0=2\pi n`} /> con <InlineMath tex={String.raw`n=3`} /> (<InlineMath tex={String.raw`k_0=6\pi\approx18.8`} />): tres longitudes de onda completas por lado.</li>
        <li><strong>Fuente:</strong> <InlineMath tex={String.raw`f=k_0^2\sin(k_0 x)\sin(k_0 y)`} />: elegida (método de soluciones manufacturadas) para que la solución exacta sea limpia.</li>
        <li><strong>Frontera:</strong> Dirichlet homogénea <InlineMath tex={String.raw`u=0`} /> en todo <InlineMath tex={String.raw`\partial\Omega`} /> (un dominio con bordes fijos: un parche de membrana sujeto).</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Usamos el <strong>método de soluciones manufacturadas</strong> (MMS): se fija a propósito la solución objetivo y
        se deriva la fuente <InlineMath tex={String.raw`f`} /> que la produce, de modo que la
        <strong> ancla de validación</strong> es exacta y el error relativo-L2 reportado es el error <em>verdadero</em>
        del PINN:
      </p>
      <Equation tex={String.raw`u^*(x,y)=\sin(k_0 x)\,\sin(k_0 y),\qquad k_0=2\pi n.`} />
      <p>
        La verificación es directa. Como
        <InlineMath tex={String.raw`\partial_{xx}u^*=-k_0^2 u^*`} /> y
        <InlineMath tex={String.raw`\partial_{yy}u^*=-k_0^2 u^*`} />, se tiene
        <InlineMath tex={String.raw`\nabla^2 u^*=-2k_0^2 u^*`} />; sustituyendo,
        <InlineMath tex={String.raw`\nabla^2 u^* + k_0^2 u^* = -k_0^2 u^* = -f`} /> con
        <InlineMath tex={String.raw`f=k_0^2 u^*`} />: el residual es <strong>cero</strong>. Además
        <InlineMath tex={String.raw`u^*`} /> se anula en los cuatro bordes (porque <InlineMath tex={String.raw`\sin(k_0\cdot 0)=\sin(k_0\cdot 1)=0`} /> al ser <InlineMath tex={String.raw`k_0=2\pi n`} /> con <InlineMath tex={String.raw`n`} /> entero), así que satisface exactamente la BC de Dirichlet. La PINN
        <InlineMath tex={String.raw`u_\theta(x,y)`} /> minimiza el residual de Helmholtz en puntos de colocación, con la
        BC impuesta de forma <strong>blanda y ponderada</strong> (peso <InlineMath tex={String.raw`100\times`} />): la
        receta robusta para soluciones oscilatorias, donde una restricción dura multiplicativa lucharía contra la
        oscilación cerca del borde.
      </p>

      <h3>El método: por qué el PINN ingenuo falla, y las características de Fourier</h3>
      <p>
        Una MLP <InlineMath tex={String.raw`\tanh`} /> sufre <strong>sesgo espectral</strong> (Rahaman et al. 2019): las
        redes aprenden primero las componentes de <em>baja</em> frecuencia y tienen dificultad enorme para representar
        las de <em>alta</em> frecuencia. Con <InlineMath tex={String.raw`k_0=6\pi`} /> el objetivo es puramente de alta
        frecuencia, así que una MLP simple se estanca en una meseta de error grande. La solución es inyectar las
        frecuencias <em>antes</em> de la red, con un <strong>mapa de características de Fourier</strong> aleatorias
        (Tancik et al. 2020; Wang-Wang-Perdikaris 2021, multi-escala):
      </p>
      <Equation tex={String.raw`\gamma(\mathbf{x})=\big[\sin(2\pi B\mathbf{x}),\ \cos(2\pi B\mathbf{x})\big],\qquad B_{ij}\sim\mathcal{N}(0,\sigma^2),`} />
      <p>
        con la matriz <InlineMath tex={String.raw`B`} /> <strong>congelada</strong> (sin entrenar, sembrada) y dos
        escalas <InlineMath tex={String.raw`\sigma\in\{1,n\}`} /> que cubren tanto la variación lenta como la frecuencia
        objetivo. Las dos escalas apiladas dan un embedding de 256 dimensiones que alimenta una FNN tanh de
        <InlineMath tex={String.raw`4\times128`} />, optimizada con Adam (25 000 pasos) y luego L-BFGS, con colocación a ~12
        puntos por longitud de onda por eje. Como <InlineMath tex={String.raw`\gamma`} /> son puras operaciones tensoriales, se traza limpiamente a
        ONNX, así que el tab <strong>Live</strong> reproduce la misma red en el navegador. El embedding convierte el
        problema de alta frecuencia en uno que la MLP <em>sí</em> puede ajustar.
      </p>

      <p>
        <strong>Resultado y honestidad.</strong> Medido con semilla 42, la PINN con características de Fourier alcanza un
        relativo-L2 de 0.1026 (~10%) contra el campo analítico, con un error absoluto máximo de 0.158; la exportación
        ONNX coincide con la red entrenada hasta 4.77e-07 (abs máx) y se ejecuta en <strong>Live</strong> a 340 KB, 18.5 ms de
        inferencia. El ~10% está honestamente limitado por CPU: el mapa de Fourier levanta la meseta de sesgo espectral
        que de otro modo dejaría a una MLP simple un orden de magnitud peor, pero con
        <InlineMath tex={String.raw`k_0=6\pi`} /> en el carril de CPU el patrón estacionario se resuelve a ~10%, no a
        1e-3. Entrenar en GPU más un recocido de frecuencias lo apretaría aún más; esto no está maquillado.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> Helmholtz 2D con número de onda fijo <InlineMath tex={String.raw`k_0=6\pi`} />
        (<InlineMath tex={String.raw`n=3`} />), fuente MMS, BC de Dirichlet homogénea, en CPU. <strong>Fuera de
        alcance:</strong> un <em>barrido</em> continuo del número de onda: una sola red con Fourier features puede
        cubrir una banda estrecha <InlineMath tex={String.raw`n\in[2,4]`} /> pero la precisión se degrada hacia el
        extremo agudo (el sesgo espectral reaparece banda arriba), por lo que aquí se publica como un
        <strong> benchmark de número de onda fijo</strong> que muestra la técnica en su punto más exigente, en vez de un
        barrido con error desparejo. También quedan fuera la <em>resonancia</em> (cuando <InlineMath tex={String.raw`k_0`} /> coincide con un valor propio del Laplaciano, el problema homogéneo deja de tener solución única), los
        dominios complejos / dispersión con condición de radiación de Sommerfeld, y el medio heterogéneo
        <InlineMath tex={String.raw`k_0(x,y)`} />.
      </p>

      <p>
        <strong>Qué muestra el benchmark.</strong> El campo es un patrón de <strong>onda estacionaria</strong>: una
        cuadrícula regular de lóbulos que alternan entre <InlineMath tex={String.raw`+1`} /> y
        <InlineMath tex={String.raw`-1`} /> (tres longitudes de onda completas por lado, un tablero de 6x6 con 36 lóbulos alternados), con nodos
        (<InlineMath tex={String.raw`u=0`} />) entre ellos y en todo el borde. Es el sello de un modo de alta frecuencia:
        estructura fina, regular y oscilatoria que un PINN <em>sin</em> características de Fourier no logra reproducir.
        Que la red recupere esta cuadrícula nítida: y no una versión borrosa de baja frecuencia: es la evidencia visual
        de que el embedding venció el sesgo espectral.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`u(x,y)`} />
        muestra el tablero de ajedrez de lóbulos claros (<InlineMath tex={String.raw`+1`} />) y oscuros
        (<InlineMath tex={String.raw`-1`} />); al pasar el cursor se lee la amplitud exacta y se confirma que los nodos caen
        en <InlineMath tex={String.raw`u=0`} /> y los bordes están limpios. Los <strong>perfiles de corte</strong>
        en <InlineMath tex={String.raw`x`} /> e <InlineMath tex={String.raw`y`} /> son cada uno una sinusoide de tres
        ciclos completos: contar los picos verifica el número de onda. Como es un benchmark de número de onda
        fijo, el tab <strong>Live</strong> re-evalúa la red entrenada (la misma física) en el navegador
        (onnxruntime-web), sin deslizador de parámetro; su salida se compara con el patrón exacto para ver el error
        residual del PINN. Más allá de las vistas propias de este caso, la vista <strong>Compare</strong> pone en una
        misma grilla el solver clásico de diferencias finitas, la PINN ingenua de tanh puro (120.8% relativo-L2) y la
        PINN con características de Fourier (9.3%); <strong>Diagnostics</strong> muestra el barrido de número de onda (la
        ingenua ~3% en n=1 subiendo a ~100% en n=2 y superiores mientras Fourier se mantiene bajo) y la energía espectral
        radial; <strong>Training</strong> reproduce checkpoints reales donde el carril ingenuo nunca baja de ~100% y
        Fourier converge a ~9%.
      </p>
    </>
  ): (
    <>
      <h2>a high-frequency standing wave: the 2D Helmholtz equation</h2>
      <p>
        The Helmholtz equation
        <InlineMath tex={String.raw`\nabla^2 u + k_0^2\,u = -f`} /> is the <em>frequency-domain</em> form of the wave
        equation: it describes the steady state of a field vibrating at a single frequency (room acoustics, microwave
        electromagnetics, the vibration of a membrane). The wavenumber <InlineMath tex={String.raw`k_0`} /> sets
        <strong> how many oscillations</strong> fit in the domain. With <InlineMath tex={String.raw`k_0=2\pi n`} /> and
        <InlineMath tex={String.raw`n=3`} />, the solution is a checkerboard pattern of alternating maxima and minima: 
        and that very <strong>high spatial frequency</strong> is what makes a naive PINN fail. This case shows the
        technique that cracks it: a <strong>Fourier-feature</strong> input embedding.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> the unit square <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, a <InlineMath tex={String.raw`121\times121`} /> field grid.</li>
        <li><strong>Unknown:</strong> the steady amplitude <InlineMath tex={String.raw`u(x,y)`} /> (e.g. acoustic pressure or membrane displacement).</li>
        <li><strong>Wavenumber:</strong> <InlineMath tex={String.raw`k_0=2\pi n`} /> with <InlineMath tex={String.raw`n=3`} /> (<InlineMath tex={String.raw`k_0=6\pi\approx18.8`} />): three full wavelengths per side.</li>
        <li><strong>Source:</strong> <InlineMath tex={String.raw`f=k_0^2\sin(k_0 x)\sin(k_0 y)`} />: chosen (method of manufactured solutions) so the exact solution is clean.</li>
        <li><strong>Boundary:</strong> homogeneous Dirichlet <InlineMath tex={String.raw`u=0`} /> on all of <InlineMath tex={String.raw`\partial\Omega`} /> (a fixed-edge domain: a clamped membrane patch).</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        We use the <strong>method of manufactured solutions</strong> (MMS): the target solution is fixed on purpose and
        the source <InlineMath tex={String.raw`f`} /> that produces it is derived, so the
        <strong> validation anchor</strong> is exact and the reported relative-L2 is the <em>true</em> PINN error:
      </p>
      <Equation tex={String.raw`u^*(x,y)=\sin(k_0 x)\,\sin(k_0 y),\qquad k_0=2\pi n.`} />
      <p>
        The check is direct. Since
        <InlineMath tex={String.raw`\partial_{xx}u^*=-k_0^2 u^*`} /> and
        <InlineMath tex={String.raw`\partial_{yy}u^*=-k_0^2 u^*`} />, we get
        <InlineMath tex={String.raw`\nabla^2 u^*=-2k_0^2 u^*`} />; substituting,
        <InlineMath tex={String.raw`\nabla^2 u^* + k_0^2 u^* = -k_0^2 u^* = -f`} /> with
        <InlineMath tex={String.raw`f=k_0^2 u^*`} />: the residual is <strong>zero</strong>. Moreover
        <InlineMath tex={String.raw`u^*`} /> vanishes on all four edges (because <InlineMath tex={String.raw`\sin(k_0\cdot 0)=\sin(k_0\cdot 1)=0`} /> for <InlineMath tex={String.raw`k_0=2\pi n`} /> with integer <InlineMath tex={String.raw`n`} />), so it satisfies the Dirichlet BC exactly. The PINN
        <InlineMath tex={String.raw`u_\theta(x,y)`} /> minimises the Helmholtz residual at collocation points, with the BC
        imposed <strong>softly and weighted</strong> (weight <InlineMath tex={String.raw`100\times`} />): the robust
        recipe for oscillatory solutions, where a multiplicative hard constraint would fight the oscillation near the
        boundary.
      </p>

      <h3>The method: why the naive PINN fails, and Fourier features</h3>
      <p>
        A <InlineMath tex={String.raw`\tanh`} /> MLP suffers <strong>spectral bias</strong> (Rahaman et al. 2019):
        networks learn <em>low</em>-frequency content first and struggle enormously to represent <em>high</em>-frequency
        content. With <InlineMath tex={String.raw`k_0=6\pi`} /> the target is purely high-frequency, so a plain MLP
        stalls on a large-error plateau. The fix is to inject the frequencies <em>before</em> the network, with a random
        <strong> Fourier-feature</strong> map (Tancik et al. 2020; Wang-Wang-Perdikaris 2021, multi-scale):
      </p>
      <Equation tex={String.raw`\gamma(\mathbf{x})=\big[\sin(2\pi B\mathbf{x}),\ \cos(2\pi B\mathbf{x})\big],\qquad B_{ij}\sim\mathcal{N}(0,\sigma^2),`} />
      <p>
        with the matrix <InlineMath tex={String.raw`B`} /> <strong>frozen</strong> (untrained, seeded) and two scales
        <InlineMath tex={String.raw`\sigma\in\{1,n\}`} /> covering both the slow variation and the target frequency. The two
        stacked scales give a 256-dim embedding feeding a <InlineMath tex={String.raw`4\times128`} /> tanh FNN, optimized
        Adam (25 000 steps) then L-BFGS, with collocation at ~12 points per wavelength per axis.
        Because <InlineMath tex={String.raw`\gamma`} /> is pure tensor algebra, it traces cleanly into ONNX, so the
        <strong> Live</strong> tab replays the exact same network in the browser. The embedding turns the
        high-frequency problem into one the MLP <em>can</em> fit.
      </p>

      <p>
        <strong>Result and honesty.</strong> Measured at seed 42, the Fourier-feature PINN reaches a relative-L2 of
        0.1026 (~10%) against the analytic field, with a max absolute error of 0.158; the ONNX export matches the
        trained net to 4.77e-07 (max abs) and runs <strong>Live</strong> at 340 KB, 18.5 ms infer. The ~10% is honestly
        CPU-limited: the Fourier map lifts the spectral-bias plateau that would otherwise leave a vanilla MLP an order of
        magnitude worse, but at <InlineMath tex={String.raw`k_0=6\pi`} /> on the CPU lane the standing pattern resolves to
        ~10%, not to 1e-3. GPU training plus frequency annealing would tighten it further; this is not dressed up.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> 2D Helmholtz at a fixed wavenumber <InlineMath tex={String.raw`k_0=6\pi`} />
        (<InlineMath tex={String.raw`n=3`} />), an MMS source, homogeneous Dirichlet BC, on CPU. <strong>Out of
        scope:</strong> a continuous wavenumber <em>sweep</em>: a single Fourier-feature network can cover a narrow band
        <InlineMath tex={String.raw`n\in[2,4]`} /> but accuracy degrades toward the sharp end (spectral bias re-emerges
        higher in the band), so this ships as a <strong>fixed-wavenumber benchmark</strong> that shows the technique at
        its most demanding rather than a sweep with uneven error. Also out of scope: <em>resonance</em> (when
        <InlineMath tex={String.raw`k_0`} /> hits a Laplacian eigenvalue the homogeneous problem loses uniqueness),
        complex scattering domains with the Sommerfeld radiation condition, and a heterogeneous medium
        <InlineMath tex={String.raw`k_0(x,y)`} />.
      </p>

      <p>
        <strong>What the benchmark shows.</strong> The field is a <strong>standing-wave</strong> pattern: a regular grid
        of lobes alternating between <InlineMath tex={String.raw`+1`} /> and <InlineMath tex={String.raw`-1`} /> (three full
        wavelengths per side, a 6x6 checkerboard of 36 alternating lobes), with nodes (<InlineMath tex={String.raw`u=0`} />) between them and along the whole
        boundary. It is the signature of a high-frequency mode: fine, regular, oscillatory structure that a PINN
        <em> without</em> Fourier features cannot reproduce. That the network recovers this crisp grid: rather than a
        blurred low-frequency version: is the visual evidence the embedding beat the spectral bias.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`u(x,y)`} /> shows the checkerboard of bright (<InlineMath tex={String.raw`+1`} />) and
        dark (<InlineMath tex={String.raw`-1`} />) lobes; hover to read the exact amplitude and confirm the nodes sit at
        <InlineMath tex={String.raw`u=0`} /> and the edges are clean. Watch the <strong>line-cut profiles</strong> in
        <InlineMath tex={String.raw`x`} /> and <InlineMath tex={String.raw`y`} />: each is a three-cycle sinusoid: count
        the peaks to verify the wavenumber. Since it is a fixed-wavenumber benchmark, the <strong>Live</strong> tab
        re-evaluates the trained network (the same physics) in the browser (onnxruntime-web), with no parameter slider;
        compare its output against the exact pattern to see the PINN's residual error. Beyond this case's own views, the
        <strong> Compare</strong> view puts the classical finite-difference solve, the naive plain-tanh PINN (120.8%
        relative-L2), and the Fourier-feature PINN (9.3%) on one grid; <strong>Diagnostics</strong> shows the wavenumber
        sweep (naive ~3% at n=1 climbing to ~100% at n=2 and above while Fourier stays low) and the radial spectral
        energy; <strong>Training</strong> replays real checkpoints where the naive lane never leaves ~100% and Fourier
        converges to ~9%.
      </p>
    </>
  );
}
