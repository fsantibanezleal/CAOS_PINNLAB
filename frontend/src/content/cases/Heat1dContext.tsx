import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-heat1d (1D transient heat / diffusion, parametric diffusivity). */
export function Heat1dContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>cómo se relaja un perfil de temperatura: difusión 1D con difusividad ajustable</h2>
      <p>
        Una barra de longitud unitaria parte con un perfil de temperatura senoidal
        <InlineMath tex={String.raw`u(x,0)=\sin(\pi x)`} /> y sus extremos se mantienen a cero. ¿Cómo decae ese perfil
        en el tiempo? La ecuación del calor (o de difusión) <InlineMath tex={String.raw`u_t=\alpha\,u_{xx}`} /> es la
        EDP <em>parabólica</em> canónica: gobierna la conducción térmica, la difusión de un soluto y, con un cambio de
        signo, el suavizado en visión por computador. Aquí la <strong>difusividad</strong>
        <InlineMath tex={String.raw`\alpha`} /> es ajustable de forma continua: una sola red aprende toda la familia
        <InlineMath tex={String.raw`u(x,t;\alpha)`} />, y en el tab <strong>Live</strong> mover el deslizador de
        <InlineMath tex={String.raw`\alpha`} /> hace que el perfil decaiga más rápido o más lento, en vivo.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> espacio <InlineMath tex={String.raw`x\in[0,1]`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`161\times101`} />.</li>
        <li><strong>Incógnita:</strong> el campo <InlineMath tex={String.raw`u(x,t)`} /> (temperatura / concentración).</li>
        <li><strong>Parámetro de control:</strong> la <em>difusividad</em> <InlineMath tex={String.raw`\alpha\in[0.1,1.0]`} />: un input de la red. Fija la tasa de decaimiento.</li>
        <li><strong>Condición inicial:</strong> <InlineMath tex={String.raw`u(x,0)=\sin(\pi x)`} /> (modo fundamental).</li>
        <li><strong>Condiciones de borde:</strong> Dirichlet homogéneas <InlineMath tex={String.raw`u(0,t)=u(1,t)=0`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Para el modo fundamental, la solución exacta es separable y decae exponencialmente, válida para cualquier
        <InlineMath tex={String.raw`\alpha`} />: es nuestra <strong>ancla de validación</strong>:
      </p>
      <Equation tex={String.raw`u^*(x,t;\alpha)=e^{-\alpha\pi^2 t}\,\sin(\pi x).`} />
      <p>
        La constante de decaimiento <InlineMath tex={String.raw`\alpha\pi^2`} /> sale de sustituir el modo en la EDP
        (<InlineMath tex={String.raw`-\alpha\pi^2 = \alpha\cdot(-\pi^2)`} />). La PINN
        <InlineMath tex={String.raw`u_\theta(x,t,\alpha)`} /> minimiza el residual en puntos de colocación sobre
        <InlineMath tex={String.raw`(0,1)\times(0,1)\times[0.1,1]`} />, y la IC + las BC se imponen de forma
        <strong>exacta</strong> por una restricción dura (sin términos de pérdida de IC/BC):
      </p>
      <Equation tex={String.raw`u_\theta = t\,x(1-x)\,\mathcal{N}_\theta(x,t,\alpha) + \sin(\pi x).`} />
      <p>
        En <InlineMath tex={String.raw`t=0`} /> el primer término se anula y queda <InlineMath tex={String.raw`\sin(\pi x)`} />
        (la IC); en <InlineMath tex={String.raw`x=0,1`} /> el factor <InlineMath tex={String.raw`x(1-x)`} /> y
        <InlineMath tex={String.raw`\sin(\pi x)`} /> se anulan (las BC). La red solo aprende la <em>relajación interior</em>.
        La red cruda <InlineMath tex={String.raw`\mathcal{N}_\theta(x,t,\alpha)`} /> es un MLP tanh
        <InlineMath tex={String.raw`[3,48,48,48,48,1]`} /> (DeepXDE), entrenado con Adam y pulido con L-BFGS sobre el
        residual puro de la EDP.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> calor 1D lineal, difusividad constante en espacio (paramétrica en
        <InlineMath tex={String.raw`\alpha`} />), IC de modo único. La restricción dura sobrevive a la exportación a
        ONNX, por eso el tab Live re-evalúa el campo fielmente (paridad ONNX ~5e-7 máx. abs. vs la red entrenada).
        <strong>Fuera de alcance:</strong> difusividad variable o
        no lineal <InlineMath tex={String.raw`\alpha(x,u)`} />, fuentes/sumideros, condiciones de Neumann, y 2-D/3-D
        (esos aparecen en los casos de transporte reactivo y barrera de difusión). Es un <strong>benchmark de libro de
        texto</strong>: la verdad de referencia es la solución separable exacta, no un dataset medido, y la red se
        califica contra esa forma cerrada.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> El barrido de difusividad recorre la física del decaimiento:
        <em>α=0.1</em> (lenta): el perfil apenas baja en la ventana de tiempo; <em>α=0.2 / 0.4 / 0.6</em> aceleran el
        decaimiento; <em>α=0.8</em> y <em>α=1.0</em> (rápida) colapsan la sinusoide a casi cero en
        <InlineMath tex={String.raw`t=1`} />. A igual perfil inicial, la difusividad fija <em>cuán rápido se olvida</em>
        el sistema su condición inicial. En las seis variantes la reconstrucción se mantiene dentro del
        <strong>0.15% de L2 relativo</strong> respecto de la forma cerrada.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> muestra
        <InlineMath tex={String.raw`u(x,t)`} /> con <InlineMath tex={String.raw`x`} /> horizontal y
        <InlineMath tex={String.raw`t`} /> vertical (el tiempo sube): una banda brillante abajo (t pequeño) que se
        desvanece hacia arriba (t grande) es la relajación. Al pasar el cursor se lee la temperatura exacta en cualquier
        <InlineMath tex={String.raw`(x,t)`} /> y se muestra el <strong>perfil de corte</strong> en
        <InlineMath tex={String.raw`x`} /> (la forma senoidal) y en <InlineMath tex={String.raw`t`} /> (el decaimiento
        exponencial). Los <strong>chips</strong> cargan cada difusividad; en <strong>Live</strong>, al deslizar
        <InlineMath tex={String.raw`\alpha`} /> el campo se recalcula en vivo en el navegador (onnxruntime-web).
      </p>
    </>
  ): (
    <>
      <h2>how a temperature profile relaxes: 1D diffusion with a tunable diffusivity</h2>
      <p>
        A unit-length bar starts with a sinusoidal temperature profile
        <InlineMath tex={String.raw`u(x,0)=\sin(\pi x)`} /> and its ends are held at zero. How does that profile decay
        in time? The heat (or diffusion) equation <InlineMath tex={String.raw`u_t=\alpha\,u_{xx}`} /> is the canonical
        <em> parabolic</em> PDE: it governs thermal conduction, solute diffusion and, with a sign flip, smoothing in
        computer vision. Here the <strong>diffusivity</strong> <InlineMath tex={String.raw`\alpha`} /> is continuously
        tunable: a single network learns the whole family <InlineMath tex={String.raw`u(x,t;\alpha)`} />, and in the
        <strong>Live</strong> tab moving the <InlineMath tex={String.raw`\alpha`} /> slider makes the profile decay
        faster or slower, live.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> space <InlineMath tex={String.raw`x\in[0,1]`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, a <InlineMath tex={String.raw`161\times101`} /> field grid.</li>
        <li><strong>Unknown:</strong> the field <InlineMath tex={String.raw`u(x,t)`} /> (temperature / concentration).</li>
        <li><strong>Control parameter:</strong> the <em>diffusivity</em> <InlineMath tex={String.raw`\alpha\in[0.1,1.0]`} />: a network input. It sets the decay rate.</li>
        <li><strong>Initial condition:</strong> <InlineMath tex={String.raw`u(x,0)=\sin(\pi x)`} /> (the fundamental mode).</li>
        <li><strong>Boundary conditions:</strong> homogeneous Dirichlet <InlineMath tex={String.raw`u(0,t)=u(1,t)=0`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        For the fundamental mode the exact solution is separable and decays exponentially, valid for any
        <InlineMath tex={String.raw`\alpha`} />: it is our <strong>validation anchor</strong>:
      </p>
      <Equation tex={String.raw`u^*(x,t;\alpha)=e^{-\alpha\pi^2 t}\,\sin(\pi x).`} />
      <p>
        The decay constant <InlineMath tex={String.raw`\alpha\pi^2`} /> follows from substituting the mode into the PDE
        (<InlineMath tex={String.raw`-\alpha\pi^2 = \alpha\cdot(-\pi^2)`} />). The PINN
        <InlineMath tex={String.raw`u_\theta(x,t,\alpha)`} /> minimises the residual at collocation points over
        <InlineMath tex={String.raw`(0,1)\times(0,1)\times[0.1,1]`} />, and the IC + BCs are imposed
        <strong>exactly</strong> by a hard constraint (no IC/BC loss terms):
      </p>
      <Equation tex={String.raw`u_\theta = t\,x(1-x)\,\mathcal{N}_\theta(x,t,\alpha) + \sin(\pi x).`} />
      <p>
        At <InlineMath tex={String.raw`t=0`} /> the first term vanishes leaving <InlineMath tex={String.raw`\sin(\pi x)`} />
        (the IC); at <InlineMath tex={String.raw`x=0,1`} /> the factor <InlineMath tex={String.raw`x(1-x)`} /> and
        <InlineMath tex={String.raw`\sin(\pi x)`} /> vanish (the BCs). The network only learns the <em>interior relaxation</em>.
        The raw net <InlineMath tex={String.raw`\mathcal{N}_\theta(x,t,\alpha)`} /> is a
        <InlineMath tex={String.raw`[3,48,48,48,48,1]`} /> tanh MLP (DeepXDE), trained with Adam then polished with
        L-BFGS on the bare PDE residual.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> linear 1D heat, space-constant diffusivity (parametric in
        <InlineMath tex={String.raw`\alpha`} />), a single-mode IC. The hard constraint survives the ONNX export, which
        is why the Live tab re-evaluates the field faithfully (ONNX parity ~5e-7 max abs vs the trained net).
        <strong>Out of scope:</strong> variable or nonlinear
        diffusivity <InlineMath tex={String.raw`\alpha(x,u)`} />, sources/sinks, Neumann conditions, and 2-D/3-D (those
        appear in the reactive-transport and diffusion-barrier cases). This is a <strong>textbook benchmark</strong>:
        the ground truth is the exact separable solution, not a measured dataset, and the network is graded against
        that closed form.
      </p>

      <p>
        <strong>What each variant shows.</strong> The diffusivity sweep walks the physics of decay: <em>α=0.1</em>
        (slow): the profile barely drops over the window; <em>α=0.2 / 0.4 / 0.6</em> accelerate the decay; <em>α=0.8</em>
        and <em>α=1.0</em> (fast) collapse the sine to near zero by <InlineMath tex={String.raw`t=1`} />. From the same
        initial profile, the diffusivity sets <em>how fast the system forgets</em> its initial condition. Across all six
        variants the reconstruction stays within <strong>0.15% relative-L2</strong> of the closed form.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> shows
        <InlineMath tex={String.raw`u(x,t)`} /> with <InlineMath tex={String.raw`x`} /> horizontal and
        <InlineMath tex={String.raw`t`} /> vertical (time flows upward): a bright band at the bottom (small t) fading upward (large t) is the
        relaxation. Hover to read the exact temperature at any <InlineMath tex={String.raw`(x,t)`} /> and watch the
        <strong>line-cut profile</strong> in <InlineMath tex={String.raw`x`} /> (the sine shape) and in
        <InlineMath tex={String.raw`t`} /> (the exponential decay). The <strong>chips</strong> load each diffusivity;
        in <strong>Live</strong>, slide <InlineMath tex={String.raw`\alpha`} /> and watch the field recompute live in
        the browser (onnxruntime-web).
      </p>
    </>
  );
}
