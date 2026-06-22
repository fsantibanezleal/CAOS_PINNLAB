import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-wave1d (1D wave equation, parametric speed, SIREN + hard constraints). */
export function Wave1dContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: una onda estacionaria en una cuerda — la ecuación de onda 1D con velocidad ajustable</h2>
      <p>
        <strong>El problema.</strong> Una cuerda fija en ambos extremos se suelta desde un perfil senoidal en reposo.
        Oscila para siempre (sin amortiguamiento) como una <em>onda estacionaria</em>. La ecuación de onda
        <InlineMath tex={String.raw`u_{tt}=c^2 u_{xx}`} /> es la EDP <em>hiperbólica</em> canónica — gobierna cuerdas
        vibrantes, acústica, ondas electromagnéticas y sísmicas. A diferencia del calor (que olvida su condición
        inicial), la onda la <strong>recuerda</strong> y la recorre. Aquí la <strong>velocidad de onda</strong>
        <InlineMath tex={String.raw`c`} /> es ajustable: una sola red aprende toda la familia
        <InlineMath tex={String.raw`u(x,t;c)`} />, y en el tab <strong>Live</strong> mover el deslizador de
        <InlineMath tex={String.raw`c`} /> hace que la onda oscile más rápido o más lento.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> espacio <InlineMath tex={String.raw`x\in[0,1]`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`161\times161`} />.</li>
        <li><strong>Incógnita:</strong> el desplazamiento <InlineMath tex={String.raw`u(x,t)`} />.</li>
        <li><strong>Parámetro de control:</strong> la <em>velocidad de onda</em> <InlineMath tex={String.raw`c\in[0.5,2]`} /> — un input de la red; fija la frecuencia de oscilación.</li>
        <li><strong>Condiciones iniciales (dos, por ser 2º orden en t):</strong> <InlineMath tex={String.raw`u(x,0)=\sin(\pi x)`} /> (forma) y <InlineMath tex={String.raw`u_t(x,0)=0`} /> (parte del reposo).</li>
        <li><strong>Condiciones de borde:</strong> extremos fijos <InlineMath tex={String.raw`u(0,t)=u(1,t)=0`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        El modo fundamental es una <strong>onda estacionaria</strong>: la forma espacial se mantiene y solo su amplitud
        oscila en el tiempo. La solución exacta, válida para cualquier <InlineMath tex={String.raw`c`} />, es nuestra
        ancla:
      </p>
      <Equation tex={String.raw`u^*(x,t;c)=\sin(\pi x)\,\cos(c\pi t).`} />
      <p>
        La frecuencia temporal <InlineMath tex={String.raw`c\pi`} /> sale de sustituir el modo en la EDP. La PINN
        <InlineMath tex={String.raw`u_\theta(x,t,c)`} /> usa <strong>activaciones senoidales (SIREN)</strong> como
        remedio al sesgo espectral de la solución oscilatoria, y satisface las DOS ICs + las BC de forma
        <strong>exacta</strong> por una restricción dura:
      </p>
      <Equation tex={String.raw`u_\theta = \sin(\pi x) + t^2\,x(1-x)\,\mathcal{N}_\theta(x,t,c).`} />
      <p>
        En <InlineMath tex={String.raw`t=0`} /> queda <InlineMath tex={String.raw`\sin(\pi x)`} /> (IC#1); como cada
        término del término añadido lleva un factor <InlineMath tex={String.raw`t^2`} />, su derivada en
        <InlineMath tex={String.raw`t=0`} /> se anula (IC#2, parte del reposo); y <InlineMath tex={String.raw`x(1-x)`} />
        anula los bordes (BC). Sin términos de pérdida de IC/BC, el entrenamiento es estable.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> onda 1D lineal sin amortiguamiento, velocidad constante (paramétrica en
        <InlineMath tex={String.raw`c`} />), modo único. <strong>Fuera de alcance:</strong> amortiguamiento, medios no
        homogéneos <InlineMath tex={String.raw`c(x)`} />, no linealidad, y reflexión/transmisión en interfaces. El reto
        numérico es la <em>oscilación</em> (sesgo espectral): SIREN la maneja donde una tanh se estancaría.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> El barrido de velocidad recorre las fracciones de período: a
        <em>c=0.5</em> la onda apenas avanza un cuarto de período en la ventana; <em>c=1</em> da medio período (el
        perfil se invierte de signo en <InlineMath tex={String.raw`t=1`} />); <em>c=2</em> completa un período entero
        (la onda vuelve a su estado inicial). Las velocidades intermedias (0.75, 1.25, 1.5) muestran la transición.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de
        <InlineMath tex={String.raw`u(x,t)`} /> (x horizontal, t vertical) muestra <em>franjas diagonales/onduladas</em>
        — el patrón espacio-tiempo de la onda, alternando crestas (claro) y valles (oscuro). Pasa el cursor para leer
        el desplazamiento exacto y mira el <strong>perfil de corte</strong> en <InlineMath tex={String.raw`t`} /> (un
        coseno) y en <InlineMath tex={String.raw`x`} /> (la forma senoidal). Los <strong>chips</strong> cargan cada
        velocidad; en <strong>Live</strong>, desliza <InlineMath tex={String.raw`c`} /> y ve la onda recalcularse en
        vivo en tu navegador (onnxruntime-web).
      </p>
    </>
  ) : (
    <>
      <h2>The problem: a standing wave on a string — the 1D wave equation with a tunable speed</h2>
      <p>
        <strong>The problem.</strong> A string fixed at both ends is released from a sinusoidal profile at rest. It
        oscillates forever (no damping) as a <em>standing wave</em>. The wave equation
        <InlineMath tex={String.raw`u_{tt}=c^2 u_{xx}`} /> is the canonical <em>hyperbolic</em> PDE — it governs
        vibrating strings, acoustics, electromagnetic and seismic waves. Unlike heat (which forgets its initial
        condition), a wave <strong>remembers</strong> and carries it. Here the <strong>wave speed</strong>
        <InlineMath tex={String.raw`c`} /> is tunable: a single network learns the whole family
        <InlineMath tex={String.raw`u(x,t;c)`} />, and in the <strong>Live</strong> tab moving the
        <InlineMath tex={String.raw`c`} /> slider makes the wave oscillate faster or slower.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> space <InlineMath tex={String.raw`x\in[0,1]`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, a <InlineMath tex={String.raw`161\times161`} /> field grid.</li>
        <li><strong>Unknown:</strong> the displacement <InlineMath tex={String.raw`u(x,t)`} />.</li>
        <li><strong>Control parameter:</strong> the <em>wave speed</em> <InlineMath tex={String.raw`c\in[0.5,2]`} /> — a network input; it sets the oscillation frequency.</li>
        <li><strong>Initial conditions (two, since 2nd order in t):</strong> <InlineMath tex={String.raw`u(x,0)=\sin(\pi x)`} /> (shape) and <InlineMath tex={String.raw`u_t(x,0)=0`} /> (released from rest).</li>
        <li><strong>Boundary conditions:</strong> fixed ends <InlineMath tex={String.raw`u(0,t)=u(1,t)=0`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        The fundamental mode is a <strong>standing wave</strong>: the spatial shape is fixed and only its amplitude
        oscillates in time. The exact solution, valid for any <InlineMath tex={String.raw`c`} />, is our anchor:
      </p>
      <Equation tex={String.raw`u^*(x,t;c)=\sin(\pi x)\,\cos(c\pi t).`} />
      <p>
        The temporal frequency <InlineMath tex={String.raw`c\pi`} /> follows from substituting the mode into the PDE.
        The PINN <InlineMath tex={String.raw`u_\theta(x,t,c)`} /> uses <strong>sinusoidal activations (SIREN)</strong> as
        the spectral-bias remedy for the oscillatory solution, and satisfies the TWO ICs + BCs <strong>exactly</strong>
        by a hard constraint:
      </p>
      <Equation tex={String.raw`u_\theta = \sin(\pi x) + t^2\,x(1-x)\,\mathcal{N}_\theta(x,t,c).`} />
      <p>
        At <InlineMath tex={String.raw`t=0`} /> it leaves <InlineMath tex={String.raw`\sin(\pi x)`} /> (IC#1); since the
        added term carries a factor <InlineMath tex={String.raw`t^2`} />, its time-derivative at
        <InlineMath tex={String.raw`t=0`} /> vanishes (IC#2, released from rest); and <InlineMath tex={String.raw`x(1-x)`} />
        kills the boundaries (BC). With no IC/BC loss terms, training is stable.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> linear 1D wave, no damping, constant speed (parametric in
        <InlineMath tex={String.raw`c`} />), single mode. <strong>Out of scope:</strong> damping, inhomogeneous media
        <InlineMath tex={String.raw`c(x)`} />, nonlinearity, and reflection/transmission at interfaces. The numerical
        challenge is the <em>oscillation</em> (spectral bias): SIREN handles it where a tanh would stall.
      </p>

      <p>
        <strong>What each variant shows.</strong> The speed sweep walks the period fractions: at <em>c=0.5</em> the wave
        advances only a quarter period over the window; <em>c=1</em> gives half a period (the profile flips sign by
        <InlineMath tex={String.raw`t=1`} />); <em>c=2</em> completes a full period (the wave returns to its initial
        state). The intermediate speeds (0.75, 1.25, 1.5) show the transition.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`u(x,t)`} /> (x horizontal, t vertical) shows <em>wavy diagonal bands</em> — the
        space-time pattern of the wave, alternating crests (light) and troughs (dark). Hover to read the exact
        displacement and watch the <strong>line-cut profile</strong> in <InlineMath tex={String.raw`t`} /> (a cosine)
        and in <InlineMath tex={String.raw`x`} /> (the sine shape). The <strong>chips</strong> load each speed; in
        <strong>Live</strong>, slide <InlineMath tex={String.raw`c`} /> and watch the wave recompute live in your
        browser (onnxruntime-web).
      </p>
    </>
  );
}
