import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for mine-flotation-kinetics (parametric first-order flotation, C(k,t) family map). */
export function FlotationContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: ¿qué tan rápido flota el mineral?: cinética de flotación de primer orden</h2>
      <p>
        <strong>El problema.</strong> En una celda de flotación batch, las partículas de mineral valioso se adhieren a
        burbujas y suben a la espuma. La fracción flotable <InlineMath tex={String.raw`C`} /> decae con una cinética de
        <strong> primer orden</strong>, <InlineMath tex={String.raw`\dot C=-kC`} />, donde la <strong>constante de
        tasa</strong> <InlineMath tex={String.raw`k`} /> (1/min) resume toda la flotabilidad: tamaño de partícula,
        reactivos, hidrodinámica. La <em>recuperación</em> es <InlineMath tex={String.raw`R(t)=1-C(t)`} />. En vez de
        un solo <InlineMath tex={String.raw`k`} />, esta PINN aprende la <strong>familia completa</strong>
        <InlineMath tex={String.raw`C(k,t)`} /> tomando <InlineMath tex={String.raw`k`} /> como un segundo input de la
        red: el modelo agrupado (lumped) que se usa para comparar circuitos de flotación.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Ejes del campo:</strong> constante de tasa <InlineMath tex={String.raw`k\in[0.5,5]`} /> (horizontal) × tiempo <InlineMath tex={String.raw`t\in[0,1]`} /> (vertical), grilla <InlineMath tex={String.raw`81\times81`} />.</li>
        <li><strong>Incógnita:</strong> la fracción flotable restante <InlineMath tex={String.raw`C(k,t)\in[0,1]`} />; la recuperación es <InlineMath tex={String.raw`R=1-C`} />.</li>
        <li><strong>Constante de tasa:</strong> <InlineMath tex={String.raw`k`} />: alta = flotación rápida (partículas bien liberadas, buen reactivo); baja = lenta.</li>
        <li><strong>Condición inicial:</strong> <InlineMath tex={String.raw`C(k,0)=1`} /> (toda la masa flotable presente al inicio).</li>
      </ul>

      <h3>Formalización</h3>
      <p>La EDO de primer orden tiene solución exacta para cualquier <InlineMath tex={String.raw`k`} />: nuestra ancla:</p>
      <Equation tex={String.raw`\frac{dC}{dt}=-k\,C,\quad C(k,0)=1\ \Longrightarrow\ C^*(k,t)=e^{-k t},\qquad R=1-e^{-kt}.`} />
      <p>
        La PINN <InlineMath tex={String.raw`C_\theta(k,t)`} /> minimiza el residual de la EDO en el plano
        <InlineMath tex={String.raw`(k,t)`} />, con la IC impuesta de forma <strong>exacta</strong> por una restricción
        dura: <InlineMath tex={String.raw`C_\theta = 1 + t\,\mathcal{N}_\theta(k,t)`} /> se anula a
        <InlineMath tex={String.raw`1`} /> en <InlineMath tex={String.raw`t=0`} />: así una sola red entrenada da la
        concentración (y la recuperación) para <em>cualquier</em> constante de tasa sin reentrenar. La red es una FNN
        tanh <InlineMath tex={String.raw`[2,32,32,32,1]`} /> (init Glorot-normal); como la CI es dura, la pérdida es solo
        el residual de la EDO (<code>num_boundary=0</code>), entrenada con 10.000 pasos Adam a lr=1e-3 sobre 2000 puntos
        de dominio + 200 iniciales, luego un pulido L-BFGS, y evaluada contra la forma cerrada en 4000 puntos de prueba.
      </p>

      <p>
        <strong>Resultado (medido, semilla 42).</strong> La red paramétrica entrenada calza con el ancla analítica
        <InlineMath tex={String.raw`C^*=e^{-kt}`} /> a un L2 relativo de <strong>0,076 %</strong> (7,6e-4), con error
        absoluto máximo <strong>2,22e-3</strong>, superando la banda propia del caso (bajo 5e-3) por un orden de
        magnitud. Es un campo suave y de baja dimensión con ancla analítica exacta, así que la corrida en CPU lo resuelve
        esencialmente a precisión de optimizador y no se necesita salvedad de exactitud. Un único ONNX de
        <strong>19,4 KB</strong> reproduce la red a <strong>7,15e-7</strong> (abs máx) e infiere en <strong>0,77 ms</strong>,
        así que la App barre <InlineMath tex={String.raw`k`} /> de forma interactiva y lee
        <InlineMath tex={String.raw`C(k,t)`} /> y <InlineMath tex={String.raw`R=1-C`} /> en vivo.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> cinética agrupada de primer orden, constante de tasa única por corrida,
        recuperación batch. Es <em>ilustrativo-sintético</em>: el modelo de primer orden es el estándar de flotación,
        pero el campo es una ilustración limpia, NO ajustada a un ensayo de planta/laboratorio (los datasets públicos
        son series de tiempo 0-D de proceso, sin un eje de constante de tasa). <strong>Fuera de alcance:</strong>
        espectros de flotabilidad (k distribuido), arrastre (entrainment), cinética de segundo orden y la dinámica de
        la fase espuma.
      </p>

      <p>
        <strong>Qué muestra el mapa.</strong> Como <InlineMath tex={String.raw`k`} /> es un <em>eje</em>, un solo
        heatmap muestra <strong>toda la familia</strong> a la vez: una esquina brillante (poco tiempo o
        <InlineMath tex={String.raw`k`} /> pequeño = poca flotación, <InlineMath tex={String.raw`C\approx1`} />) que se
        oscurece hacia tiempos largos y <InlineMath tex={String.raw`k`} /> grande (<InlineMath tex={String.raw`C\to0`} />,
        recuperación casi total). Las <em>isolíneas</em> son curvas <InlineMath tex={String.raw`kt=\text{const}`} />.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> Pasa el cursor para leer <InlineMath tex={String.raw`C`} /> exacto en
        cualquier <InlineMath tex={String.raw`(k,t)`} /> (la recuperación es <InlineMath tex={String.raw`1-C`} />). El
        <strong> perfil de corte</strong> en <InlineMath tex={String.raw`t`} /> da la curva de decaimiento exponencial
        para un <InlineMath tex={String.raw`k`} /> fijo (la curva clásica de recuperación-vs-tiempo); el corte en
        <InlineMath tex={String.raw`k`} /> muestra cuánto más rápido flota un mineral mejor liberado. En
        <strong> Live</strong>, la red paramétrica re-evalúa el mapa completo en tu navegador (onnxruntime-web).
      </p>
    </>
  ): (
    <>
      <h2>The problem: how fast does the ore float?: first-order flotation kinetics</h2>
      <p>
        <strong>The problem.</strong> In a batch flotation cell, valuable-mineral particles attach to bubbles and rise
        into the froth. The floatable fraction <InlineMath tex={String.raw`C`} /> decays with <strong>first-order</strong>
        kinetics, <InlineMath tex={String.raw`\dot C=-kC`} />, where the <strong>rate constant</strong>
        <InlineMath tex={String.raw`k`} /> (1/min) lumps all the floatability: particle size, reagents, hydrodynamics.
        The <em>recovery</em> is <InlineMath tex={String.raw`R(t)=1-C(t)`} />. Instead of one fixed
        <InlineMath tex={String.raw`k`} />, this PINN learns the <strong>whole family</strong>
        <InlineMath tex={String.raw`C(k,t)`} /> by taking <InlineMath tex={String.raw`k`} /> as a second network input: 
        the lumped model used to compare flotation circuits.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Field axes:</strong> rate constant <InlineMath tex={String.raw`k\in[0.5,5]`} /> (horizontal) × time <InlineMath tex={String.raw`t\in[0,1]`} /> (vertical), an <InlineMath tex={String.raw`81\times81`} /> grid.</li>
        <li><strong>Unknown:</strong> the remaining floatable fraction <InlineMath tex={String.raw`C(k,t)\in[0,1]`} />; recovery is <InlineMath tex={String.raw`R=1-C`} />.</li>
        <li><strong>Rate constant:</strong> <InlineMath tex={String.raw`k`} />: high = fast flotation (well-liberated particles, good reagent); low = slow.</li>
        <li><strong>Initial condition:</strong> <InlineMath tex={String.raw`C(k,0)=1`} /> (all floatable mass present at the start).</li>
      </ul>

      <h3>Formalization</h3>
      <p>The first-order ODE has an exact solution for any <InlineMath tex={String.raw`k`} />: our anchor:</p>
      <Equation tex={String.raw`\frac{dC}{dt}=-k\,C,\quad C(k,0)=1\ \Longrightarrow\ C^*(k,t)=e^{-k t},\qquad R=1-e^{-kt}.`} />
      <p>
        The PINN <InlineMath tex={String.raw`C_\theta(k,t)`} /> minimises the ODE residual over the
        <InlineMath tex={String.raw`(k,t)`} /> plane, with the IC imposed <strong>exactly</strong> by a hard constraint: 
        <InlineMath tex={String.raw`C_\theta = 1 + t\,\mathcal{N}_\theta(k,t)`} /> collapses to
        <InlineMath tex={String.raw`1`} /> at <InlineMath tex={String.raw`t=0`} />: so one trained network gives the
        concentration (and recovery) for <em>any</em> rate constant without retraining. The network is a
        <InlineMath tex={String.raw`[2,32,32,32,1]`} /> tanh FNN (Glorot-normal init); because the IC is hard, the loss
        is PDE-residual-only (<code>num_boundary=0</code>), trained by 10,000 Adam steps at lr=1e-3 over 2000 domain +
        200 initial collocation points, then an L-BFGS polish, and scored against the closed form on 4000 test points.
      </p>

      <p>
        <strong>Result (measured, seed 42).</strong> The trained parametric net matches the analytic anchor
        <InlineMath tex={String.raw`C^*=e^{-kt}`} /> to a relative-L2 of <strong>0.076 %</strong> (7.6e-4), with max
        absolute error <strong>2.22e-3</strong>, clearing this case's own band (under 5e-3) by an order of magnitude.
        This is a smooth, low-dimensional field with an exact analytic anchor, so the CPU lane resolves it essentially to
        optimizer precision and no accuracy caveat is needed. A single <strong>19.4 KB</strong> ONNX reproduces the net
        to <strong>7.15e-7</strong> (max abs) and infers in <strong>0.77 ms</strong>, so the App sweeps
        <InlineMath tex={String.raw`k`} /> interactively and reads <InlineMath tex={String.raw`C(k,t)`} /> and
        <InlineMath tex={String.raw`R=1-C`} /> live.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> lumped first-order kinetics, a single rate constant per run, batch recovery. It is
        <em> illustrative-synthetic</em>: the first-order model is the flotation standard, but the field is a clean
        illustration, NOT fit to a plant/lab assay (public datasets are 0-D process time-series with no rate-constant
        axis). <strong>Out of scope:</strong> floatability spectra (distributed k), entrainment, second-order kinetics,
        and froth-phase dynamics.
      </p>

      <p>
        <strong>What the map shows.</strong> Because <InlineMath tex={String.raw`k`} /> is an <em>axis</em>, one heatmap
        shows the <strong>whole family</strong> at once: a bright corner (short time or small
        <InlineMath tex={String.raw`k`} /> = little flotation, <InlineMath tex={String.raw`C\approx1`} />) darkening
        toward long times and large <InlineMath tex={String.raw`k`} /> (<InlineMath tex={String.raw`C\to0`} />, near-total
        recovery). The <em>contours</em> are curves of <InlineMath tex={String.raw`kt=\text{const}`} />.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> Hover to read the exact <InlineMath tex={String.raw`C`} /> at any
        <InlineMath tex={String.raw`(k,t)`} /> (recovery is <InlineMath tex={String.raw`1-C`} />). The <strong>line-cut
        profile</strong> in <InlineMath tex={String.raw`t`} /> gives the exponential decay curve for a fixed
        <InlineMath tex={String.raw`k`} /> (the classic recovery-vs-time curve); the cut in <InlineMath tex={String.raw`k`} />
        shows how much faster a better-liberated mineral floats. In <strong>Live</strong>, the parametric network
        re-evaluates the full map in your browser (onnxruntime-web).
      </p>
    </>
  );
}
