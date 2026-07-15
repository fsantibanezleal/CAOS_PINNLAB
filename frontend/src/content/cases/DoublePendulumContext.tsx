import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for dyn-double-pendulum (chaotic ODE system; PINN as t -> state; RK45 anchor). */
export function DoublePendulumContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: el péndulo doble: un sistema dinámico caótico, y la PINN como mapa t to estado</h2>
      <p>
        <strong>El problema.</strong> Dos brazos rígidos articulados, el segundo colgando del primero, soltados desde
        el reposo. Es uno de los sistemas <em>caóticos</em> más simples de la mecánica clásica: dos condiciones
        iniciales casi idénticas divergen exponencialmente (el <em>efecto mariposa</em>). A diferencia de todos los
        demás casos de este catálogo, <strong>no hay campo espacial</strong>: la incógnita es una <em>trayectoria</em>
        en el espacio de estados. La red toma el <strong>tiempo</strong> <InlineMath tex={String.raw`t`} /> y devuelve
        los dos ángulos <InlineMath tex={String.raw`(\theta_1,\theta_2)`} />; la "solución" es una animación que se
        mueve, no una imagen fija.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Estado:</strong> los ángulos desde la vertical <InlineMath tex={String.raw`\theta_1(t),\theta_2(t)`} /> y sus velocidades <InlineMath tex={String.raw`\dot\theta_1,\dot\theta_2`} />.</li>
        <li><strong>Constantes:</strong> masas <InlineMath tex={String.raw`m_1=m_2=1`} />, longitudes <InlineMath tex={String.raw`\ell_1=\ell_2=1`} />, gravedad <InlineMath tex={String.raw`g=9.81`} />.</li>
        <li><strong>Condición inicial:</strong> <InlineMath tex={String.raw`\theta_1(0)=\theta_2(0)=120^\circ`} />, soltado del reposo <InlineMath tex={String.raw`\dot\theta_i(0)=0`} />: un régimen claramente caótico.</li>
        <li><strong>Dominio:</strong> tiempo <InlineMath tex={String.raw`t\in[0,3]`} /> s, 601 muestras horneadas.</li>
        <li><strong>Ancla:</strong> un integrador <strong>RK45</strong> de alta precisión (<InlineMath tex={String.raw`\text{rtol}=\text{atol}=10^{-10}`} />): la verdad numérica contra la que se mide la PINN.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        El Lagrangiano da dos EDOs de segundo orden acopladas y no lineales, las aceleraciones
        <InlineMath tex={String.raw`\ddot\theta_1,\ddot\theta_2`} /> en función del estado. La pérdida física de la PINN
        es el residuo de esas dos ecuaciones en puntos de colocación en el tiempo:
      </p>
      <Equation tex={String.raw`r_i(t)=\ddot{\theta}_i^{\,\theta}(t)-f_i\big(\theta_1,\theta_2,\dot\theta_1,\dot\theta_2\big),\quad \mathcal{L}=\tfrac{1}{N}\sum_t\big(r_1^2+r_2^2\big).`} />
      <p>
        La condición inicial se impone de forma <strong>suave</strong>: un término <em>dde.icbc.IC</em> fija
        <InlineMath tex={String.raw`\theta_i(0)`} /> y un <em>OperatorBC</em> fija el soltado del reposo
        <InlineMath tex={String.raw`\dot\theta_i(0)=0`} />, ambos <strong>ponderados 100x sobre el residuo</strong> para
        sujetar firmemente el PVI. Un ansatz <em>duro</em> <InlineMath tex={String.raw`t^2`} /> se probó primero y se
        <strong>descartó</strong>:
      </p>
      <Equation tex={String.raw`\hat\theta_i(t)=\theta_i(0)+t^2\,\mathcal{N}_{\theta,i}(t)\quad(\text{descartado}).`} />
      <p>
        Cumple <InlineMath tex={String.raw`\theta_i(0)`} /> y <InlineMath tex={String.raw`\dot\theta_i(0)=0`} /> por
        construcción, pero como el término añadido lleva un factor <InlineMath tex={String.raw`t^2`} />, su gradiente
        respecto a los parámetros <InlineMath tex={String.raw`\partial\hat\theta_i/\partial\,\text{params}\propto t^2\to 0`} />
        cerca de <InlineMath tex={String.raw`t=0`} /> anula la señal que la red necesita para aprender la aceleración
        inicial (movió <InlineMath tex={String.raw`\theta_1`} /> en el <em>sentido equivocado</em>). La IC suave está bien
        condicionada y es el enfoque estándar de PVI en DeepXDE. La red es un MLP simple
        <InlineMath tex={String.raw`[1,\,96\times4,\,2]`} /> con activaciones <strong>tanh</strong> (SIREN fue inestable
        con IC suave aquí), entrenado Adam (25k) y luego L-BFGS, y exportado a ONNX (paridad
        <InlineMath tex={String.raw`\sim 1.6\times 10^{-6}`} />).
      </p>

      <h3>Honestidad: el caos tiene un muro</h3>
      <p>
        Un péndulo doble es <strong>caótico</strong>: ninguna red fija puede seguir la trayectoria verdadera más allá de
        un horizonte finito (el <em>horizonte de Lyapunov</em>). Por eso la métrica principal NO es un calce a largo
        plazo, sino el <strong>leave-time</strong>: el primer instante en que la PINN se separa del RK45 por más de
        <InlineMath tex={String.raw`0.30`} /> rad. Aquí es un <strong>leave-time medido = 1.99 s</strong>: la PINN sigue
        al RK45 hasta unos <InlineMath tex={String.raw`0.02`} /> rad durante casi dos segundos, y luego la trayectoria se
        despega. La App muestra la PINN (fantasma) sobre el RK45 (sólido): la sigue un rato y luego la pierde: y el brazo
        de la PINN se pone <span style={{ color: "#ff5d5d" }}>rojo</span> tras el leave-time. Las dos curvas naranjas (dos
        inicios separados <InlineMath tex={String.raw`10^{-2}`} /> rad) divergen exponencialmente: ese es el límite,
        mostrado honestamente, no escondido. En toda la ventana <InlineMath tex={String.raw`[0,3]`} /> s el error de
        trayectoria es <strong>9.3%</strong> (L2 relativo), y la separación exponencial del gemelo da una tasa cruda de
        Lyapunov máxima <InlineMath tex={String.raw`\lambda\approx 0.02`} /> 1/s, ambos consistentes con un horizonte de
        predictibilidad de unos dos segundos.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> el péndulo doble ideal sin fricción ni amortiguamiento, masas puntuales, brazos
        rígidos sin masa. <strong>Fuera de alcance:</strong> fricción, brazos con masa distribuida, forzamiento. Datos
        <em>sintéticos-ilustrativos</em> (no es una medición real). El objetivo del caso es <em>didáctico</em>: mostrar
        cómo una PINN aborda un sistema dinámico (no un campo) y dónde el caos la derrota.
      </p>

      <p>
        <strong>Cómo leer la viz.</strong> El <strong>péndulo animado</strong> es el héroe: RK45 sólido (azul) + PINN
        fantasma (discontinua) + la estela del brazo inferior. El <strong>retrato de fase</strong>
        <InlineMath tex={String.raw`\theta_1`} />-<InlineMath tex={String.raw`\theta_2`} /> traza el camino en el espacio
        de estados. El panel <strong>mariposa</strong> muestra <InlineMath tex={String.raw`|\Delta\theta|`} /> de dos
        inicios vecinos creciendo (escala log). Las <strong>series temporales</strong> comparan los ángulos PINN vs RK45
        con la línea de leave-time. Usa Play / la barra para recorrer el tiempo.
      </p>
    </>
  ): (
    <>
      <h2>The problem: the double pendulum: a chaotic dynamical system, and the PINN as a t to state map</h2>
      <p>
        <strong>The problem.</strong> Two rigid arms hinged in series, the second hanging off the first, released from
        rest. It is one of the simplest <em>chaotic</em> systems in classical mechanics: two almost-identical initial
        conditions diverge exponentially (the <em>butterfly effect</em>). Unlike every other case in this catalogue
        there is <strong>no spatial field</strong>: the unknown is a <em>trajectory</em> in state space. The network
        takes <strong>time</strong> <InlineMath tex={String.raw`t`} /> and returns the two angles
        <InlineMath tex={String.raw`(\theta_1,\theta_2)`} />; the "solution" is an animation that moves, not a still
        picture.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>State:</strong> the angles from vertical <InlineMath tex={String.raw`\theta_1(t),\theta_2(t)`} /> and their rates <InlineMath tex={String.raw`\dot\theta_1,\dot\theta_2`} />.</li>
        <li><strong>Constants:</strong> masses <InlineMath tex={String.raw`m_1=m_2=1`} />, lengths <InlineMath tex={String.raw`\ell_1=\ell_2=1`} />, gravity <InlineMath tex={String.raw`g=9.81`} />.</li>
        <li><strong>Initial condition:</strong> <InlineMath tex={String.raw`\theta_1(0)=\theta_2(0)=120^\circ`} />, released from rest <InlineMath tex={String.raw`\dot\theta_i(0)=0`} />: a firmly chaotic regime.</li>
        <li><strong>Domain:</strong> time <InlineMath tex={String.raw`t\in[0,3]`} /> s, 601 baked samples.</li>
        <li><strong>Anchor:</strong> a high-accuracy <strong>RK45</strong> integrator (<InlineMath tex={String.raw`\text{rtol}=\text{atol}=10^{-10}`} />): the numerical truth the PINN is measured against.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        The Lagrangian gives two coupled nonlinear second-order ODEs, the accelerations
        <InlineMath tex={String.raw`\ddot\theta_1,\ddot\theta_2`} /> as functions of the state. The PINN physics loss is
        the residual of those two equations at collocation times:
      </p>
      <Equation tex={String.raw`r_i(t)=\ddot{\theta}_i^{\,\theta}(t)-f_i\big(\theta_1,\theta_2,\dot\theta_1,\dot\theta_2\big),\quad \mathcal{L}=\tfrac{1}{N}\sum_t\big(r_1^2+r_2^2\big).`} />
      <p>
        The initial condition is imposed <strong>softly</strong>: a <em>dde.icbc.IC</em> term pins
        <InlineMath tex={String.raw`\theta_i(0)`} /> and an <em>OperatorBC</em> pins the release from rest
        <InlineMath tex={String.raw`\dot\theta_i(0)=0`} />, both <strong>weighted 100x above the residual</strong> so the
        IVP is firmly held. A <em>hard</em> <InlineMath tex={String.raw`t^2`} /> ansatz was tried first and
        <strong>rejected</strong>:
      </p>
      <Equation tex={String.raw`\hat\theta_i(t)=\theta_i(0)+t^2\,\mathcal{N}_{\theta,i}(t)\quad(\text{rejected}).`} />
      <p>
        It matches <InlineMath tex={String.raw`\theta_i(0)`} /> and <InlineMath tex={String.raw`\dot\theta_i(0)=0`} /> by
        construction, but because the added term carries a factor <InlineMath tex={String.raw`t^2`} />, its parameter
        gradient <InlineMath tex={String.raw`\partial\hat\theta_i/\partial\,\text{params}\propto t^2\to 0`} /> near
        <InlineMath tex={String.raw`t=0`} /> kills the signal the net needs to learn the initial acceleration (it drove
        <InlineMath tex={String.raw`\theta_1`} /> the <em>wrong way</em>). Soft IC is well-conditioned and the standard
        DeepXDE IVP approach. The network is a plain <InlineMath tex={String.raw`[1,\,96\times4,\,2]`} /> MLP with
        <strong>tanh</strong> activations (SIREN was unstable with soft IC here), trained Adam (25k) then L-BFGS, and
        exported to ONNX (parity <InlineMath tex={String.raw`\sim 1.6\times 10^{-6}`} />).
      </p>

      <h3>Honesty: chaos has a hard wall</h3>
      <p>
        A double pendulum is <strong>chaotic</strong>: no fixed network can follow the true trajectory past a finite
        horizon (the <em>Lyapunov horizon</em>). So the headline metric is NOT a long-term match but the
        <strong>leave-time</strong>: the first instant the PINN departs RK45 by more than
        <InlineMath tex={String.raw`0.30`} /> rad. Here it is a measured <strong>leave-time = 1.99 s</strong>: the PINN
        tracks RK45 to about <InlineMath tex={String.raw`0.02`} /> rad for nearly two seconds, then the trajectory peels
        away. The App shows the PINN (ghost) over RK45 (solid): it tracks for a while then loses it: and the PINN arm
        turns <span style={{ color: "#ff5d5d" }}>red</span> after the leave-time. The two orange curves (two starts
        <InlineMath tex={String.raw`10^{-2}`} /> rad apart) diverge exponentially: that is the limit, shown honestly, not
        hidden. Over the full <InlineMath tex={String.raw`[0,3]`} /> s window the trajectory error is
        <strong>9.3%</strong> (relative-L2), and the twin's exponential separation gives a crude largest-Lyapunov rate
        <InlineMath tex={String.raw`\lambda\approx 0.02`} /> 1/s, both consistent with a roughly two-second
        predictability horizon.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> the ideal frictionless, undamped double pendulum, point masses, massless rigid arms.
        <strong>Out of scope:</strong> friction, distributed-mass arms, forcing. Data is <em>synthetic-illustrative</em>
        (not a real measurement). The case is <em>didactic</em>: it shows how a PINN attacks a dynamical system (not a
        field) and where chaos defeats it.
      </p>

      <p>
        <strong>How to read the viz.</strong> The <strong>animated pendulum</strong> is the hero: RK45 solid (blue) +
        PINN ghost (dashed) + the lower-arm trail. The <strong>phase portrait</strong>
        <InlineMath tex={String.raw`\theta_1`} />-<InlineMath tex={String.raw`\theta_2`} /> traces the state-space path.
        The <strong>butterfly</strong> panel shows <InlineMath tex={String.raw`|\Delta\theta|`} /> of two nearby starts
        growing (log scale). The <strong>time-series</strong> compares PINN vs RK45 angles with the leave-time line.
        Use Play / the bar to scrub through time.
      </p>
    </>
  );
}
