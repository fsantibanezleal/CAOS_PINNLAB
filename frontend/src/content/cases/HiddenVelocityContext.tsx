import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for ind-hidden-velocity (the HFM mechanism: estimate the hidden current from
 *  sparse passive-scalar observations; the flagship of the estimation reframe, issue #48). */
export function HiddenVelocityContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: solo puedes ver el tinte. ¿Cuál es la corriente debajo?</h2>
      <p>
        <strong>El problema.</strong> En un flujo real casi nunca se mide la velocidad directamente: se ve lo que el
        flujo <em>arrastra</em>: un tinte, humo, un contraste, sedimento. Este caso reproduce el mecanismo de{" "}
        <em>Hidden Fluid Mechanics</em> (Raissi, Yazdani y Karniadakis, Science 2020): estimar el campo de velocidad{" "}
        <InlineMath tex={String.raw`\mathbf{u}(x,y)`} /> completo a partir de <strong>solo</strong> muestras dispersas
        y ruidosas de la concentración del tinte <InlineMath tex={String.raw`c(x,y,t)`} />, usando la física del
        transporte como instrumento. La red <strong>nunca ve un dato de velocidad</strong>, ni condiciones iniciales
        ni de borde de <InlineMath tex={String.raw`c`} />: esa ausencia es el punto (el paper es explícitamente
        &ldquo;agnóstico a la geometría y a las condiciones iniciales y de borde&rdquo;).
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Verdad de flujo (cerrada):</strong> la celda de vórtice incompresible <InlineMath tex={String.raw`\mathbf{u}^*=A(\sin\pi x\cos\pi y,\ -\cos\pi x\sin\pi y)`} />, <InlineMath tex={String.raw`A=1.5`} />: divergencia cero por construcción y paredes que son líneas de corriente.</li>
        <li><strong>Verdad del tinte (numérica):</strong> un parche gaussiano (centro <InlineMath tex={String.raw`(0.5,\,0.2)`} />) advectado-difundido por ese flujo con <InlineMath tex={String.raw`D=0.02`} />, resuelto por diferencias finitas con estabilidad VERIFICADA (Péclet de celda 0.59 &lt; 2, dt seguro por CFL, masa conservada, acotación afirmada): nunca se hornea una referencia numérica sin sus chequeos.</li>
        <li><strong>Datos (lo ÚNICO observado):</strong> ~640 muestras espacio-temporales de <InlineMath tex={String.raw`c`} /> con ruido gaussiano (0.5% del máximo); otras 160 quedan RETENIDAS y nunca entrenan (validación fuera de muestra).</li>
        <li><strong>Incógnita (objetivo):</strong> el campo de velocidad completo <InlineMath tex={String.raw`(u,v)`} />: dos campos ocultos que jamás se midieron.</li>
      </ul>

      <h3>Formalización</h3>
      <Equation tex={String.raw`c_t+\mathbf{u}\cdot\nabla c = D\,\nabla^2 c,\qquad \nabla\cdot\mathbf{u}=0,`} />
      <p>
        con <InlineMath tex={String.raw`D`} /> conocida (como en HFM, donde la viscosidad se asume o se aprende). La red{" "}
        <InlineMath tex={String.raw`\mathcal{N}_\theta(x,y,t)=(u,v,c)`} /> emite el tinte <em>y</em> la velocidad a la
        vez, y la pérdida acopla tres términos: el residual de transporte, el residual de incompresibilidad y los
        datos de tinte:
      </p>
      <Equation tex={String.raw`\mathcal{L}=\big\|c_t+u\,c_x+v\,c_y-D\nabla^2c\big\|^2+\big\|u_x+v_y\big\|^2+\big\|u_t\big\|^2+\big\|v_t\big\|^2+\lambda\sum_i\big(c_\theta(x_i,y_i,t_i)-c_i^{\text{obs}}\big)^2.`} />
      <p>
        El transporte conecta lo visible con lo oculto: donde el tinte tiene gradiente, el término{" "}
        <InlineMath tex={String.raw`\mathbf{u}\cdot\nabla c`} /> obliga a la velocidad a explicar cómo se mueve la
        mancha; la incompresibilidad propaga esa información a lo largo de las líneas de corriente. Los residuales{" "}
        <InlineMath tex={String.raw`u_t=v_t=0`} /> codifican el supuesto declarado de <strong>corriente
        estacionaria</strong>, y son estructurales: sin ellos (medido, no supuesto) la red gastaba su libertad en una
        velocidad variable en el tiempo que ~640 muestras de tinte no pueden fijar, y la corriente recuperada quedaba
        38-60% desviada incluso en la región barrida. Declarar lo que legítimamente se sabe (una corriente
        cuasi-estacionaria en la ventana) agrega la información de TODOS los tiempos en UN campo.
      </p>

      <h3>La honestidad: identificabilidad barrida por el tinte</h3>
      <p>
        Donde el tinte nunca pasó, <InlineMath tex={String.raw`\nabla c\approx 0`} /> y el transporte NO restringe a{" "}
        <InlineMath tex={String.raw`\mathbf{u}`} />: la velocidad es <strong>no identificable</strong> allí, para
        cualquier método, no solo para los PINN. Por eso el caso hornea la <strong>máscara barrida</strong> (donde{" "}
        <InlineMath tex={String.raw`\max_t|\nabla c|`} /> supera un umbral, ~2/3 del dominio) y publica el error en dos
        números separados: dentro de la región barrida y en las zonas muertas. El L2 global se publica tal cual,
        aunque las zonas muertas lo inflen: nunca se maquilla.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> flujo estacionario 2D incompresible, un escalar pasivo, D conocida, ventana{" "}
        <InlineMath tex={String.raw`t\in[0,1]`} />. Es <em>sintético</em> con verdad cerrada (la velocidad) y numérica
        verificada (el tinte): eso permite puntuar la recuperación exactamente, cosa imposible con un flujo real.
        <strong> Fuera de alcance:</strong> flujo no estacionario o turbulento, presión (aquí no hace falta: la
        velocidad no se acopla a Navier-Stokes sino al transporte), tinte reactivo, y D desconocida (una extensión
        natural: D como variable entrenable).
      </p>

      <p>
        <strong>Cómo leer y usar la viz.</strong> El panel 1 es el producto: la <strong>corriente recuperada</strong>{" "}
        con los puntos de tinte que son TODA la evidencia; compárala con la verdadera (panel 2, misma escala) y mira
        el <strong>mapa de error</strong> contra la <strong>máscara barrida</strong>: el error vive donde el tinte no
        pasó. El tab <strong>Live</strong> barre el tiempo con la ONNX entrenada; los chips de régimen fijan el t del
        tinte reconstruido.
      </p>
    </>
  ) : (
    <>
      <h2>The problem: you can only see the dye. What is the current underneath?</h2>
      <p>
        <strong>The problem.</strong> In a real flow you almost never measure velocity directly: you see what the flow{" "}
        <em>carries</em>: a dye, smoke, a contrast agent, sediment. This case reproduces the{" "}
        <em>Hidden Fluid Mechanics</em> mechanism (Raissi, Yazdani &amp; Karniadakis, Science 2020): estimate the whole
        velocity field <InlineMath tex={String.raw`\mathbf{u}(x,y)`} /> from <strong>only</strong> sparse noisy samples
        of the dye concentration <InlineMath tex={String.raw`c(x,y,t)`} />, using the transport physics as the
        instrument. The network <strong>never sees a velocity datum</strong>, nor initial or boundary conditions on{" "}
        <InlineMath tex={String.raw`c`} />: that absence is the point (the paper is explicitly &ldquo;agnostic to the
        geometry or the initial and boundary conditions&rdquo;).
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Flow truth (closed form):</strong> the incompressible vortex cell <InlineMath tex={String.raw`\mathbf{u}^*=A(\sin\pi x\cos\pi y,\ -\cos\pi x\sin\pi y)`} />, <InlineMath tex={String.raw`A=1.5`} />: divergence-free by construction, walls are streamlines.</li>
        <li><strong>Dye truth (numerical):</strong> a Gaussian patch (center <InlineMath tex={String.raw`(0.5,\,0.2)`} />) advected-diffused by that flow with <InlineMath tex={String.raw`D=0.02`} />, solved by finite differences with VERIFIED stability (cell Péclet 0.59 &lt; 2, CFL-safe dt, mass conserved, boundedness asserted): a numerical reference is never baked without its checks.</li>
        <li><strong>Data (the ONLY thing observed):</strong> ~640 space-time samples of <InlineMath tex={String.raw`c`} /> with Gaussian noise (0.5% of max); another 160 are HELD OUT and never train (out-of-sample validation).</li>
        <li><strong>Unknown (the target):</strong> the full velocity field <InlineMath tex={String.raw`(u,v)`} />: two hidden fields that were never measured.</li>
      </ul>

      <h3>Formalization</h3>
      <Equation tex={String.raw`c_t+\mathbf{u}\cdot\nabla c = D\,\nabla^2 c,\qquad \nabla\cdot\mathbf{u}=0,`} />
      <p>
        with <InlineMath tex={String.raw`D`} /> known (as in HFM, where viscosity is assumed or learned). The network{" "}
        <InlineMath tex={String.raw`\mathcal{N}_\theta(x,y,t)=(u,v,c)`} /> emits the dye <em>and</em> the velocity at
        once, and the loss couples three terms: the transport residual, the incompressibility residual, and the dye
        data:
      </p>
      <Equation tex={String.raw`\mathcal{L}=\big\|c_t+u\,c_x+v\,c_y-D\nabla^2c\big\|^2+\big\|u_x+v_y\big\|^2+\big\|u_t\big\|^2+\big\|v_t\big\|^2+\lambda\sum_i\big(c_\theta(x_i,y_i,t_i)-c_i^{\text{obs}}\big)^2.`} />
      <p>
        Transport is what couples the visible to the hidden: wherever the dye has a gradient, the{" "}
        <InlineMath tex={String.raw`\mathbf{u}\cdot\nabla c`} /> term forces the velocity to explain how the patch
        moves; incompressibility propagates that information along streamlines. The{" "}
        <InlineMath tex={String.raw`u_t=v_t=0`} /> residuals encode the stated <strong>steady-flow assumption</strong>,
        and they are load-bearing: without them (measured, not guessed) the net spent its freedom on a time-varying
        velocity that ~640 sparse dye samples cannot pin, and the recovered current was 38-60% off even inside the
        swept region. Asserting what you legitimately know (a quasi-steady current over the window) aggregates the
        dye information from ALL times into ONE field.
      </p>

      <h3>The honesty: dye-swept identifiability</h3>
      <p>
        Where dye never passed, <InlineMath tex={String.raw`\nabla c\approx 0`} /> and transport does NOT constrain{" "}
        <InlineMath tex={String.raw`\mathbf{u}`} />: the velocity is <strong>unidentifiable</strong> there, for any
        method, not just PINNs. So the case bakes the <strong>swept mask</strong> (where{" "}
        <InlineMath tex={String.raw`\max_t|\nabla c|`} /> exceeds a threshold, ~2/3 of the domain) and publishes the
        error as two separate numbers: inside the swept region and in the dead zones. The global L2 is published
        as-is even though the dead zones inflate it: never dressed up.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> steady 2D incompressible flow, a passive scalar, known D, window{" "}
        <InlineMath tex={String.raw`t\in[0,1]`} />. It is <em>synthetic</em> with a closed-form truth (the velocity)
        and a verified numerical truth (the dye): that is what makes the recovery exactly scoreable, which no real
        flow allows. <strong>Out of scope:</strong> unsteady or turbulent flow, pressure (not needed here: the
        velocity couples to transport, not to Navier-Stokes), reactive dye, and unknown D (a natural extension: D as
        a trainable variable).
      </p>

      <p>
        <strong>How to read &amp; use the viz.</strong> Panel 1 is the product: the <strong>recovered current</strong>{" "}
        with the dye points that are ALL the evidence; compare it against the true current (panel 2, same color
        scale) and read the <strong>error map</strong> against the <strong>swept mask</strong>: the error lives where
        the dye never went. The <strong>Live</strong> tab sweeps time through the trained ONNX; the regime chips set
        the t of the reconstructed dye.
      </p>
    </>
  );
}
