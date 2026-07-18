import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-burgers1d (1D viscous Burgers, parametric viscosity, traveling-shock family). */
export function Burgers1dContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: un frente de choque que se mueve: Burgers viscoso 1D con viscosidad ajustable</h2>
      <p>
        <strong>El problema.</strong> La ecuación de Burgers <InlineMath tex={String.raw`u_t + u\,u_x = \nu\,u_{xx}`} />
        es el modelo no lineal más simple que combina <em>advección</em> (el término <InlineMath tex={String.raw`u\,u_x`} />,
        que empina los perfiles hasta formar choques) con <em>difusión</em> (el término <InlineMath tex={String.raw`\nu\,u_{xx}`} />,
        que los suaviza). Es el banco de pruebas canónico de la dinámica de fluidos y del tráfico: la competencia
        advección-difusión produce un <strong>frente de choque viscoso</strong> de grosor finito. Aquí la
        <strong> viscosidad</strong> <InlineMath tex={String.raw`\nu`} /> es ajustable: una sola red aprende toda la
        familia <InlineMath tex={String.raw`u(x,t;\nu)`} />, y en el tab <strong>Live</strong> mover el deslizador de
        <InlineMath tex={String.raw`\nu`} /> hace que el choque se afile (poca viscosidad) o se difumine (mucha).
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> espacio <InlineMath tex={String.raw`x\in[-1,1]`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`241\times121`} />.</li>
        <li><strong>Incógnita:</strong> el campo <InlineMath tex={String.raw`u(x,t)`} /> (p. ej. velocidad).</li>
        <li><strong>Parámetro de control:</strong> la <em>viscosidad</em> <InlineMath tex={String.raw`\nu\in[0.02,0.08]`} />: un input de la red. Fija el <em>grosor</em> del frente (<InlineMath tex={String.raw`\sim 4\nu`} />).</li>
        <li><strong>Estados izquierdo/derecho:</strong> <InlineMath tex={String.raw`u_L=1`} /> y <InlineMath tex={String.raw`u_R=0`} />, de modo que <InlineMath tex={String.raw`\Delta=u_L-u_R=1`} /> y la velocidad del choque es <InlineMath tex={String.raw`s=\tfrac{u_L+u_R}{2}=\tfrac12`} /> (Rankine-Hugoniot).</li>
        <li><strong>Condiciones:</strong> el perfil inicial es el frente <InlineMath tex={String.raw`\tanh`} /> centrado en <InlineMath tex={String.raw`x_0=-0.4`} />; los bordes quedan fijos en <InlineMath tex={String.raw`u_L,u_R`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Burgers admite una <strong>solución de onda viajera</strong> exacta: un frente <InlineMath tex={String.raw`\tanh`} />
        que se traslada sin cambiar de forma. Es válida para <em>cualquier</em> <InlineMath tex={String.raw`\nu`} /> y es
        nuestra <strong>ancla de validación</strong> (Whitham, <em>Linear and Nonlinear Waves</em>):
      </p>
      <Equation tex={String.raw`u^*(x,t;\nu)=s-\frac{\Delta}{2}\,\tanh\!\Big(k\,(x-x_0-s\,t)\Big),\qquad k=\frac{\Delta}{4\nu}.`} />
      <p>
        Al sustituir en la EDP, el ancho del frente queda fijado por <InlineMath tex={String.raw`k=\Delta/4\nu`} />: la
        difusión <InlineMath tex={String.raw`\nu`} /> equilibra exactamente al empinamiento advectivo. La PINN
        <InlineMath tex={String.raw`u_\theta(x,t,\nu)`} /> minimiza el residual de Burgers en puntos de colocación, y la
        IC + ambas BC de Dirichlet se imponen de forma <strong>exacta</strong> por una restricción dura:
      </p>
      <Equation tex={String.raw`u_\theta = g(x;\nu) + t\,(1-x^2)\,\mathcal{N}_\theta(x,t,\nu),\qquad g(x;\nu)=u^*(x,0;\nu).`} />
      <p>
        En <InlineMath tex={String.raw`t=0`} /> queda <InlineMath tex={String.raw`g`} /> (la IC, el frente inicial); en
        <InlineMath tex={String.raw`x=\pm1`} /> el factor <InlineMath tex={String.raw`(1-x^2)`} /> se anula y queda
        <InlineMath tex={String.raw`g(\pm1)=u_L,u_R`} /> (las BC, constantes en el tiempo porque el frente se mantiene
        interior). La red solo aprende la <em>traslación interior</em>.
      </p>

      <h3>El método: hard-constraints + RAR</h3>
      <p>
        El reto numérico es el <em>frente afilado</em>: con poca viscosidad el residual se concentra en una banda
        delgada que una grilla uniforme resuelve mal. Sobre la base anterior aplicamos <strong>RAR</strong>
        (<em>residual-based adaptive refinement</em>, Wu et al., CMAME 2023): tras el ajuste base, se evalúa el residual
        en un gran pozo de puntos y se <strong>añaden</strong> los de mayor error: que caen sobre el frente móvil:
        repitiendo varias rondas. Así el choque se resuelve sin densificar todo el dominio. Como el ajuste base
        <InlineMath tex={String.raw`g(x;\nu)`} /> ya carga el frente exacto, la red solo aprende la pequeña traslación
        interior, por lo que la precisión es <strong>mejor en el frente más agudo</strong>: la L2 relativa medida es
        0.08% en <InlineMath tex={String.raw`\nu=0.02`} /> y sube a 1.2% en <InlineMath tex={String.raw`\nu=0.08`} />
        (semilla 42, todas dentro de la banda <InlineMath tex={String.raw`<2\text{e-}2`} />), y la exportación ONNX
        coincide con la red entrenada a 6.5e-7 en máximo absoluto, por eso el tab Live reproduce el campo con exactitud.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> Burgers viscoso 1D, viscosidad constante (paramétrica en
        <InlineMath tex={String.raw`\nu`} />), familia de onda viajera de un solo frente. La restricción dura sobrevive
        a la exportación a ONNX, por eso el tab Live re-evalúa el campo exacto. <strong>Fuera de alcance:</strong> el
        límite no viscoso <InlineMath tex={String.raw`\nu\to0`} /> (choque verdadero, sin grosor), IC de múltiples
        frentes o interacción de choques, y Burgers estocástico. La famosa IC senoidal de Raissi
        (<InlineMath tex={String.raw`u(x,0)=-\sin\pi x`} />, <InlineMath tex={String.raw`\nu=0.01/\pi`} />) no tiene
        forma cerrada paramétrica, por eso aquí usamos la familia de onda viajera, que sí la tiene.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> El barrido de viscosidad recorre la competencia advección-difusión:
        <em>ν=0.02</em> (agudo): el frente es una capa finísima (ancho <InlineMath tex={String.raw`\sim0.08`} />);
        <em>ν=0.03/0.04/0.05</em> lo ensanchan progresivamente; <em>ν=0.06</em> y <em>ν=0.08</em> (difuso) dan una
        rampa ancha (ancho <InlineMath tex={String.raw`\sim0.32`} />). En todas, el frente <strong>viaja a la misma
        velocidad</strong> <InlineMath tex={String.raw`s=\tfrac12`} /> (la viscosidad cambia el grosor, no la
        velocidad).
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`u(x,t)`} />
        (x horizontal, t vertical) muestra una <em>banda diagonal</em>: el frente que avanza hacia la derecha; su
        <em>inclinación</em> es la velocidad <InlineMath tex={String.raw`s`} /> y su <em>nitidez</em> es la viscosidad.
        Al pasar el cursor se lee el valor exacto y se muestra el <strong>perfil de corte</strong> en
        <InlineMath tex={String.raw`x`} /> (la forma <InlineMath tex={String.raw`\tanh`} /> del frente) y en
        <InlineMath tex={String.raw`t`} /> (cómo pasa el frente por un punto fijo). Los <strong>chips</strong> cargan
        cada viscosidad; en <strong>Live</strong>, al deslizar <InlineMath tex={String.raw`\nu`} /> el frente se afila
        o se difumina en vivo en el navegador (onnxruntime-web).
      </p>
    </>
  ): (
    <>
      <h2>The problem: a moving shock front: 1D viscous Burgers with a tunable viscosity</h2>
      <p>
        <strong>The problem.</strong> The Burgers equation <InlineMath tex={String.raw`u_t + u\,u_x = \nu\,u_{xx}`} />
        is the simplest nonlinear model that pits <em>advection</em> (the <InlineMath tex={String.raw`u\,u_x`} /> term,
        which steepens profiles into shocks) against <em>diffusion</em> (the <InlineMath tex={String.raw`\nu\,u_{xx}`} />
        term, which smooths them). It is the canonical test bed of fluid dynamics and traffic flow: the
        advection-diffusion balance produces a <strong>viscous shock front</strong> of finite thickness. Here the
        <strong> viscosity</strong> <InlineMath tex={String.raw`\nu`} /> is tunable: a single network learns the whole
        family <InlineMath tex={String.raw`u(x,t;\nu)`} />, and in the <strong>Live</strong> tab moving the
        <InlineMath tex={String.raw`\nu`} /> slider makes the shock sharpen (small viscosity) or smear (large).
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> space <InlineMath tex={String.raw`x\in[-1,1]`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, a <InlineMath tex={String.raw`241\times121`} /> field grid.</li>
        <li><strong>Unknown:</strong> the field <InlineMath tex={String.raw`u(x,t)`} /> (e.g. velocity).</li>
        <li><strong>Control parameter:</strong> the <em>viscosity</em> <InlineMath tex={String.raw`\nu\in[0.02,0.08]`} />: a network input. It sets the front <em>thickness</em> (<InlineMath tex={String.raw`\sim 4\nu`} />).</li>
        <li><strong>Left/right states:</strong> <InlineMath tex={String.raw`u_L=1`} /> and <InlineMath tex={String.raw`u_R=0`} />, so <InlineMath tex={String.raw`\Delta=u_L-u_R=1`} /> and the shock speed is <InlineMath tex={String.raw`s=\tfrac{u_L+u_R}{2}=\tfrac12`} /> (Rankine-Hugoniot).</li>
        <li><strong>Conditions:</strong> the initial profile is the <InlineMath tex={String.raw`\tanh`} /> front centred at <InlineMath tex={String.raw`x_0=-0.4`} />; the ends are held at <InlineMath tex={String.raw`u_L,u_R`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        Burgers admits an exact <strong>traveling-wave solution</strong>: a <InlineMath tex={String.raw`\tanh`} /> front
        that translates without changing shape. It is valid for <em>any</em> <InlineMath tex={String.raw`\nu`} /> and is
        our <strong>validation anchor</strong> (Whitham, <em>Linear and Nonlinear Waves</em>):
      </p>
      <Equation tex={String.raw`u^*(x,t;\nu)=s-\frac{\Delta}{2}\,\tanh\!\Big(k\,(x-x_0-s\,t)\Big),\qquad k=\frac{\Delta}{4\nu}.`} />
      <p>
        Substituting into the PDE fixes the front width through <InlineMath tex={String.raw`k=\Delta/4\nu`} />:
        diffusion <InlineMath tex={String.raw`\nu`} /> exactly balances the advective steepening. The PINN
        <InlineMath tex={String.raw`u_\theta(x,t,\nu)`} /> minimises the Burgers residual at collocation points, and the
        IC + both Dirichlet BCs are imposed <strong>exactly</strong> by a hard constraint:
      </p>
      <Equation tex={String.raw`u_\theta = g(x;\nu) + t\,(1-x^2)\,\mathcal{N}_\theta(x,t,\nu),\qquad g(x;\nu)=u^*(x,0;\nu).`} />
      <p>
        At <InlineMath tex={String.raw`t=0`} /> it leaves <InlineMath tex={String.raw`g`} /> (the IC, the initial
        front); at <InlineMath tex={String.raw`x=\pm1`} /> the factor <InlineMath tex={String.raw`(1-x^2)`} /> vanishes
        leaving <InlineMath tex={String.raw`g(\pm1)=u_L,u_R`} /> (the BCs, time-constant because the front stays
        interior). The network only learns the <em>interior translation</em>.
      </p>

      <h3>The method: hard constraints + RAR</h3>
      <p>
        The numerical challenge is the <em>sharp front</em>: at low viscosity the residual concentrates in a thin band
        that a uniform grid resolves poorly. On top of the baseline above we apply <strong>RAR</strong>
        (<em>residual-based adaptive refinement</em>, Wu et al., CMAME 2023): after the base fit, the residual is
        evaluated over a large pool of points and the highest-error ones: which land on the moving front: are
        <strong>added</strong>, over several rounds. The shock is resolved without densifying the whole domain. Because
        the base fit <InlineMath tex={String.raw`g(x;\nu)`} /> already carries the exact front, the network only has to
        learn the small interior translation, so accuracy is <strong>best at the sharpest front</strong>: measured
        relative-L2 is 0.08% at <InlineMath tex={String.raw`\nu=0.02`} /> and rises to 1.2% at
        <InlineMath tex={String.raw`\nu=0.08`} /> (seed 42, all variants inside the <InlineMath tex={String.raw`<2\text{e-}2`} />
        band), and the ONNX export matches the trained net to 6.5e-7 max abs, which is why the Live tab reproduces the
        field exactly.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> 1D viscous Burgers, constant viscosity (parametric in
        <InlineMath tex={String.raw`\nu`} />), a single-front traveling-wave family. The hard constraint survives the
        ONNX export, which is why the Live tab re-evaluates the exact field. <strong>Out of scope:</strong> the inviscid
        limit <InlineMath tex={String.raw`\nu\to0`} /> (a true, thickness-less shock), multi-front ICs or shock
        interaction, and stochastic Burgers. Raissi's famous sinusoidal IC
        (<InlineMath tex={String.raw`u(x,0)=-\sin\pi x`} />, <InlineMath tex={String.raw`\nu=0.01/\pi`} />) has no
        closed parametric form, which is why we use the traveling-wave family here, which does.
      </p>

      <p>
        <strong>What each variant shows.</strong> The viscosity sweep walks the advection-diffusion balance:
        <em>ν=0.02</em> (sharp): the front is a razor-thin layer (width <InlineMath tex={String.raw`\sim0.08`} />);
        <em>ν=0.03/0.04/0.05</em> broaden it progressively; <em>ν=0.06</em> and <em>ν=0.08</em> (diffuse) give a wide
        ramp (width <InlineMath tex={String.raw`\sim0.32`} />). In all of them the front <strong>travels at the same
        speed</strong> <InlineMath tex={String.raw`s=\tfrac12`} /> (viscosity changes the thickness, not the speed).
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`u(x,t)`} /> (x horizontal, t vertical) shows a <em>diagonal band</em>: the front
        advancing to the right; its <em>slope</em> is the speed <InlineMath tex={String.raw`s`} /> and its
        <em> sharpness</em> is the viscosity. Hover to read the exact value and watch the <strong>line-cut profile</strong>
        in <InlineMath tex={String.raw`x`} /> (the <InlineMath tex={String.raw`\tanh`} /> front shape) and in
        <InlineMath tex={String.raw`t`} /> (how the front passes a fixed point). The <strong>chips</strong> load each
        viscosity; in <strong>Live</strong>, slide <InlineMath tex={String.raw`\nu`} /> and watch the front sharpen or
        smear live in the browser (onnxruntime-web).
      </p>
    </>
  );
}
