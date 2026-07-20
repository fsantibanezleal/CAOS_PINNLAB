import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for ctrl-zero-source: the control case, baked as a parametric
 *  manufactured-solution (MMS) verification sweep that contains the degenerate zero field at a=0. */
export function ZeroSourceContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>el caso de control: ¿el solver recupera una respuesta conocida?</h2>
      <p>
        Todo solver de EDP necesita un <strong>caso de control</strong>: un problema
        cuya solución exacta conocemos de antemano, para medir el error real del método en lugar de confiar a ciegas
        en él. El control canónico es la ecuación de <strong>Poisson</strong> sobre el cuadrado unitario con borde
        cero, <InlineMath tex={String.raw`-\nabla^2 u = f`} /> en <InlineMath tex={String.raw`(0,1)^2`} />,
        <InlineMath tex={String.raw`\,u|_{\partial\Omega}=0`} />. Su miembro degenerado: 
        <strong> fuente cero, campo cero</strong> (<InlineMath tex={String.raw`f\equiv0\Rightarrow u\equiv0`} />): es
        la prueba de cordura mínima: el motor debe ejecutarse sin caer y devolver un campo plano en cero. Aquí lo
        extendemos a una <strong>familia paramétrica verificada</strong> por el <em>método de soluciones manufacturadas</em>
        (MMS): elegimos una solución exacta <InlineMath tex={String.raw`u^*`} />, derivamos la fuente
        <InlineMath tex={String.raw`f=-\nabla^2 u^*`} /> que la produce, y un solo deslizador
        <InlineMath tex={String.raw`a`} /> barre desde el control degenerado (<InlineMath tex={String.raw`a=0`} />)
        hasta una solución estructurada (<InlineMath tex={String.raw`a=1`} />).
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Ejes del campo:</strong> espacio <InlineMath tex={String.raw`x\in[0,1]`} /> × <InlineMath tex={String.raw`y\in[0,1]`} />, grilla <InlineMath tex={String.raw`41\times41`} />.</li>
        <li><strong>Incógnita:</strong> el potencial <InlineMath tex={String.raw`u(x,y)`} /> que satisface Poisson con borde cero.</li>
        <li><strong>Perilla de amplitud:</strong> <InlineMath tex={String.raw`a\in[0,1]`} />: input de la red. En <InlineMath tex={String.raw`a=0`} /> recupera el <em>control degenerado</em> exacto (fuente y campo nulos); al crecer, "enciende" la solución manufacturada.</li>
        <li><strong>Fuente impuesta:</strong> <InlineMath tex={String.raw`f(x,y;a)=-\nabla^2 u^*`} />: derivada en forma cerrada de la solución elegida (abajo).</li>
        <li><strong>Borde:</strong> <InlineMath tex={String.raw`u=0`} /> en <InlineMath tex={String.raw`\partial\Omega`} />, impuesto de forma <strong>dura</strong> (exacta) para todo <InlineMath tex={String.raw`a`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Fijamos una <strong>forma manufacturada</strong> de dos modos que se anula en todo el borde, y la escalamos por
        la amplitud <InlineMath tex={String.raw`a`} />:
      </p>
      <Equation tex={String.raw`u^*(x,y;a)=a\,g(x,y),\qquad g(x,y)=\sin(\pi x)\sin(\pi y)+\tfrac12\sin(2\pi x)\sin(2\pi y).`} />
      <p>
        Cada modo se anula en <InlineMath tex={String.raw`x\in\{0,1\}`} /> y <InlineMath tex={String.raw`y\in\{0,1\}`} />,
        así que <InlineMath tex={String.raw`u^*|_{\partial\Omega}=0`} /> para <em>todo</em>
        <InlineMath tex={String.raw`a`} />. La fuente que produce exactamente esta solución sale por sustitución
        directa <InlineMath tex={String.raw`(\nabla^2[\sin(k\pi x)\sin(m\pi y)]=-(k^2+m^2)\pi^2\sin(k\pi x)\sin(m\pi y))`} />:
      </p>
      <Equation tex={String.raw`f(x,y;a)=-\nabla^2 u^* = a\,\Big(2\pi^2\sin(\pi x)\sin(\pi y)+4\pi^2\sin(2\pi x)\sin(2\pi y)\Big).`} />
      <p>
        Como <InlineMath tex={String.raw`u^*`} /> resuelve <InlineMath tex={String.raw`-\nabla^2 u^*=f`} /> por
        construcción, para <em>cualquier</em> <InlineMath tex={String.raw`a`} />, el error relativo-L2 de la red frente
        a <InlineMath tex={String.raw`u^*`} /> es el <strong>error verdadero</strong>. En
        <InlineMath tex={String.raw`a=0`} /> ambos lados son cero: se recupera el control degenerado y el L2 colapsa a
        <InlineMath tex={String.raw`\lVert\hat u\rVert`} /> (la norma del residuo, idealmente diminuta).
      </p>

      <h3>El método: restricción dura de borde, una sola red paramétrica</h3>
      <p>
        El borde cero se incorpora de forma <strong>exacta</strong> en el ansatz: no hay término de pérdida de borde que
        compita:
      </p>
      <Equation tex={String.raw`u_\theta(x,y;a)=\underbrace{x(1-x)\,y(1-y)}_{=\,0\ \text{en }\partial\Omega}\;\mathcal{N}_\theta(x,y,a).`} />
      <p>
        La amplitud <InlineMath tex={String.raw`a`} /> es un <strong>input de la red</strong> (no un eje del campo), así
        que una sola red entrenada cubre toda la familia y el tab <strong>Live</strong> barre
        <InlineMath tex={String.raw`a`} /> de forma continua reusando el mismo ONNX. La red entrenada es un MLP tanh <InlineMath tex={String.raw`[3,48,48,48,48,1]`} /> (DeepXDE) sobre el hipercubo <InlineMath tex={String.raw`(x,y,a)`} />, optimizada con Adam (12000 pasos, lr 1e-3) y luego L-BFGS, anclada vía la solución analítica. La red solo aprende a igualar el
        residuo de Poisson en el interior; el borde ya es exacto. Este es el mismo motor de <em>restricciones duras</em>
        del caso original, ahora verificado a través de una familia.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> verificación tipo MMS del operador de Poisson con borde-cero duro, sobre una familia
        de amplitud que contiene el control degenerado en <InlineMath tex={String.raw`a=0`} />. Es
        <em> sintético</em> por naturaleza: un caso de control no representa datos físicos, su trabajo es <strong>medir
        la corrección del pipeline</strong> contra una verdad exacta. <strong>Fuera de alcance:</strong> dominios no
        cuadrados, bordes no homogéneos, coeficientes variables y fuentes no manufacturadas: todo eso vive en los casos
        de física propiamente tales. La forma <InlineMath tex={String.raw`g`} /> es fija; solo se barre su amplitud, así
        que la variedad es de <em>magnitud</em> (incluyendo el cero exacto), no de geometría.
      </p>

      <h3>Qué muestra cada variante</h3>
      <p>
        El barrido va de <InlineMath tex={String.raw`a=0`} /> (campo idénticamente cero: el control degenerado, un
        heatmap plano) subiendo hasta <InlineMath tex={String.raw`a=1`} />, donde aparece la estructura de dos modos:
        un lóbulo central dominante (modo fundamental) con una ondulación más fina superpuesta (segundo modo). Cada
        variante reporta su relative-L2 frente a la solución manufacturada exacta: el error <em>real</em> del solver
        en esa amplitud. Medido (seed 42) el relative-L2 se mantiene &le; 0.15% para <InlineMath tex={String.raw`a\ge0.2`} />,
        y en <InlineMath tex={String.raw`a=0`} /> la métrica colapsa a la norma del campo
        (<InlineMath tex={String.raw`\lVert\text{pred}\rVert=2.1\%`} />, esencialmente plano en cero); el ONNX en el
        navegador iguala a la red entrenada hasta 4.2e-7. Que el error se mantenga diminuto a lo largo de toda la familia (y que el campo sea
        exactamente plano en <InlineMath tex={String.raw`a=0`} />) es justo lo que un caso de control debe certificar.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de
        <InlineMath tex={String.raw`u(x,y)`} /> muestra los lóbulos manufacturados; al pasar el cursor se lee el valor
        exacto para compararlo mentalmente con <InlineMath tex={String.raw`u^*=a\,g`} />. Los <strong>perfiles de corte</strong>
        en <InlineMath tex={String.raw`x`} /> e <InlineMath tex={String.raw`y`} /> deben trazar senos suaves anclados a
        cero en los bordes. En <strong>Live</strong>, al arrastrar el deslizador de amplitud
        <InlineMath tex={String.raw`a`} /> hacia <InlineMath tex={String.raw`0`} /> el campo se desvanece a
        cero plano: el control degenerado emergiendo como límite de la familia, re-evaluado en el navegador
        (onnxruntime-web).
      </p>
    </>
  ): (
    <>
      <h2>the control case: does the solver recover a known answer?</h2>
      <p>
        Every PDE solver needs a <strong>control case</strong>: a problem whose exact
        solution we know in advance, so we can measure the method's true error instead of trusting it blindly. The
        canonical control is the <strong>Poisson</strong> equation on the unit square with zero boundary,
        <InlineMath tex={String.raw`-\nabla^2 u = f`} /> on <InlineMath tex={String.raw`(0,1)^2`} />,
        <InlineMath tex={String.raw`\,u|_{\partial\Omega}=0`} />. Its degenerate member: 
        <strong> zero source, zero field</strong> (<InlineMath tex={String.raw`f\equiv0\Rightarrow u\equiv0`} />): is
        the minimal sanity check: the engine must run without crashing and return a flat-zero field. Here we extend it
        into a <strong>verified parametric family</strong> via the <em>method of manufactured solutions</em> (MMS): we
        pick an exact solution <InlineMath tex={String.raw`u^*`} />, derive the source
        <InlineMath tex={String.raw`f=-\nabla^2 u^*`} /> that produces it, and a single slider
        <InlineMath tex={String.raw`a`} /> sweeps from the degenerate control (<InlineMath tex={String.raw`a=0`} />) to
        a structured solution (<InlineMath tex={String.raw`a=1`} />).
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Field axes:</strong> space <InlineMath tex={String.raw`x\in[0,1]`} /> × <InlineMath tex={String.raw`y\in[0,1]`} />, a <InlineMath tex={String.raw`41\times41`} /> grid.</li>
        <li><strong>Unknown:</strong> the potential <InlineMath tex={String.raw`u(x,y)`} /> satisfying Poisson with zero boundary.</li>
        <li><strong>Amplitude knob:</strong> <InlineMath tex={String.raw`a\in[0,1]`} />: a network input. At <InlineMath tex={String.raw`a=0`} /> it recovers the exact <em>degenerate control</em> (null source and field); as it grows it "switches on" the manufactured solution.</li>
        <li><strong>Imposed source:</strong> <InlineMath tex={String.raw`f(x,y;a)=-\nabla^2 u^*`} />: the closed-form derivative of the chosen solution (below).</li>
        <li><strong>Boundary:</strong> <InlineMath tex={String.raw`u=0`} /> on <InlineMath tex={String.raw`\partial\Omega`} />, imposed as a <strong>hard</strong> (exact) constraint for every <InlineMath tex={String.raw`a`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        We fix a two-mode <strong>manufactured shape</strong> that vanishes on the entire boundary, and scale it by the
        amplitude <InlineMath tex={String.raw`a`} />:
      </p>
      <Equation tex={String.raw`u^*(x,y;a)=a\,g(x,y),\qquad g(x,y)=\sin(\pi x)\sin(\pi y)+\tfrac12\sin(2\pi x)\sin(2\pi y).`} />
      <p>
        Each mode vanishes at <InlineMath tex={String.raw`x\in\{0,1\}`} /> and <InlineMath tex={String.raw`y\in\{0,1\}`} />,
        so <InlineMath tex={String.raw`u^*|_{\partial\Omega}=0`} /> for <em>all</em> <InlineMath tex={String.raw`a`} />.
        The source producing exactly this solution follows by direct substitution
        <InlineMath tex={String.raw`(\nabla^2[\sin(k\pi x)\sin(m\pi y)]=-(k^2+m^2)\pi^2\sin(k\pi x)\sin(m\pi y))`} />:
      </p>
      <Equation tex={String.raw`f(x,y;a)=-\nabla^2 u^* = a\,\Big(2\pi^2\sin(\pi x)\sin(\pi y)+4\pi^2\sin(2\pi x)\sin(2\pi y)\Big).`} />
      <p>
        Since <InlineMath tex={String.raw`u^*`} /> solves <InlineMath tex={String.raw`-\nabla^2 u^*=f`} /> by
        construction, for <em>any</em> <InlineMath tex={String.raw`a`} />, the network's relative-L2 against
        <InlineMath tex={String.raw`u^*`} /> is the <strong>true error</strong>. At
        <InlineMath tex={String.raw`a=0`} /> both sides are zero: the degenerate control is recovered and the L2
        collapses to <InlineMath tex={String.raw`\lVert\hat u\rVert`} /> (the residual norm, ideally tiny).
      </p>

      <h3>The method: hard boundary constraint, one parametric network</h3>
      <p>
        The zero boundary is baked <strong>exactly</strong> into the ansatz: there is no competing boundary loss term:
      </p>
      <Equation tex={String.raw`u_\theta(x,y;a)=\underbrace{x(1-x)\,y(1-y)}_{=\,0\ \text{on }\partial\Omega}\;\mathcal{N}_\theta(x,y,a).`} />
      <p>
        The amplitude <InlineMath tex={String.raw`a`} /> is a <strong>network input</strong> (not a field axis), so one
        trained network covers the whole family and the <strong>Live</strong> tab sweeps
        <InlineMath tex={String.raw`a`} /> continuously by reusing the same ONNX. The trained net is a tanh MLP <InlineMath tex={String.raw`[3,48,48,48,48,1]`} /> (DeepXDE) over the hypercube <InlineMath tex={String.raw`(x,y,a)`} />, optimized with Adam (12000 steps, lr 1e-3) then L-BFGS, anchored via the analytic solution. The network only learns to match the
        Poisson residual in the interior; the boundary is already exact. This is the same <em>hard-constraints</em>
        engine as the original case, now verified across a family.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> an MMS verification of the Poisson operator with a hard zero boundary, over an
        amplitude family that contains the degenerate control at <InlineMath tex={String.raw`a=0`} />. It is
        <em> synthetic</em> by nature: a control case represents no physical data: its job is to <strong>measure the
        pipeline's correctness</strong> against an exact truth. <strong>Out of scope:</strong> non-square domains,
        inhomogeneous boundaries, variable coefficients, and non-manufactured sources: those live in the physics cases
        proper. The shape <InlineMath tex={String.raw`g`} /> is fixed; only its amplitude is swept, so the variety is in
        <em> magnitude</em> (including the exact zero), not geometry.
      </p>

      <h3>What each variant shows</h3>
      <p>
        The sweep runs from <InlineMath tex={String.raw`a=0`} /> (an identically-zero field: the degenerate control, a
        flat heatmap) up to <InlineMath tex={String.raw`a=1`} />, where the two-mode structure appears: a dominant
        central lobe (the fundamental mode) with a finer ripple superimposed (the second mode). Each variant reports its
        relative-L2 against the exact manufactured solution: the solver's <em>real</em> error at that amplitude. Measured (seed 42) the relative-L2 stays &le; 0.15% for <InlineMath tex={String.raw`a\ge0.2`} />,
        and at <InlineMath tex={String.raw`a=0`} /> the metric collapses to the field norm
        (<InlineMath tex={String.raw`\lVert\text{pred}\rVert=2.1\%`} />, essentially flat zero); the in-browser ONNX
        matches the trained network to 4.2e-7. That
        the error stays tiny across the whole family (and that the field is exactly flat at
        <InlineMath tex={String.raw`a=0`} />) is precisely what a control case must certify.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`u(x,y)`} /> shows the manufactured lobes; hover to read the exact value and compare
        it mentally to <InlineMath tex={String.raw`u^*=a\,g`} />. The <strong>line-cut profiles</strong> in
        <InlineMath tex={String.raw`x`} /> and <InlineMath tex={String.raw`y`} /> should trace smooth sines pinned to
        zero at the edges. In <strong>Live</strong>, drag the amplitude slider
        <InlineMath tex={String.raw`a`} /> toward <InlineMath tex={String.raw`0`} /> and watch the field fade to flat
        zero: the degenerate control emerging as the limit of the family, re-evaluated in the browser
        (onnxruntime-web).
      </p>
    </>
  );
}
