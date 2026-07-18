import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for mine-thickener-settling (Bürger-Concha degenerate settling, parametric descent rate). */
export function ThickenerContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: una interfase de barro que desciende: sedimentación de espesadores y relaves</h2>
      <p>
        <strong>El problema.</strong> En un espesador (o en una columna de relaves) una suspensión floculada
        <em> sedimenta</em>: las partículas caen, el sobrenadante clarifica arriba y un <em>lecho consolidado</em> crece
        abajo. Entre ambos baja una <strong>interfase de barro</strong> (mud-line) afilada. La fracción volumétrica de
        sólidos <InlineMath tex={String.raw`\phi(z,t)`} /> obedece a una ley de conservación
        <em> convección-difusión fuertemente degenerada</em> de Bürger-Concha:
        <InlineMath tex={String.raw`\phi_t + \partial_z f_{bk}(\phi) = \partial_z(D(\phi)\,\phi_z)`} />. Aquí la
        <strong> tasa de descenso del frente</strong> <InlineMath tex={String.raw`R`} /> es ajustable: una sola red
        aprende toda la familia <InlineMath tex={String.raw`\phi(z,t;R)`} />, y en el tab <strong>Live</strong> mover el
        deslizador de <InlineMath tex={String.raw`R`} /> hace que la interfase caiga más rápido o más lento.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> altura de columna <InlineMath tex={String.raw`z\in[0,1]`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} /> (escalados), grilla del campo <InlineMath tex={String.raw`101\times51`} />.</li>
        <li><strong>Incógnita:</strong> la fracción de sólidos <InlineMath tex={String.raw`\phi(z,t)`} />: qué tan concentrada está la suspensión en cada altura e instante.</li>
        <li><strong>Parámetro de control:</strong> la <em>tasa de descenso</em> <InlineMath tex={String.raw`R\in[0.3,0.9]`} />: un input de la red; es la velocidad a la que cae la mud-line (un proxy de la sedimentabilidad).</li>
        <li><strong>Flujo de Kynch (batch):</strong> <InlineMath tex={String.raw`f_{bk}=v_0\,\phi\,(1-\phi/\phi_{max})^C`} />: sedimentación <em>obstaculizada</em> (Richardson-Zaki), con <InlineMath tex={String.raw`v_0=-1`} />, <InlineMath tex={String.raw`\phi_{max}=0.66`} />, <InlineMath tex={String.raw`C=5`} />.</li>
        <li><strong>Difusión degenerada:</strong> <InlineMath tex={String.raw`D(\phi)`} /> se <em>enciende</em> solo por encima de la concentración de gel <InlineMath tex={String.raw`\phi_c=0.23`} /> (consolidación del sedimento), regularizada con un interruptor <InlineMath tex={String.raw`\tanh`} /> para que el residual sea <InlineMath tex={String.raw`C^1`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Es una ley de conservación <strong>fuertemente degenerada</strong>: hiperbólica (puro frente) donde
        <InlineMath tex={String.raw`\phi<\phi_c`} /> y parabólica (difusiva) donde el sedimento consolida. El campo de
        referencia es una solución manufacturada (MMS) con un <strong>frente <InlineMath tex={String.raw`\tanh`} />
        descendente</strong> exacto, que ejercita el operador no lineal genuino:
      </p>
      <Equation tex={String.raw`\phi^*(z,t;R)=\phi_{lo}+(\phi_{hi}-\phi_{lo})\,\tfrac12\Big(1-\tanh\frac{z-s(t)}{W}\Big),\qquad s(t)=z_0-R\,t.`} />
      <p>
        La interfase parte de <InlineMath tex={String.raw`z_0=0.9`} /> y está en <InlineMath tex={String.raw`z=s(t)=z_0-R\,t`} />,
        bajando a velocidad <InlineMath tex={String.raw`R`} />; su <em>grosor</em> lo fija <InlineMath tex={String.raw`W=0.10`} />. La fuente
        manufacturada <InlineMath tex={String.raw`f=\mathcal{L}[\phi^*]`} /> se deriva analíticamente, de modo que
        <InlineMath tex={String.raw`\phi^*`} /> resuelve la EDP modificada de forma <strong>exacta para cualquier</strong>
        <InlineMath tex={String.raw`R`} />: esa es la <strong>ancla de validación</strong>. Sus derivadas son
        cerradas (con <InlineMath tex={String.raw`u=(z-s)/W`} />, <InlineMath tex={String.raw`\operatorname{sech}^2 u=1-\tanh^2 u`} />):
      </p>
      <Equation tex={String.raw`\phi^*_z=-\frac{\Delta\phi}{2W}\operatorname{sech}^2 u,\quad \phi^*_t=-\frac{\Delta\phi\,R}{2W}\operatorname{sech}^2 u,\quad \phi^*_{zz}=\frac{\Delta\phi}{W^2}\operatorname{sech}^2 u\,\tanh u.`} />
      <p>
        Como <InlineMath tex={String.raw`R`} /> entra <em>solo</em> por <InlineMath tex={String.raw`s(t)`} /> (y por
        <InlineMath tex={String.raw`\phi^*_t`} />), y se arrastra analíticamente a la fuente
        <InlineMath tex={String.raw`f`} />, el residual del PINN se anula en <InlineMath tex={String.raw`\phi^*`} /> para
        todo <InlineMath tex={String.raw`R`} />. La PINN <InlineMath tex={String.raw`\phi_\theta(z,t,R)`} /> minimiza ese
        residual; la IC y las BC de Dirichlet se imponen de forma <strong>blanda</strong> e igualadas a
        <InlineMath tex={String.raw`\phi^*`} />, así la red aprende de verdad el campo interior y el L2 reportado es el
        error real del PINN, y se mantiene <InlineMath tex={String.raw`\le 0.41\%`} /> en L2 relativo en las seis variantes
        de <InlineMath tex={String.raw`R`} />, bien dentro de la banda de aceptación <InlineMath tex={String.raw`<2\times10^{-2}`} />;
        el ONNX exportado coincide con la red entrenada hasta un máx-abs de <InlineMath tex={String.raw`5.1\times10^{-7}`} />.
      </p>

      <h3>El método: un frente ensanchado + Adam luego L-BFGS</h3>
      <p>
        El reto es el <em>frente delgado y móvil</em> (<InlineMath tex={String.raw`W=0.10`} />), y añadir
        <InlineMath tex={String.raw`R`} /> como input de la red hace crecer el problema una dimensión. La red es
        <InlineMath tex={String.raw`[3,64,64,64,64,1]`} /> con activaciones <InlineMath tex={String.raw`\tanh`} /> (DeepXDE);
        el entrenamiento es Adam por 24000 pasos a <InlineMath tex={String.raw`\text{lr}=10^{-3}`} />, luego un pulido
        L-BFGS, con pesos de pérdida <InlineMath tex={String.raw`[1,10]`} /> sobre (residual, frontera). El refino
        adaptativo por residual (RAR) se probó y se <strong>descartó aquí</strong>: desestabilizaba el rígido residual de
        difusión degenerada. Lo que sí convergió limpiamente fue ensanchar el frente a
        <InlineMath tex={String.raw`W=0.10`} /> para que un esquema simple de Adam luego L-BFGS resuelva la interfase móvil
        en toda la familia <InlineMath tex={String.raw`R`} />. El flujo obstaculizado <InlineMath tex={String.raw`f_{bk}`} />
        y la difusión degenerada <InlineMath tex={String.raw`D(\phi)`} /> con el interruptor de gel son los
        <em> genuinos</em> de Bürger-Concha; la MMS solo nos da una verdad exacta contra la cual medir.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> sedimentación batch 1D (columna), flujo de Kynch obstaculizado, difusión degenerada
        de consolidación por encima del gel, un único frente descendente paramétrico en <InlineMath tex={String.raw`R`} />.
        Es <em>ilustrativo-sintético</em>: NO existe un campo <InlineMath tex={String.raw`(z,t,\phi)`} /> medido público
        de un espesador, así que el anclaje es una MMS físicamente fiel (rangos tomados de la literatura de
        Bürger-Concha), <strong>no</strong> un ajuste a datos de planta. <strong>Fuera de alcance:</strong> el espesador
        continuo (con alimentación y descarga, un término fuente real), 2-D/3-D y efectos de pared, floculación
        dependiente del tiempo, y el límite verdaderamente hiperbólico <InlineMath tex={String.raw`W\to0`} /> (un choque
        sin grosor). La velocidad de descenso es un proxy: a mayor <InlineMath tex={String.raw`R`} />, mayor
        sedimentabilidad efectiva, no un parámetro de planta calibrado.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> El barrido de <InlineMath tex={String.raw`R`} /> recorre la velocidad
        de clarificación: <em>R=0.30</em> (lento): la mud-line baja apenas, el lecho casi no se forma;
        <em> R=0.42/0.54/0.66</em> la aceleran progresivamente; <em>R=0.78</em> (rápido, cerca del caso original) y
        <em> R=0.90</em>: la interfase llega al fondo de la columna en <InlineMath tex={String.raw`t=1`} />. En todas, el
        <em> grosor</em> del frente es el mismo (<InlineMath tex={String.raw`W`} /> fijo); lo que cambia es cuán rápido
        desciende y, por tanto, cuán pronto se consolida el sedimento.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`\phi(z,t)`} />
        (z vertical = altura de la columna, t horizontal) muestra una <em>banda diagonal</em>: el sobrenadante claro
        (<InlineMath tex={String.raw`\phi`} /> bajo) arriba, el lecho concentrado (<InlineMath tex={String.raw`\phi`} />
        alto) abajo, y la interfase descendiendo entre ambos: su <em>pendiente</em> es la tasa
        <InlineMath tex={String.raw`R`} /> y su <em>nitidez</em> es el grosor <InlineMath tex={String.raw`W`} />. Al pasar el
        cursor se lee la fracción de sólidos exacta y se muestra el <strong>perfil de corte</strong> en
        <InlineMath tex={String.raw`z`} /> (la forma <InlineMath tex={String.raw`\tanh`} /> de la interfase, de clarificado a lecho)
        y en <InlineMath tex={String.raw`t`} /> (cuándo pasa el frente por una altura fija). Los <strong>chips</strong>
        cargan cada tasa de descenso; en <strong>Live</strong>, al deslizar <InlineMath tex={String.raw`R`} /> la
        mud-line cae más rápido o más lento en vivo en el navegador (onnxruntime-web).
      </p>
    </>
  ): (
    <>
      <h2>The problem: a descending mud-line: thickener &amp; tailings sedimentation</h2>
      <p>
        <strong>The problem.</strong> In a thickener (or a tailings column) a flocculated suspension <em>settles</em>:
        particles fall, the supernatant clears at the top, and a <em>consolidated bed</em> grows at the bottom. Between
        them a sharp <strong>mud-line</strong> descends. The solid volume fraction
        <InlineMath tex={String.raw`\phi(z,t)`} /> obeys a <em>strongly-degenerate convection-diffusion</em> conservation
        law of Bürger-Concha type:
        <InlineMath tex={String.raw`\phi_t + \partial_z f_{bk}(\phi) = \partial_z(D(\phi)\,\phi_z)`} />. Here the
        <strong> front descent rate</strong> <InlineMath tex={String.raw`R`} /> is tunable: a single network learns the
        whole family <InlineMath tex={String.raw`\phi(z,t;R)`} />, and in the <strong>Live</strong> tab moving the
        <InlineMath tex={String.raw`R`} /> slider makes the interface fall faster or slower.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> column height <InlineMath tex={String.raw`z\in[0,1]`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} /> (scaled), a <InlineMath tex={String.raw`101\times51`} /> field grid.</li>
        <li><strong>Unknown:</strong> the solid fraction <InlineMath tex={String.raw`\phi(z,t)`} />: how concentrated the suspension is at each height and instant.</li>
        <li><strong>Control parameter:</strong> the <em>descent rate</em> <InlineMath tex={String.raw`R\in[0.3,0.9]`} />: a network input; the speed at which the mud-line falls (a proxy for settleability).</li>
        <li><strong>Kynch batch flux:</strong> <InlineMath tex={String.raw`f_{bk}=v_0\,\phi\,(1-\phi/\phi_{max})^C`} />: <em>hindered</em> settling (Richardson-Zaki), with <InlineMath tex={String.raw`v_0=-1`} />, <InlineMath tex={String.raw`\phi_{max}=0.66`} />, <InlineMath tex={String.raw`C=5`} />.</li>
        <li><strong>Degenerate diffusion:</strong> <InlineMath tex={String.raw`D(\phi)`} /> <em>switches on</em> only above the gel concentration <InlineMath tex={String.raw`\phi_c=0.23`} /> (bed consolidation), regularized by a <InlineMath tex={String.raw`\tanh`} /> switch so the residual is <InlineMath tex={String.raw`C^1`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        It is a <strong>strongly-degenerate</strong> conservation law: hyperbolic (pure front) where
        <InlineMath tex={String.raw`\phi<\phi_c`} /> and parabolic (diffusive) where the sediment consolidates. The
        reference field is a manufactured solution (MMS) with an exact <strong>descending <InlineMath tex={String.raw`\tanh`} />
        front</strong> that exercises the genuine nonlinear operator:
      </p>
      <Equation tex={String.raw`\phi^*(z,t;R)=\phi_{lo}+(\phi_{hi}-\phi_{lo})\,\tfrac12\Big(1-\tanh\frac{z-s(t)}{W}\Big),\qquad s(t)=z_0-R\,t.`} />
      <p>
        The interface starts at <InlineMath tex={String.raw`z_0=0.9`} /> and sits at <InlineMath tex={String.raw`z=s(t)=z_0-R\,t`} />,
        descending at speed <InlineMath tex={String.raw`R`} />; its <em>thickness</em> is set by <InlineMath tex={String.raw`W=0.10`} />. The
        manufactured source <InlineMath tex={String.raw`f=\mathcal{L}[\phi^*]`} /> is derived analytically, so
        <InlineMath tex={String.raw`\phi^*`} /> solves the modified PDE <strong>exactly for any</strong>
        <InlineMath tex={String.raw`R`} />: that is the <strong>validation anchor</strong>. Its derivatives are
        closed-form (with <InlineMath tex={String.raw`u=(z-s)/W`} />, <InlineMath tex={String.raw`\operatorname{sech}^2 u=1-\tanh^2 u`} />):
      </p>
      <Equation tex={String.raw`\phi^*_z=-\frac{\Delta\phi}{2W}\operatorname{sech}^2 u,\quad \phi^*_t=-\frac{\Delta\phi\,R}{2W}\operatorname{sech}^2 u,\quad \phi^*_{zz}=\frac{\Delta\phi}{W^2}\operatorname{sech}^2 u\,\tanh u.`} />
      <p>
        Because <InlineMath tex={String.raw`R`} /> enters <em>only</em> through <InlineMath tex={String.raw`s(t)`} /> (and
        through <InlineMath tex={String.raw`\phi^*_t`} />), and is carried analytically into the source
        <InlineMath tex={String.raw`f`} />, the PINN residual vanishes at <InlineMath tex={String.raw`\phi^*`} /> for all
        <InlineMath tex={String.raw`R`} />. The PINN <InlineMath tex={String.raw`\phi_\theta(z,t,R)`} /> minimises that
        residual; the IC and Dirichlet BCs are imposed <strong>softly</strong> and equal to
        <InlineMath tex={String.raw`\phi^*`} />, so the network genuinely learns the interior field and the reported L2 is
        the true PINN error, and it stays <InlineMath tex={String.raw`\le 0.41\%`} /> relative-L2 across all six
        <InlineMath tex={String.raw`R`} /> variants, well inside the <InlineMath tex={String.raw`<2\times10^{-2}`} />
        acceptance band; the exported ONNX matches the trained net to max-abs <InlineMath tex={String.raw`5.1\times10^{-7}`} />.
      </p>

      <h3>The method: a widened front + Adam then L-BFGS</h3>
      <p>
        The challenge is the <em>thin, moving front</em> (<InlineMath tex={String.raw`W=0.10`} />), and adding
        <InlineMath tex={String.raw`R`} /> as a network input grows the problem by a dimension. The net is
        <InlineMath tex={String.raw`[3,64,64,64,64,1]`} /> with <InlineMath tex={String.raw`\tanh`} /> activations
        (DeepXDE); training is Adam for 24000 steps at <InlineMath tex={String.raw`\text{lr}=10^{-3}`} />, then an L-BFGS
        polish, with loss weights <InlineMath tex={String.raw`[1,10]`} /> on (residual, boundary). Residual adaptive
        refinement (RAR) was tried and <strong>dropped here</strong>: it de-stabilised the stiff degenerate-diffusion
        residual. What converged cleanly instead was widening the front to <InlineMath tex={String.raw`W=0.10`} /> so a
        plain Adam then L-BFGS schedule resolves the moving interface across the whole
        <InlineMath tex={String.raw`R`} /> family. The hindered flux <InlineMath tex={String.raw`f_{bk}`} /> and the
        degenerate diffusion <InlineMath tex={String.raw`D(\phi)`} /> with the gel switch are the <em>genuine</em>
        Bürger-Concha terms; the MMS only supplies an exact truth to measure against.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> 1D batch settling (a column), hindered Kynch flux, degenerate consolidation diffusion
        above the gel, a single descending front parametric in <InlineMath tex={String.raw`R`} />. It is
        <em> illustrative-synthetic</em>: there is NO public measured <InlineMath tex={String.raw`(z,t,\phi)`} /> thickener
        field, so the anchor is a physically-faithful MMS (ranges from the Bürger-Concha literature), <strong>not</strong>
        a fit to plant data. <strong>Out of scope:</strong> the continuous thickener (with feed and underflow, a real
        source term), 2-D/3-D and wall effects, time-dependent flocculation, and the truly hyperbolic limit
        <InlineMath tex={String.raw`W\to0`} /> (a thickness-less shock). The descent speed is a proxy: larger
        <InlineMath tex={String.raw`R`} /> means greater effective settleability, not a calibrated plant parameter.
      </p>

      <p>
        <strong>What each variant shows.</strong> The <InlineMath tex={String.raw`R`} /> sweep walks the clarification
        speed: <em>R=0.30</em> (slow): the mud-line barely descends, the bed hardly forms;
        <em> R=0.42/0.54/0.66</em> accelerate it progressively; <em>R=0.78</em> (fast, near the legacy case) and
        <em> R=0.90</em>: the interface reaches the column bottom by <InlineMath tex={String.raw`t=1`} />. In all of them
        the front <em>thickness</em> is the same (fixed <InlineMath tex={String.raw`W`} />); what changes is how fast it
        descends and hence how soon the sediment consolidates.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`\phi(z,t)`} /> (z vertical = column height, t horizontal) shows a <em>diagonal band</em>:
        the clear supernatant (low <InlineMath tex={String.raw`\phi`} />) on top, the concentrated bed (high
        <InlineMath tex={String.raw`\phi`} />) at the bottom, and the interface descending between them: its
        <em> slope</em> is the rate <InlineMath tex={String.raw`R`} /> and its <em>sharpness</em> is the thickness
        <InlineMath tex={String.raw`W`} />. Hover to read the exact solid fraction and watch the
        <strong> line-cut profile</strong> in <InlineMath tex={String.raw`z`} /> (the <InlineMath tex={String.raw`\tanh`} />
        interface shape, clarified to bed) and in <InlineMath tex={String.raw`t`} /> (when the front passes a fixed height).
        The <strong>chips</strong> load each descent rate; in <strong>Live</strong>, slide
        <InlineMath tex={String.raw`R`} /> and watch the mud-line fall faster or slower live in the browser
        (onnxruntime-web).
      </p>
    </>
  );
}
