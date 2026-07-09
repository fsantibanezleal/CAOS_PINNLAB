import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for poll-soil-barrier (diffusion through a low-permeability barrier: a coefficient
 *  jump that puts a kink in the plume, solved with a domain-decomposition / FBPINN-style PINN; single benchmark). */
export function SoilBarrierContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: un contaminante frente a una barrera de baja permeabilidad: y el quiebre que deja</h2>
      <p>
        <strong>El problema.</strong> En un sitio contaminado se instala una <em>barrera vertical</em> de baja
        permeabilidad (un muro de lodo bentonítico o una pantalla de arcilla) para frenar el avance de un contaminante
        disuelto. El contaminante difunde por la columna de suelo, pero al atravesar la barrera se topa con una
        difusividad mucho menor: la barrera actúa como una <strong>resistencia en serie</strong> que ralentiza el
        plume. La ecuación es difusión pura con un coeficiente <em>discontinuo</em>
        <InlineMath tex={String.raw`D(x)`} />:
      </p>
      <Equation tex={String.raw`c_t = D(x)\,c_{xx} + f,\qquad D(x)=\begin{cases} D_{\text{suelo}} & x\notin[a,b]\\[2pt] D_{\text{barrera}} & x\in[a,b]\end{cases}.`} />
      <p>
        El <em>salto</em> de coeficiente es la dificultad central: la concentración <InlineMath tex={String.raw`c`} />
        sigue siendo continua y el flujo difusivo <InlineMath tex={String.raw`D\,c_x`} /> también, pero la derivada
        <InlineMath tex={String.raw`c_x`} /> se vuelve <strong>discontinua</strong> en cada cara de la barrera: el
        perfil desarrolla un <strong>quiebre</strong> (un &laquo;kink&raquo;) justo ahí. Reproducir ese quiebre con una
        única red suave que pelea contra el salto es el reto que motiva el método de este caso.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> profundidad/transecto <InlineMath tex={String.raw`x\in[0,1]`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`101\times51`} />.</li>
        <li><strong>Incógnita:</strong> la concentración normalizada <InlineMath tex={String.raw`c(x,t)\in[0,1]`} /> del contaminante.</li>
        <li><strong>Difusividad por tramos:</strong> <InlineMath tex={String.raw`D_{\text{suelo}}=1`} /> fuera de <InlineMath tex={String.raw`[a,b]=[0.45,0.55]`} /> y <InlineMath tex={String.raw`D_{\text{barrera}}=0.1`} /> dentro: un contraste de 10×.</li>
        <li><strong>Barrera:</strong> una franja delgada centrada en <InlineMath tex={String.raw`x_c=0.5`} />; su baja difusividad es la resistencia dominante.</li>
        <li><strong>Condiciones:</strong> entrada <InlineMath tex={String.raw`c(0,t)=1-e^{-t}`} /> (la fuente que sube), salida <InlineMath tex={String.raw`c(1,t)=0`} /> (extracción) e inicio limpio <InlineMath tex={String.raw`c(x,0)=0`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Construimos una <strong>solución manufacturada</strong> (MMS) que <em>exhibe</em> el quiebre y sirve de
        <strong> ancla de validación</strong> en forma cerrada. La idea física: el perfil de
        <em> resistencia en serie</em>. Definimos la resistencia difusiva acumulada y el perfil estacionario asociado,
      </p>
      <Equation tex={String.raw`R(x)=\int_0^x \frac{dx'}{D(x')},\qquad \Psi(x)=1-\frac{R(x)}{R(L)},\qquad c^*(x,t)=\big(1-e^{-t}\big)\,\Psi(x),`} />
      <p>
        con el término fuente <InlineMath tex={String.raw`f(x,t)=e^{-t}\,\Psi(x)`} />. La <strong>prueba</strong> de que
        <InlineMath tex={String.raw`c^*`} /> resuelve la EDP para <em>cualquier</em> contraste es directa. Como
        <InlineMath tex={String.raw`R'(x)=1/D(x)`} />, se tiene <InlineMath tex={String.raw`\Psi'(x)=-1/\big(D(x)\,R(L)\big)`} />,
        de modo que el flujo difusivo es <strong>constante</strong>:
      </p>
      <Equation tex={String.raw`D(x)\,\Psi'(x)=-\frac{1}{R(L)}=\text{const}\ \Rightarrow\ \frac{d}{dx}\big[D(x)\,\Psi'(x)\big]=0\ \text{en el interior de cada tramo.}`} />
      <p>
        Entonces <InlineMath tex={String.raw`c^*_t=e^{-t}\Psi`} /> y <InlineMath tex={String.raw`D(x)\,c^*_{xx}=(1-e^{-t})\,\tfrac{d}{dx}[D\,\Psi']=0`} />,
        así que <InlineMath tex={String.raw`c^*_t-D(x)\,c^*_{xx}-f=e^{-t}\Psi-0-e^{-t}\Psi=0`} />. El flujo constante
        a través de un <InlineMath tex={String.raw`D`} /> que <em>baja</em> obliga a <InlineMath tex={String.raw`c_x`} />
        a <em>subir</em> dentro de la barrera (<InlineMath tex={String.raw`c_x=-1/(D\,R(L))`} />): ese cambio brusco de
        pendiente en cada cara <strong>es</strong> el quiebre.
      </p>

      <h3>El método: descomposición de dominio (estilo FBPINN)</h3>
      <p>
        Una sola red <InlineMath tex={String.raw`\tanh`} /> suave aproxima mal una derivada discontinua: tiende a
        redondear el quiebre. La técnica de este caso es la <strong>descomposición de dominio</strong> con
        <em> partición de la unidad</em> (en la línea de las <strong>FBPINN</strong>, Moseley, Markham &amp; Nissen-Meyer,
        <em> Advances in Computational Mathematics</em> 2023): se usan <strong>dos sub-redes</strong> mezcladas por
        ventanas suaves <InlineMath tex={String.raw`w_{\text{izq}}+w_{\text{der}}=1`} /> que conmutan a través del centro
        de la barrera,
      </p>
      <Equation tex={String.raw`c_{\text{raw}}(x,t)=w_{\text{izq}}(x)\,\mathcal{N}_1+w_{\text{der}}(x)\,\mathcal{N}_2,\qquad w_{\text{izq}}=\sigma\!\big(\beta\,(x_c-x)\big).`} />
      <p>
        Así el quiebre en <InlineMath tex={String.raw`c_x`} /> lo produce el <em>encuentro de dos redes</em> en la
        frontera del subdominio, no una sola red luchando contra el salto. Encima, las condiciones se imponen de forma
        <strong> dura</strong> (hard constraints) mediante una transformación de salida que ya satisface entrada, salida
        e inicial de forma exacta:
      </p>
      <Equation tex={String.raw`c_\theta=\underbrace{(1-e^{-t})\,(1-x)}_{\text{satisface entrada/salida/IC}}+\;t\,x\,(1-x)\;c_{\text{raw}}(x,t).`} />
      <p>
        El segundo término se anula en <InlineMath tex={String.raw`t=0`} /> y en <InlineMath tex={String.raw`x=0,1`} />,
        así que la red solo aprende la <em>corrección interior</em>. El residual de la EDP se evalúa con la
        <InlineMath tex={String.raw`D(x)`} /> por tramos; se siembran <em>puntos de anclaje</em> sobre las dos caras de
        la barrera para resolver bien el quiebre.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> difusión 1D con una barrera de coeficiente por tramos, contraste fijo de 10×,
        geometría de barrera fija, ancla MMS de resistencia en serie. Es <em>ilustrativo-sintético</em>: valores de
        ingeniería razonables, NO un sitio calibrado ni datos reales. <strong>Fuera de alcance:</strong> advección o un
        campo de velocidad (aquí <InlineMath tex={String.raw`V=0`} />), adsorción/retardo y decaimiento del
        contaminante, múltiples barreras o barrera 2-D, y el <em>límite de contraste extremo</em>
        (<InlineMath tex={String.raw`D_{\text{barrera}}/D_{\text{suelo}}\to0`} />, un quiebre casi-discontinuo).
      </p>
      <p>
        <strong>Por qué una sola variante.</strong> El contraste de difusión <em>es</em> un parámetro físico con
        ancla en forma cerrada para todo valor (la prueba de arriba vale para cualquier contraste). Pero el quiebre ya
        cuesta <InlineMath tex={String.raw`\sim2\times10^{-1}`} /> de L2 relativo en este lane de CPU con dos canales: su
        nitidez crece como <InlineMath tex={String.raw`1/(D_{\text{barrera}}/D_{\text{suelo}})`} />, de modo que barrer
        el contraste obligaría a una sola red a cubrir desde quiebres suaves hasta muy severos, y el régimen más severo
        <em> domina</em> el error: la familia paramétrica <em>empeoraría</em> la precisión y difuminaría justo el rasgo
        que el caso busca mostrar. Por honestidad (ADR-0016 §9.A: nunca fabricar regímenes para inflar un contador de
        chips) se publica como un <strong>benchmark de parámetro fijo</strong> (una variante), no como un barrido.
      </p>

      <p>
        <strong>Qué muestra el benchmark.</strong> El plume entra por la izquierda y avanza, pero la barrera lo
        <strong> frena</strong>: aguas abajo de ella la concentración es notablemente menor. En el perfil espacial se ve
        el sello del problema: tres tramos casi-lineales con <strong>dos quiebres</strong> (uno en cada cara de la
        barrera): la pendiente se empina dentro de la barrera (poco <InlineMath tex={String.raw`D`} /> ⇒ mucho
        <InlineMath tex={String.raw`c_x`} /> para el mismo flujo) y se relaja fuera. El heatmap muestra ese contraste
        como una franja de gradiente intenso a la altura de la barrera.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`c(x,t)`} />
        (x horizontal, t vertical) muestra cómo crece la concentración con el tiempo y cómo la barrera deprime la zona a
        su derecha. Pasa el cursor para leer el valor exacto y localizar la caída a través de la barrera. El
        <strong> perfil de corte</strong> en <InlineMath tex={String.raw`x`} /> es lo más informativo: busca los
        <strong> dos quiebres</strong> en <InlineMath tex={String.raw`x=0.45`} /> y <InlineMath tex={String.raw`x=0.55`} />
        (cambio brusco de pendiente): son la firma de la barrera. El corte en <InlineMath tex={String.raw`t`} /> muestra
        la subida <InlineMath tex={String.raw`1-e^{-t}`} /> hacia el estado estacionario. Como es un benchmark de
        parámetro fijo, el tab <strong>Live</strong> re-evalúa la red entrenada (la misma física) en tu navegador
        (onnxruntime-web), sin deslizador de parámetro.
      </p>
    </>
  ): (
    <>
      <h2>The problem: a contaminant facing a low-permeability barrier: and the kink it leaves</h2>
      <p>
        <strong>The problem.</strong> At a contaminated site a <em>vertical barrier</em> of low permeability (a
        bentonite slurry wall or a clay cutoff) is installed to slow the advance of a dissolved contaminant. The
        contaminant diffuses through the soil column, but crossing the barrier it meets a much lower diffusivity: the
        barrier acts as a <strong>series resistance</strong> that holds the plume back. The equation is pure diffusion
        with a <em>discontinuous</em> coefficient <InlineMath tex={String.raw`D(x)`} />:
      </p>
      <Equation tex={String.raw`c_t = D(x)\,c_{xx} + f,\qquad D(x)=\begin{cases} D_{\text{soil}} & x\notin[a,b]\\[2pt] D_{\text{barrier}} & x\in[a,b]\end{cases}.`} />
      <p>
        The coefficient <em>jump</em> is the central difficulty: the concentration <InlineMath tex={String.raw`c`} />
        stays continuous and so does the diffusive flux <InlineMath tex={String.raw`D\,c_x`} />, but the derivative
        <InlineMath tex={String.raw`c_x`} /> becomes <strong>discontinuous</strong> at each barrier face: the profile
        develops a <strong>kink</strong> there. Reproducing that kink with a single smooth network fighting the jump is
        the challenge that motivates this case's method.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> depth/transect <InlineMath tex={String.raw`x\in[0,1]`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, a <InlineMath tex={String.raw`101\times51`} /> field grid.</li>
        <li><strong>Unknown:</strong> the normalised concentration <InlineMath tex={String.raw`c(x,t)\in[0,1]`} /> of the contaminant.</li>
        <li><strong>Piecewise diffusivity:</strong> <InlineMath tex={String.raw`D_{\text{soil}}=1`} /> outside <InlineMath tex={String.raw`[a,b]=[0.45,0.55]`} /> and <InlineMath tex={String.raw`D_{\text{barrier}}=0.1`} /> inside: a 10× contrast.</li>
        <li><strong>Barrier:</strong> a thin slab centred at <InlineMath tex={String.raw`x_c=0.5`} />; its low diffusivity is the dominant resistance.</li>
        <li><strong>Conditions:</strong> inlet <InlineMath tex={String.raw`c(0,t)=1-e^{-t}`} /> (a rising source), outlet <InlineMath tex={String.raw`c(1,t)=0`} /> (extraction) and a clean start <InlineMath tex={String.raw`c(x,0)=0`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        We build a <strong>manufactured solution</strong> (MMS) that <em>exhibits</em> the kink and serves as the
        closed-form <strong>validation anchor</strong>. The physical idea is the <em>series-resistance</em> steady
        profile. Define the accumulated diffusive resistance and the associated stationary profile,
      </p>
      <Equation tex={String.raw`R(x)=\int_0^x \frac{dx'}{D(x')},\qquad \Psi(x)=1-\frac{R(x)}{R(L)},\qquad c^*(x,t)=\big(1-e^{-t}\big)\,\Psi(x),`} />
      <p>
        with source term <InlineMath tex={String.raw`f(x,t)=e^{-t}\,\Psi(x)`} />. The <strong>proof</strong> that
        <InlineMath tex={String.raw`c^*`} /> solves the PDE for <em>any</em> contrast is direct. Since
        <InlineMath tex={String.raw`R'(x)=1/D(x)`} />, we have <InlineMath tex={String.raw`\Psi'(x)=-1/\big(D(x)\,R(L)\big)`} />,
        so the diffusive flux is <strong>constant</strong>:
      </p>
      <Equation tex={String.raw`D(x)\,\Psi'(x)=-\frac{1}{R(L)}=\text{const}\ \Rightarrow\ \frac{d}{dx}\big[D(x)\,\Psi'(x)\big]=0\ \text{in the interior of each layer.}`} />
      <p>
        Then <InlineMath tex={String.raw`c^*_t=e^{-t}\Psi`} /> and <InlineMath tex={String.raw`D(x)\,c^*_{xx}=(1-e^{-t})\,\tfrac{d}{dx}[D\,\Psi']=0`} />,
        so <InlineMath tex={String.raw`c^*_t-D(x)\,c^*_{xx}-f=e^{-t}\Psi-0-e^{-t}\Psi=0`} />. A constant flux through a
        <InlineMath tex={String.raw`D`} /> that <em>drops</em> forces <InlineMath tex={String.raw`c_x`} /> to
        <em> rise</em> inside the barrier (<InlineMath tex={String.raw`c_x=-1/(D\,R(L))`} />): that abrupt slope change
        at each face <strong>is</strong> the kink.
      </p>

      <h3>The method: domain decomposition (FBPINN-style)</h3>
      <p>
        A single smooth <InlineMath tex={String.raw`\tanh`} /> network approximates a discontinuous derivative poorly:
        it tends to round the kink off. This case's technique is <strong>domain decomposition</strong> with a
        <em> partition of unity</em> (in the spirit of <strong>FBPINNs</strong>, Moseley, Markham &amp; Nissen-Meyer,
        <em> Advances in Computational Mathematics</em> 2023): <strong>two sub-networks</strong> are blended by smooth
        windows <InlineMath tex={String.raw`w_{\text{left}}+w_{\text{right}}=1`} /> that switch across the barrier
        centre,
      </p>
      <Equation tex={String.raw`c_{\text{raw}}(x,t)=w_{\text{left}}(x)\,\mathcal{N}_1+w_{\text{right}}(x)\,\mathcal{N}_2,\qquad w_{\text{left}}=\sigma\!\big(\beta\,(x_c-x)\big).`} />
      <p>
        The kink in <InlineMath tex={String.raw`c_x`} /> is then produced by <em>two networks meeting</em> on the
        subdomain boundary, rather than a single network fighting the jump. On top of that, the conditions are imposed
        as <strong>hard constraints</strong> through an output transform that already satisfies inlet, outlet and IC
        exactly:
      </p>
      <Equation tex={String.raw`c_\theta=\underbrace{(1-e^{-t})\,(1-x)}_{\text{satisfies inlet/outlet/IC}}+\;t\,x\,(1-x)\;c_{\text{raw}}(x,t).`} />
      <p>
        The second term vanishes at <InlineMath tex={String.raw`t=0`} /> and at <InlineMath tex={String.raw`x=0,1`} />,
        so the network only learns the <em>interior correction</em>. The PDE residual is evaluated with the piecewise
        <InlineMath tex={String.raw`D(x)`} />; <em>anchor points</em> are seeded on the two barrier faces so the kink is
        resolved well.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> 1D diffusion with a piecewise-coefficient barrier, a fixed 10× contrast, a fixed
        barrier geometry, a series-resistance MMS anchor. It is <em>illustrative-synthetic</em>: reasonable engineering
        values, NOT a calibrated site or real data. <strong>Out of scope:</strong> advection or a velocity field (here
        <InlineMath tex={String.raw`V=0`} />), adsorption/retardation and contaminant decay, multiple barriers or a 2-D
        barrier, and the <em>extreme-contrast limit</em> (<InlineMath tex={String.raw`D_{\text{barrier}}/D_{\text{soil}}\to0`} />,
        an almost-discontinuous kink).
      </p>
      <p>
        <strong>Why a single variant.</strong> The diffusion contrast <em>is</em> a physical parameter with a
        closed-form anchor for every value (the proof above holds for any contrast). But the kink already costs
        <InlineMath tex={String.raw`\sim2\times10^{-1}`} /> relative L2 on this CPU two-channel lane: its sharpness grows
        as <InlineMath tex={String.raw`1/(D_{\text{barrier}}/D_{\text{soil}})`} />, so sweeping the contrast would force a
        single network to span from mild to very severe kinks, and the most severe regime <em>dominates</em> the error: 
        the parametric family would <em>worsen</em> accuracy and blur the very feature the case exists to show. For
        honesty (ADR-0016 §9.A: never fabricate regimes to inflate a chip count) it ships as a <strong>fixed-parameter
        benchmark</strong> (one variant), not a sweep.
      </p>

      <p>
        <strong>What the benchmark shows.</strong> The plume enters from the left and advances, but the barrier
        <strong> slows it</strong>: downstream of the barrier the concentration is markedly lower. The spatial profile
        shows the problem's signature: three nearly-linear segments with <strong>two kinks</strong> (one at each barrier
        face): the slope steepens inside the barrier (low <InlineMath tex={String.raw`D`} /> ⇒ large
        <InlineMath tex={String.raw`c_x`} /> for the same flux) and relaxes outside. The heatmap shows that contrast as a
        band of steep gradient at the barrier.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`c(x,t)`} /> (x horizontal, t vertical) shows how the concentration grows with time
        and how the barrier depresses the region to its right. Hover to read the exact value and locate the drop across
        the barrier. The <strong>line-cut profile</strong> in <InlineMath tex={String.raw`x`} /> is the most
        informative: look for the <strong>two kinks</strong> at <InlineMath tex={String.raw`x=0.45`} /> and
        <InlineMath tex={String.raw`x=0.55`} /> (abrupt slope change): they are the barrier's fingerprint. The cut in
        <InlineMath tex={String.raw`t`} /> shows the <InlineMath tex={String.raw`1-e^{-t}`} /> rise toward steady state.
        Since it is a fixed-parameter benchmark, the <strong>Live</strong> tab re-evaluates the trained network (the
        same physics) in your browser (onnxruntime-web), with no parameter slider.
      </p>
    </>
  );
}
