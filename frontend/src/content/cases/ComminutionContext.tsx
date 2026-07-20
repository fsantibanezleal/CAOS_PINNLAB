import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for mine-comminution-pbe (size-transport reduction of the comminution PBE, parametric in
 *  the grind rate: an advected-diffused size distribution shifting toward smaller particles). */
export function ComminutionContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>la molienda corre la distribución de tamaños hacia partículas finas: con una tasa de molienda ajustable</h2>
      <p>
        En la conminución (molienda SAG / de bolas) las partículas se fracturan
        continuamente, de modo que la <strong>distribución de tamaños</strong> <InlineMath tex={String.raw`n(s,t)`} />
        se desplaza hacia tamaños menores con el tiempo de residencia. El modelo completo es la <em>ecuación de balance
        poblacional</em> (PBE), una ecuación íntegro-diferencial con núcleos de <em>selección</em> y <em>ruptura</em>.
        Aquí trabajamos la <strong>reducción de transporte en tamaño</strong>: la fragmentación neta se representa como
        un <em>arrastre</em> hacia tamaños menores (la tasa de molienda <InlineMath tex={String.raw`g`} />) más una
        <em> dispersión</em> en tamaño <InlineMath tex={String.raw`D`} /> que ensancha la distribución: 
        <InlineMath tex={String.raw`n_t + (-g)\,n_s = D\,n_{ss}`} />. La <strong>tasa de molienda</strong>
        <InlineMath tex={String.raw`g`} /> es ajustable: una sola red aprende toda la familia
        <InlineMath tex={String.raw`n(s,t;g)`} />, y en el tab <strong>Live</strong> el deslizador de
        <InlineMath tex={String.raw`g`} /> mueve la distribución hacia los finos más rápido (molienda intensa) o más
        lento (molienda suave).
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> tamaño normalizado <InlineMath tex={String.raw`s\in[0,1]`} /> (1 = grueso, 0 = fino) × tiempo de molienda <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`81\times81`} /> por instante.</li>
        <li><strong>Incógnita:</strong> la densidad de la distribución de tamaños <InlineMath tex={String.raw`n(s,t)`} /> (masa por intervalo de tamaño).</li>
        <li><strong>Parámetro de control:</strong> la <em>tasa de molienda</em> <InlineMath tex={String.raw`g\in[0,0.6]`} />: un input de la red. Fija cuán rápido el centro de la distribución baja en tamaño.</li>
        <li><strong>Dispersión en tamaño:</strong> <InlineMath tex={String.raw`D=0.012`} />: ensancha la distribución (la firma de la fragmentación, que reparte masa en un rango de tamaños).</li>
        <li><strong>Condición inicial:</strong> una alimentación gruesa estrecha centrada en <InlineMath tex={String.raw`s_0=0.8`} /> (varianza inicial <InlineMath tex={String.raw`\sigma_0^2=0.01`} />).</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Para una alimentación gaussiana, la reducción de transporte en tamaño tiene <strong>solución exacta</strong> (la
        función de Green 1D advectada por el arrastre de molienda): es nuestra <strong>ancla de validación</strong>, no
        una fuente manufacturada:
      </p>
      <Equation tex={String.raw`n^*(s,t;g)=\sqrt{\frac{\sigma_0^2}{\sigma_0^2+2Dt}}\;\exp\!\Big(-\frac{\big(s-(s_0-g\,t)\big)^2}{2\,(\sigma_0^2+2Dt)}\Big).`} />
      <p>
        El <em>centro</em> baja en tamaño con la tasa de molienda (<InlineMath tex={String.raw`s_0-g\,t`} />, el
        desplazamiento neto hacia finos), la <em>varianza</em> crece linealmente
        (<InlineMath tex={String.raw`\sigma^2=\sigma_0^2+2Dt`} />, el ensanchamiento por fragmentación) y el
        <em> pico</em> decae como <InlineMath tex={String.raw`\sigma_0/\sigma`} /> porque la masa total se conserva. Al
        sustituir <InlineMath tex={String.raw`n^*`} /> en la EDP el residual es idénticamente cero para
        <em> cualquier</em> <InlineMath tex={String.raw`g`} />: la velocidad de arrastre del centro,
        <InlineMath tex={String.raw`-g`} />, equilibra exactamente al término de transporte. La PINN
        usa una <strong>transformada de salida con IC dura</strong> (el patrón burgers/flotación):
        <InlineMath tex={String.raw`n_\theta(s,t,g)=g_0(s)+t\,\mathcal{N}_\theta(s,t,g)`} /> con
        <InlineMath tex={String.raw`g_0(s)=n^*(s,0)`} /> la gaussiana inicial independiente de la tasa de molienda, de
        modo que la <strong>condición inicial se cumple de forma exacta por construcción</strong> y la red solo debe
        aprender la evolución interior. Minimiza el residual de transporte en puntos de colocación y se ancla al campo
        exacto (la solución analítica), <strong>sin un término de pérdida de frontera aparte</strong>, así el L2 relativo
        reportado es el error real del PINN.
      </p>

      <h3>El método: balance poblacional reducido</h3>
      <p>
        La PBE completa es <InlineMath tex={String.raw`n_t = \int_s^{\infty} b(s,s')\,S(s')\,n(s',t)\,ds' - S(s)\,n(s,t)`} />,
        con un núcleo de <em>selección</em> <InlineMath tex={String.raw`S`} /> (qué tan rápido se rompe cada tamaño) y
        uno de <em>ruptura</em> <InlineMath tex={String.raw`b`} /> (cómo se reparten los fragmentos). Aquí ejercitamos su
        <strong> reducción de transporte</strong>: el primer momento del núcleo de ruptura es un <em>arrastre</em> hacia
        tamaños menores y el segundo momento, una <em>dispersión</em>: la aproximación de Fokker-Planck del operador de
        fragmentación. Esto convierte una ecuación íntegro-diferencial en una EDP de deriva-difusión con
        <strong> ancla cerrada exacta</strong>, ideal para un banco de pruebas de PINN; el modelo de núcleos completo se
        documenta como el modelo verdadero. La red sustituta es un MLP tanh pequeño
        <InlineMath tex={String.raw`[3,48,48,48,48,1]`} /> (DeepXDE) entrenado con Adam y luego L-BFGS sobre un hipercubo
        <InlineMath tex={String.raw`(s,t,g)`} />; el ONNX exportado coincide con el modelo entrenado a 9.5e-7 (máx. abs).
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> el desplazamiento neto de la distribución de tamaños hacia los finos (arrastre +
        dispersión), tasa de molienda constante por corrida (paramétrica en <InlineMath tex={String.raw`g`} />), una
        alimentación gaussiana. Es <em>ilustrativo-sintético</em>: una reducción limpia de la PBE, NO una PSD de molino
        SAG/bolas ajustada (no hay un dataset público de PSD de molienda con un eje de tasa). <strong>Fuera de
        alcance:</strong> los núcleos de selección/ruptura completos (dependientes del tamaño), la aglomeración, una
        tasa de molienda distribuida o dependiente del estado, y la dinámica de carga/llenado del molino. La frontera
        inferior es abierta: la masa puede salir por debajo de <InlineMath tex={String.raw`s=0`} /> (finos por debajo
        del rango resuelto), como cuando se produce material ultrafino.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> El barrido de la tasa de molienda recorre la intensidad de la
        conminución: <em>g=0</em>: sin arrastre, la distribución solo se ensancha en su sitio (molino apenas activo);
        <em> g=0.12/0.24/0.36</em> la corren progresivamente hacia tamaños menores; <em>g=0.48</em> y <em>g=0.6</em>
        (molienda intensa) la llevan muy abajo, con el grueso de la masa ya en los finos al final de la residencia. En
        todas, la distribución <strong>se ensancha al mismo ritmo</strong> <InlineMath tex={String.raw`\sigma\propto\sqrt{\sigma_0^2+2Dt}`} />
        (la tasa cambia el desplazamiento, no el ensanchamiento). La precisión no es uniforme en el barrido: las variantes de
        baja molienda quedan muy por debajo del 1% de L2 relativo, mientras que la esquina de molienda intensa
        <em>g=0.6</em> se vuelve dominada por advección (Péclet ~50) y llega a ~2%, el borde honesto de la banda esperada.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`n(s,t)`} />
        (s horizontal, t vertical) muestra una <em>banda diagonal</em> que baja en tamaño al avanzar la molienda: su
        <em> inclinación</em> es la tasa <InlineMath tex={String.raw`g`} /> y su <em>anchura creciente</em> es la
        dispersión <InlineMath tex={String.raw`D`} />. Al pasar el cursor se lee el valor exacto y se muestra el
        <strong> perfil de corte</strong> en <InlineMath tex={String.raw`s`} /> (la distribución de tamaños en un
        instante: la campana que se corre y se aplana) y en <InlineMath tex={String.raw`t`} /> (cómo crece y luego cae
        la masa en un tamaño fijo conforme la moda pasa por él). Los <strong>chips</strong> cargan cada tasa de molienda;
        en <strong>Live</strong>, al deslizar <InlineMath tex={String.raw`g`} /> la distribución se corre hacia los
        finos en vivo en el navegador (onnxruntime-web).
      </p>
    </>
  ): (
    <>
      <h2>grinding shifts the size distribution toward fines: with a tunable grind rate</h2>
      <p>
        In comminution (SAG / ball milling) particles fracture continuously, so the
        <strong> particle-size distribution</strong> <InlineMath tex={String.raw`n(s,t)`} /> shifts toward smaller sizes
        with residence time. The full model is the <em>population-balance equation</em> (PBE), an integro-differential
        equation with <em>selection</em> and <em>breakage</em> kernels. Here we work its <strong>size-transport
        reduction</strong>: net fragmentation is represented as a <em>drift</em> toward smaller size (the grind rate
        <InlineMath tex={String.raw`g`} />) plus a <em>dispersion</em> in size <InlineMath tex={String.raw`D`} /> that
        broadens the distribution: <InlineMath tex={String.raw`n_t + (-g)\,n_s = D\,n_{ss}`} />. The <strong>grind
        rate</strong> <InlineMath tex={String.raw`g`} /> is tunable: a single network learns the whole family
        <InlineMath tex={String.raw`n(s,t;g)`} />, and in the <strong>Live</strong> tab moving the
        <InlineMath tex={String.raw`g`} /> slider drives the distribution toward fines faster (hard grinding) or slower
        (gentle grinding).
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> normalized size <InlineMath tex={String.raw`s\in[0,1]`} /> (1 = coarse, 0 = fine) × grind time <InlineMath tex={String.raw`t\in[0,1]`} />, an <InlineMath tex={String.raw`81\times81`} /> field grid per instant.</li>
        <li><strong>Unknown:</strong> the size-distribution density <InlineMath tex={String.raw`n(s,t)`} /> (mass per size interval).</li>
        <li><strong>Control parameter:</strong> the <em>grind rate</em> <InlineMath tex={String.raw`g\in[0,0.6]`} />: a network input. It sets how fast the distribution's center moves down in size.</li>
        <li><strong>Size dispersion:</strong> <InlineMath tex={String.raw`D=0.012`} />: broadens the distribution (the fragmentation signature, spreading mass over a range of sizes).</li>
        <li><strong>Initial condition:</strong> a narrow coarse feed centered at <InlineMath tex={String.raw`s_0=0.8`} /> (initial variance <InlineMath tex={String.raw`\sigma_0^2=0.01`} />).</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        For a Gaussian feed, the size-transport reduction has an <strong>exact solution</strong> (the 1D Green's
        function advected by the grind drift): it is our <strong>validation anchor</strong>, not a manufactured source:
      </p>
      <Equation tex={String.raw`n^*(s,t;g)=\sqrt{\frac{\sigma_0^2}{\sigma_0^2+2Dt}}\;\exp\!\Big(-\frac{\big(s-(s_0-g\,t)\big)^2}{2\,(\sigma_0^2+2Dt)}\Big).`} />
      <p>
        The <em>center</em> moves down in size with the grind rate (<InlineMath tex={String.raw`s_0-g\,t`} />, the net
        shift toward fines), the <em>variance</em> grows linearly (<InlineMath tex={String.raw`\sigma^2=\sigma_0^2+2Dt`} />,
        the fragmentation broadening) and the <em>peak</em> decays as <InlineMath tex={String.raw`\sigma_0/\sigma`} />
        because total mass is conserved. Substituting <InlineMath tex={String.raw`n^*`} /> into the PDE makes the
        residual identically zero for <em>any</em> <InlineMath tex={String.raw`g`} />: the center's drift velocity,
        <InlineMath tex={String.raw`-g`} />, exactly balances the transport term. The PINN uses a <strong>hard-IC output
        transform</strong> (the burgers/flotation pattern):
        <InlineMath tex={String.raw`n_\theta(s,t,g)=g_0(s)+t\,\mathcal{N}_\theta(s,t,g)`} /> with
        <InlineMath tex={String.raw`g_0(s)=n^*(s,0)`} /> the grind-rate-independent initial Gaussian, so the
        <strong> initial condition is satisfied exactly by construction</strong> and the network only has to learn the
        interior evolution. It minimises the transport residual at collocation points and is anchored to the exact field
        (the analytic solution), with <strong>no separate boundary-loss term</strong>, so the reported relative-L2 is the
        true PINN error.
      </p>

      <h3>The method: reduced population balance</h3>
      <p>
        The full PBE is <InlineMath tex={String.raw`n_t = \int_s^{\infty} b(s,s')\,S(s')\,n(s',t)\,ds' - S(s)\,n(s,t)`} />,
        with a <em>selection</em> kernel <InlineMath tex={String.raw`S`} /> (how fast each size breaks) and a
        <em> breakage</em> kernel <InlineMath tex={String.raw`b`} /> (how fragments are distributed). Here we exercise
        its <strong>transport reduction</strong>: the first moment of the breakage kernel is a <em>drift</em> toward
        smaller sizes and the second moment a <em>dispersion</em>: the Fokker-Planck approximation of the fragmentation
        operator. This turns an integro-differential equation into a drift-diffusion PDE with an <strong>exact
        closed-form anchor</strong>, ideal for a PINN test bed; the full kernel model is documented as the true model. The
        surrogate net is a small tanh MLP <InlineMath tex={String.raw`[3,48,48,48,48,1]`} /> (DeepXDE) trained with Adam
        then L-BFGS on a <InlineMath tex={String.raw`(s,t,g)`} /> hypercube; the exported ONNX matches the trained model
        to 9.5e-7 (max abs).
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> the net shift of the size distribution toward fines (drift + dispersion), a constant
        grind rate per run (parametric in <InlineMath tex={String.raw`g`} />), a Gaussian feed. It is
        <em> illustrative-synthetic</em>: a clean reduction of the PBE, not a fitted SAG/ball-mill PSD (there is no
        public grinding-PSD dataset with a rate axis). <strong>Out of scope:</strong> the full size-dependent
        selection/breakage kernels, agglomeration, a distributed or state-dependent grind rate, and mill load/fill
        dynamics. The lower boundary is open: mass may leave below <InlineMath tex={String.raw`s=0`} /> (fines below the
        resolved range), as when ultrafine material is produced.
      </p>

      <p>
        <strong>What each variant shows.</strong> The grind-rate sweep walks the intensity of comminution: <em>g=0</em>: 
        no drift, the distribution only broadens in place (a barely-active mill); <em>g=0.12/0.24/0.36</em> shift it
        progressively toward smaller sizes; <em>g=0.48</em> and <em>g=0.6</em> (hard grinding) drive it far down, with
        the bulk of the mass already in the fines by the end of residence. In all of them the distribution
        <strong> broadens at the same rate</strong> <InlineMath tex={String.raw`\sigma\propto\sqrt{\sigma_0^2+2Dt}`} />
        (the rate changes the shift, not the broadening). Accuracy is not uniform across the sweep: the low-grind
        variants sit well under 1% relative-L2, while the high-grind corner <em>g=0.6</em> becomes advection-leaning
        (Péclet ~50) and reaches ~2%, the honest edge of the expected band.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`n(s,t)`} /> (s horizontal, t vertical) shows a <em>diagonal band</em> moving down in
        size as grinding proceeds: its <em>slope</em> is the rate <InlineMath tex={String.raw`g`} /> and its
        <em> growing width</em> is the dispersion <InlineMath tex={String.raw`D`} />. Hover to read the exact value and
        watch the <strong>line-cut profile</strong> in <InlineMath tex={String.raw`s`} /> (the size distribution at an
        instant: the bell shifting and flattening) and in <InlineMath tex={String.raw`t`} /> (how mass at a fixed size
        rises then falls as the mode passes through it). The <strong>chips</strong> load each grind rate; in
        <strong> Live</strong>, slide <InlineMath tex={String.raw`g`} /> and watch the distribution shift toward fines
        live in the browser (onnxruntime-web).
      </p>
    </>
  );
}
