import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for poll-ocean-transport (2D advection-diffusion, time-scrubber over an advected Gaussian). */
export function OceanTransportContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: una mancha de contaminante que deriva y se dispersa: advección-difusión 2D</h2>
      <p>
        <strong>El problema.</strong> Se vierte un contaminante pasivo (plástico, hidrocarburo) en una zona costera. La
        corriente lo <em>arrastra</em> (advección) mientras la turbulencia de remolinos lo <em>dispersa</em> (difusión
        de Fick). La ecuación de transporte <InlineMath tex={String.raw`c_t + \mathbf{v}\cdot\nabla c = D\nabla^2 c`} />
        gobierna la concentración <InlineMath tex={String.raw`c(x,y,t)`} />. Aquí el <strong>tiempo</strong> es el
        parámetro que se barre: una sola red aprende todo el historial <InlineMath tex={String.raw`c(x,y;t)`} />, y en
        el tab <strong>Live</strong> el deslizador de <InlineMath tex={String.raw`t`} /> actúa como un
        <strong> scrubber temporal</strong>: ves la mancha derivar y diluirse cuadro a cuadro.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> mar <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`81\times81`} /> por instante.</li>
        <li><strong>Incógnita:</strong> la concentración <InlineMath tex={String.raw`c(x,y,t)`} /> del contaminante.</li>
        <li><strong>Parámetro barrido:</strong> el <em>tiempo</em> <InlineMath tex={String.raw`t\in[0,1]`} />: un input de la red; cada instante es un campo 2D distinto.</li>
        <li><strong>Corriente:</strong> uniforme <InlineMath tex={String.raw`\mathbf{v}=(0.45,0.35)`} />: arrastra el centro de la mancha.</li>
        <li><strong>Difusividad de remolinos:</strong> <InlineMath tex={String.raw`D=0.01`} />: fija cuán rápido se ensancha (Péclet <InlineMath tex={String.raw`\mathrm{Pe}=|\mathbf{v}|L/D\approx45`} />, advección-dominado).</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Para un vertido puntual gaussiano, la ecuación de advección-difusión tiene <strong>solución exacta</strong> (la
        función de Green 2D trasladada por la corriente): es nuestra <strong>ancla de validación</strong>, no una
        fuente manufacturada:
      </p>
      <Equation tex={String.raw`c^*(x,y,t)=\frac{s_0^2}{s_0^2+2Dt}\,\exp\!\Big(-\frac{(x-x_0-v_x t)^2+(y-y_0-v_y t)^2}{2\,(s_0^2+2Dt)}\Big).`} />
      <p>
        El <em>centro</em> se mueve con la corriente (<InlineMath tex={String.raw`\mathbf{x}_0+\mathbf{v}t`} />), la
        <em> varianza</em> crece linealmente (<InlineMath tex={String.raw`s^2=s_0^2+2Dt`} />, la firma de la difusión) y
        el <em>pico</em> decae como <InlineMath tex={String.raw`s_0^2/s^2`} /> porque la masa total se conserva. La PINN
        <InlineMath tex={String.raw`c_\theta(x,y,t)`} /> minimiza el residual de transporte en puntos de colocación, con
        la IC (el vertido) y las BC (<InlineMath tex={String.raw`c^*`} /> en el borde) impuestas de forma blanda y
        ponderada: así la red aprende de verdad el campo interior y el L2 reportado es el error real del PINN. En
        concreto: una red tanh de 4x64 con entradas <InlineMath tex={String.raw`(x,y,t)`} /> (DeepXDE), entrenada con
        Adam (18000 pasos, lr 1e-3) y pulida con L-BFGS, con pesos de pérdida <InlineMath tex={String.raw`[1,10,50]`} />
        sobre [EDP, BC, IC]: el peso mayor en la IC ancla el vertido sin fijar duramente el interior.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> escalar pasivo, corriente uniforme constante, difusión isótropa, un único vertido
        gaussiano. Es <em>ilustrativo-sintético</em>: físicamente fiel pero NO ajustado a un derrame real ni a un
        producto de corrientes oceánicas reales. <strong>Fuera de alcance:</strong> corrientes giratorias/variables en
        el tiempo, difusión anisótropa o dependiente del estado, reacción/decaimiento del contaminante, y batimetría o
        línea de costa. La frontera abierta deja salir masa (la mancha puede tocar el borde lejano), como en una costa.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> Los chips son <em>instantes</em>: <em>t=0</em> es el vertido
        (compacto e intenso); <em>t=0.2/0.4</em> la mancha ya derivó con la corriente y se ensanchó; <em>t=0.6/0.8</em>
        cruza el centro del dominio diluyéndose; <em>t=1.0</em> es la máxima dispersión. En todos, el centro avanza en
        línea recta a velocidad <InlineMath tex={String.raw`\mathbf{v}`} /> y el ancho crece como
        <InlineMath tex={String.raw`\sqrt{s_0^2+2Dt}`} />.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`c(x,y)`} />
        en un instante muestra la mancha como un disco brillante sobre fondo oscuro; pasa el cursor para leer la
        concentración exacta en cualquier punto y los <strong>perfiles de corte</strong> en <InlineMath tex={String.raw`x`} />
        e <InlineMath tex={String.raw`y`} /> dan las campanas gaussianas (su ancho = la dispersión, su altura = la
        dilución). Los <strong>chips</strong> saltan a cada instante; en <strong>Live</strong>, arrastra
        <InlineMath tex={String.raw`t`} /> como un <em>scrubber</em> y ve el vertido derivar y dispersarse en vivo en tu
        navegador (onnxruntime-web). El ONNX del navegador coincide con la red entrenada a 4.8e-7 (máx abs), así que el
        scrubber en vivo es el mismo solucionador, no una aproximación.
      </p>
    </>
  ): (
    <>
      <h2>The problem: a pollutant patch that drifts and disperses: 2D advection-diffusion</h2>
      <p>
        <strong>The problem.</strong> A passive pollutant (plastic, oil) is released in a coastal zone. The current
        <em> carries</em> it (advection) while eddy turbulence <em>disperses</em> it (Fickian diffusion). The transport
        equation <InlineMath tex={String.raw`c_t + \mathbf{v}\cdot\nabla c = D\nabla^2 c`} /> governs the concentration
        <InlineMath tex={String.raw`c(x,y,t)`} />. Here <strong>time</strong> is the swept parameter: a single network
        learns the whole history <InlineMath tex={String.raw`c(x,y;t)`} />, and in the <strong>Live</strong> tab the
        <InlineMath tex={String.raw`t`} /> slider acts as a <strong>time scrubber</strong>: you watch the patch drift
        and dilute frame by frame.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> sea <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, an <InlineMath tex={String.raw`81\times81`} /> field grid per instant.</li>
        <li><strong>Unknown:</strong> the pollutant concentration <InlineMath tex={String.raw`c(x,y,t)`} />.</li>
        <li><strong>Swept parameter:</strong> <em>time</em> <InlineMath tex={String.raw`t\in[0,1]`} />: a network input; each instant is a different 2D field.</li>
        <li><strong>Current:</strong> uniform <InlineMath tex={String.raw`\mathbf{v}=(0.45,0.35)`} />: carries the patch center.</li>
        <li><strong>Eddy diffusivity:</strong> <InlineMath tex={String.raw`D=0.01`} />: sets how fast it widens (Péclet <InlineMath tex={String.raw`\mathrm{Pe}=|\mathbf{v}|L/D\approx45`} />, advection-dominated).</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        For a Gaussian point release, the advection-diffusion equation has an <strong>exact solution</strong> (the 2D
        Green's function translated by the current): it is our <strong>validation anchor</strong>, not a manufactured
        source:
      </p>
      <Equation tex={String.raw`c^*(x,y,t)=\frac{s_0^2}{s_0^2+2Dt}\,\exp\!\Big(-\frac{(x-x_0-v_x t)^2+(y-y_0-v_y t)^2}{2\,(s_0^2+2Dt)}\Big).`} />
      <p>
        The <em>center</em> moves with the current (<InlineMath tex={String.raw`\mathbf{x}_0+\mathbf{v}t`} />), the
        <em> variance</em> grows linearly (<InlineMath tex={String.raw`s^2=s_0^2+2Dt`} />, the diffusion signature) and
        the <em>peak</em> decays as <InlineMath tex={String.raw`s_0^2/s^2`} /> because total mass is conserved. The PINN
        <InlineMath tex={String.raw`c_\theta(x,y,t)`} /> minimises the transport residual at collocation points, with the
        IC (the release) and BCs (<InlineMath tex={String.raw`c^*`} /> on the boundary) imposed softly and weighted: so
        the network genuinely learns the interior field and the reported L2 is the true PINN error. Concretely: a 4x64
        tanh network with <InlineMath tex={String.raw`(x,y,t)`} /> inputs (DeepXDE), trained by Adam (18000 steps,
        lr 1e-3) then polished with L-BFGS, with loss weights <InlineMath tex={String.raw`[1,10,50]`} /> on [PDE, BC, IC]:
        the heavier IC weight anchors the release without hard-clamping the interior.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> a passive scalar, constant uniform current, isotropic diffusion, a single Gaussian
        release. It is <em>illustrative-synthetic</em>: physically faithful but NOT fit to a real spill or a real
        ocean-current product. <strong>Out of scope:</strong> rotating / time-varying currents, anisotropic or
        state-dependent diffusion, pollutant reaction/decay, and bathymetry or coastline. The open boundary lets mass
        leave (the patch may touch the far edge), as at a coast.
      </p>

      <p>
        <strong>What each variant shows.</strong> The chips are <em>instants</em>: <em>t=0</em> is the release (compact
        and intense); <em>t=0.2/0.4</em> the patch has drifted with the current and broadened; <em>t=0.6/0.8</em> it
        crosses the domain center, diluting; <em>t=1.0</em> is maximal spread. In all of them the center advances in a
        straight line at speed <InlineMath tex={String.raw`\mathbf{v}`} /> and the width grows as
        <InlineMath tex={String.raw`\sqrt{s_0^2+2Dt}`} />.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`c(x,y)`} /> at an instant shows the patch as a bright disc on a dark background;
        hover to read the exact concentration anywhere, and the <strong>line-cut profiles</strong> in
        <InlineMath tex={String.raw`x`} /> and <InlineMath tex={String.raw`y`} /> give the Gaussian bells (their width =
        the dispersion, their height = the dilution). The <strong>chips</strong> jump to each instant; in
        <strong> Live</strong>, drag <InlineMath tex={String.raw`t`} /> like a <em>scrubber</em> and watch the spill
        drift and disperse live in your browser (onnxruntime-web). The browser ONNX matches the trained network to
        4.8e-7 (max abs), so the live scrubber is the same solver, not an approximation.
      </p>
    </>
  );
}
