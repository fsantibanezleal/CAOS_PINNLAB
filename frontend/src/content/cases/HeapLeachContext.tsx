import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for mine-heap-leach-rt (2-species advection-diffusion-reaction, time-scrubber over an MMS solution). */
export function HeapLeachContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>dos reactivos que percolan, se dispersan y reaccionan en una pila de lixiviación</h2>
      <p>
        En la lixiviación en pilas (o in-situ) de cobre / tierras raras, una solución
        lixiviante percola hacia abajo a través del medio poroso saturado. Dos reactivos acuosos
        <InlineMath tex={String.raw`c_A`} /> y <InlineMath tex={String.raw`c_B`} /> son <em>arrastrados</em> por el
        flujo de Darcy (advección), <em>dispersados</em> por la tortuosidad del poro (difusión) y <em>consumidos</em>
        al reaccionar bimolecularmente <InlineMath tex={String.raw`A+B\to C`} /> a tasa
        <InlineMath tex={String.raw`k_f\,c_A c_B`} />. La concentración de cada especie evoluciona como
        <InlineMath tex={String.raw`c_i(x,z,t)`} />. Aquí el <strong>tiempo</strong> es el parámetro que se barre: una
        sola red aprende todo el historial <InlineMath tex={String.raw`c(x,z;t)`} /> y, en el tab
        <strong> Live</strong>, el deslizador de <InlineMath tex={String.raw`t`} /> actúa como un
        <strong> scrubber temporal</strong>: ves los frentes percolar, reaccionar y relajarse cuadro a cuadro.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> una sección vertical de la pila <InlineMath tex={String.raw`(x,z)\in[0,1]^2`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`41\times41`} /> por instante (<InlineMath tex={String.raw`z`} /> apunta hacia abajo, en el sentido de la percolación).</li>
        <li><strong>Incógnitas:</strong> las concentraciones de los dos reactivos <InlineMath tex={String.raw`c_A(x,z,t)`} /> y <InlineMath tex={String.raw`c_B(x,z,t)`} />.</li>
        <li><strong>Parámetro barrido:</strong> el <em>tiempo</em> <InlineMath tex={String.raw`t\in[0,1]`} />: un input de la red; cada instante es un par de campos 2D distinto.</li>
        <li><strong>Percolación (Darcy):</strong> velocidad descendente <InlineMath tex={String.raw`\mathbf{v}=(0,1)`} />: arrastra ambas especies hacia abajo.</li>
        <li><strong>Dispersión:</strong> <InlineMath tex={String.raw`D=0.05`} />: isótropa; ensancha los frentes.</li>
        <li><strong>Reacción:</strong> <InlineMath tex={String.raw`k_f=1`} />: el sumidero bimolecular <InlineMath tex={String.raw`k_f c_A c_B`} /> acopla las dos ecuaciones.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Cada especie obedece una ecuación de <strong>advección-difusión-reacción</strong>, acopladas por el término de
        reacción común:
      </p>
      <Equation tex={String.raw`\partial_t c_i + \mathbf{v}\cdot\nabla c_i = D\,\nabla^2 c_i \;-\; k_f\,c_A c_B \;+\; f_i,\qquad i\in\{A,B\}.`} />
      <p>
        Para validar la PINN con un <strong>error exacto</strong> usamos una <em>solución manufacturada</em> (MMS): se
        elige un campo objetivo <InlineMath tex={String.raw`c^*`} /> y se deriva analíticamente la fuente
        <InlineMath tex={String.raw`f_i = \mathcal{L}[c^*]`} /> que lo hace solución exacta. El campo elegido (el offset
        <InlineMath tex={String.raw`+1.5`} /> mantiene <InlineMath tex={String.raw`c>0`} />, como toda concentración
        física) es:
      </p>
      <Equation tex={String.raw`c_A^*(x,z,t)=e^{-t}\,\sin(\pi x)\cos(\pi z)+1.5,\qquad c_B^*(x,z,t)=e^{-t/2}\,\cos(\pi x)\sin(\pi z)+1.5.`} />
      <p>
        Sustituyendo <InlineMath tex={String.raw`c^*`} /> en el operador, cada término tiene forma cerrada:
        <InlineMath tex={String.raw`\partial_t c_A^*=-e^{-t}\sin\pi x\cos\pi z`} />, la advección
        <InlineMath tex={String.raw`\mathbf{v}\cdot\nabla c_A^*=-\pi e^{-t}\sin\pi x\sin\pi z`} /> y el laplaciano
        <InlineMath tex={String.raw`\nabla^2 c_A^*=-2\pi^2 e^{-t}\sin\pi x\cos\pi z`} /> (el <InlineMath tex={String.raw`+1.5`} />
        desaparece al derivar). Definiendo <InlineMath tex={String.raw`f_i`} /> como <em>exactamente</em> el operador
        aplicado a <InlineMath tex={String.raw`c^*`} />, el residual se anula
        <InlineMath tex={String.raw`\mathcal{L}[c^*]-f_i\equiv0`} /> para <strong>todo</strong>
        <InlineMath tex={String.raw`(x,z,t)`} />. Las dos especies decaen a ritmos <em>distintos</em>
        (<InlineMath tex={String.raw`e^{-t}`} /> frente a <InlineMath tex={String.raw`e^{-t/2}`} />), lo que hace el
        barrido temporal genuinamente interesante. La PINN
        <InlineMath tex={String.raw`c_\theta`} /> minimiza el residual en puntos de colocación con BC de Dirichlet
        (<InlineMath tex={String.raw`=c^*`} /> en el borde) e IC (<InlineMath tex={String.raw`=c^*`} /> en
        <InlineMath tex={String.raw`t=0`} />) blandas y ponderadas; el L2 reportado por especie es el error real del PINN.
      </p>

      <h3>Resultado (medido, semilla 42)</h3>
      <p>
        <strong>Medido (semilla 42):</strong> <InlineMath tex={String.raw`c_A`} /> alcanza L2 relativo
        <InlineMath tex={String.raw`\le 10^{-4}`} /> y <InlineMath tex={String.raw`c_B`} />
        <InlineMath tex={String.raw`\le 2\times 10^{-4}`} /> contra la MMS analítica en los 6 instantes (banda objetivo
        <InlineMath tex={String.raw`< 2\times 10^{-2}`} /> por especie), con paridad ONNX (máx. abs.)
        <InlineMath tex={String.raw`1.19\times 10^{-6}`} />. Es un solve acoplado de dos especies genuinamente bien
        convergido, validado contra una verdad manufacturada exacta, no un ajuste a datos.
        <strong> Arquitectura:</strong> una sola red compartida <InlineMath tex={String.raw`[3]\to[40]\times 4\to[2]`} />
        con tanh que predice ambas <InlineMath tex={String.raw`c_A`} /> y <InlineMath tex={String.raw`c_B`} /> a través de
        un único grafo de autodiferenciación; la pérdida se pondera <InlineMath tex={String.raw`[1,1,10,10,10,10]`} />
        sobre <InlineMath tex={String.raw`[\text{eqA},\text{eqB},\text{bcA},\text{bcB},\text{icA},\text{icB}]`} /> (BC/IC
        reforzadas 10x); optimización Adam (20 000 pasos, lr <InlineMath tex={String.raw`10^{-3}`} />) seguida de L-BFGS.
      </p>

      <h3>El método: por qué una solución manufacturada (y por qué tiempo, no Péclet)</h3>
      <p>
        El modelo real de una pila incluye un sumidero de <em>núcleo decreciente</em> (shrinking-core), doble porosidad
        y velocidad variable; la reacción bimolecular única <InlineMath tex={String.raw`k_f c_A c_B`} /> es una
        simplificación didáctica <em>bien planteada</em>. Como la fuente <InlineMath tex={String.raw`f_i`} /> se deriva
        para hacer exacta a <InlineMath tex={String.raw`c^*`} />, <strong>no existe</strong> una familia en forma cerrada
        sobre un parámetro físico (Péclet <InlineMath tex={String.raw`\mathrm{Pe}=v_z L/D`} /> o Damköhler
        <InlineMath tex={String.raw`\mathrm{Da}=k_f L/v_z`} />): cambiar <InlineMath tex={String.raw`D`} />,
        <InlineMath tex={String.raw`v_z`} /> o <InlineMath tex={String.raw`k_f`} /> exigiría re-derivar <em>otra</em>
        fuente para el <em>mismo</em> campo, y el campo no cambiaría: sería un régimen <em>fabricado</em>. Lo que sí es
        exacto en toda la historia es la <strong>dinámica temporal</strong>; por eso el eje barrido honesto es
        <InlineMath tex={String.raw`t`} />, no un Péclet inventado.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> dos reactivos acuosos, percolación de Darcy uniforme descendente, dispersión
        isótropa, una reacción bimolecular <InlineMath tex={String.raw`A+B\to C`} />, dominio saturado. Es
        <em> ilustrativo-sintético</em>: rangos de parámetros <em>relevantes</em> a la biolixiviación de Cu/REE chilena,
        pero <strong>NO</strong> ajustado a ningún ensayo de columna ni a datos de planta. Precedente metodológico:
        transporte reactivo con PIML para minerales críticos (arXiv:2506.15960). <strong>Fuera de alcance:</strong>
        cinética de núcleo decreciente, doble porosidad, velocidad/saturación variables, química multi-especie y
        precipitación; y un barrido en un parámetro físico (no es honesto bajo MMS, ver arriba).
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> Los chips son <em>instantes</em>: <em>t=0</em> es el estado inicial
        manufacturado (máximo contraste entre especies); <em>t=0.2/0.4</em> los frentes ya advectaron hacia abajo y
        <InlineMath tex={String.raw`c_A`} /> decayó más rápido que <InlineMath tex={String.raw`c_B`} />;
        <em> t=0.6/0.8</em> ambos relajan hacia la línea base <InlineMath tex={String.raw`+1.5`} /> con
        <InlineMath tex={String.raw`c_A`} /> casi plana; <em>t=1.0</em> es la percolación tardía
        (<InlineMath tex={String.raw`c_A`} /> casi uniforme, <InlineMath tex={String.raw`c_B`} /> aún modulada). El
        decaimiento desigual <InlineMath tex={String.raw`e^{-t}`} /> vs <InlineMath tex={String.raw`e^{-t/2}`} /> es el
        sello del barrido.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de
        <InlineMath tex={String.raw`c_A(x,z)`} /> en un instante muestra el patrón de seno-coseno modulado por el
        decaimiento; al pasar el cursor se lee la concentración exacta en cualquier punto y los <strong>perfiles de
        corte</strong> en <InlineMath tex={String.raw`x`} /> y <InlineMath tex={String.raw`z`} /> dan los lóbulos del
        frente (su amplitud = cuánto ha decaído/reaccionado la especie). Los <strong>chips</strong> saltan a cada
        instante; en <strong>Live</strong>, al arrastrar <InlineMath tex={String.raw`t`} /> como un <em>scrubber</em>
        los reactivos percolan, reaccionan y se relajan en vivo en el navegador (onnxruntime-web). La especie primaria
        graficada es <InlineMath tex={String.raw`c_A`} />; el error de <InlineMath tex={String.raw`c_B`} /> contra la
        MMS se reporta junto al de <InlineMath tex={String.raw`c_A`} />.
      </p>
    </>
  ): (
    <>
      <h2>two reactants that percolate, disperse and react in a leach heap</h2>
      <p>
        In heap (or in-situ) leaching of copper / rare earths, a lixiviant solution
        percolates downward through the saturated porous medium. Two aqueous reactants
        <InlineMath tex={String.raw`c_A`} /> and <InlineMath tex={String.raw`c_B`} /> are <em>carried</em> by the Darcy
        flow (advection), <em>dispersed</em> by pore tortuosity (diffusion) and <em>consumed</em> as they react
        bimolecularly <InlineMath tex={String.raw`A+B\to C`} /> at rate <InlineMath tex={String.raw`k_f\,c_A c_B`} />.
        Each species' concentration evolves as <InlineMath tex={String.raw`c_i(x,z,t)`} />. Here <strong>time</strong> is
        the swept parameter: a single network learns the whole history <InlineMath tex={String.raw`c(x,z;t)`} />, and in
        the <strong>Live</strong> tab the <InlineMath tex={String.raw`t`} /> slider acts as a
        <strong> time scrubber</strong>: the fronts percolate, react and relax frame by frame.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> a vertical heap section <InlineMath tex={String.raw`(x,z)\in[0,1]^2`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, a <InlineMath tex={String.raw`41\times41`} /> field grid per instant (<InlineMath tex={String.raw`z`} /> points downward, along the percolation).</li>
        <li><strong>Unknowns:</strong> the two reactant concentrations <InlineMath tex={String.raw`c_A(x,z,t)`} /> and <InlineMath tex={String.raw`c_B(x,z,t)`} />.</li>
        <li><strong>Swept parameter:</strong> <em>time</em> <InlineMath tex={String.raw`t\in[0,1]`} />: a network input; each instant is a different pair of 2D fields.</li>
        <li><strong>Percolation (Darcy):</strong> downward velocity <InlineMath tex={String.raw`\mathbf{v}=(0,1)`} />: carries both species down.</li>
        <li><strong>Dispersion:</strong> <InlineMath tex={String.raw`D=0.05`} />: isotropic; broadens the fronts.</li>
        <li><strong>Reaction:</strong> <InlineMath tex={String.raw`k_f=1`} />: the bimolecular sink <InlineMath tex={String.raw`k_f c_A c_B`} /> couples the two equations.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        Each species obeys an <strong>advection-diffusion-reaction</strong> equation, coupled through the shared
        reaction term:
      </p>
      <Equation tex={String.raw`\partial_t c_i + \mathbf{v}\cdot\nabla c_i = D\,\nabla^2 c_i \;-\; k_f\,c_A c_B \;+\; f_i,\qquad i\in\{A,B\}.`} />
      <p>
        To validate the PINN with an <strong>exact error</strong> we use a <em>manufactured solution</em> (MMS): pick a
        target field <InlineMath tex={String.raw`c^*`} /> and analytically derive the source
        <InlineMath tex={String.raw`f_i = \mathcal{L}[c^*]`} /> that makes it exact. The chosen field (the
        <InlineMath tex={String.raw`+1.5`} /> offset keeps <InlineMath tex={String.raw`c>0`} />, as every physical
        concentration is) is:
      </p>
      <Equation tex={String.raw`c_A^*(x,z,t)=e^{-t}\,\sin(\pi x)\cos(\pi z)+1.5,\qquad c_B^*(x,z,t)=e^{-t/2}\,\cos(\pi x)\sin(\pi z)+1.5.`} />
      <p>
        Substituting <InlineMath tex={String.raw`c^*`} /> into the operator, every term is closed form:
        <InlineMath tex={String.raw`\partial_t c_A^*=-e^{-t}\sin\pi x\cos\pi z`} />, advection
        <InlineMath tex={String.raw`\mathbf{v}\cdot\nabla c_A^*=-\pi e^{-t}\sin\pi x\sin\pi z`} /> and the Laplacian
        <InlineMath tex={String.raw`\nabla^2 c_A^*=-2\pi^2 e^{-t}\sin\pi x\cos\pi z`} /> (the <InlineMath tex={String.raw`+1.5`} />
        drops on differentiation). Defining <InlineMath tex={String.raw`f_i`} /> as <em>exactly</em> the operator applied
        to <InlineMath tex={String.raw`c^*`} />, the residual vanishes
        <InlineMath tex={String.raw`\mathcal{L}[c^*]-f_i\equiv0`} /> for <strong>all</strong>
        <InlineMath tex={String.raw`(x,z,t)`} />. The two species decay at <em>different</em> rates
        (<InlineMath tex={String.raw`e^{-t}`} /> vs <InlineMath tex={String.raw`e^{-t/2}`} />), which makes the time sweep
        genuinely interesting. The PINN <InlineMath tex={String.raw`c_\theta`} /> minimises the residual at collocation
        points with soft, weighted Dirichlet BCs (<InlineMath tex={String.raw`=c^*`} /> on the boundary) and IC
        (<InlineMath tex={String.raw`=c^*`} /> at <InlineMath tex={String.raw`t=0`} />); the per-species reported L2 is
        the true PINN error.
      </p>

      <h3>Result (measured, seed 42)</h3>
      <p>
        <strong>Measured (seed 42):</strong> <InlineMath tex={String.raw`c_A`} /> reaches relative-L2
        <InlineMath tex={String.raw`\le 10^{-4}`} /> and <InlineMath tex={String.raw`c_B`} />
        <InlineMath tex={String.raw`\le 2\times 10^{-4}`} /> against the analytic MMS across all 6 snapshots (target band
        <InlineMath tex={String.raw`< 2\times 10^{-2}`} /> per species), with ONNX parity (max abs)
        <InlineMath tex={String.raw`1.19\times 10^{-6}`} />. A genuinely well-converged coupled two-species solve,
        validated against an exact manufactured truth, not a fit to data.
        <strong> Architecture:</strong> one shared <InlineMath tex={String.raw`[3]\to[40]\times 4\to[2]`} /> tanh network
        predicting both <InlineMath tex={String.raw`c_A`} /> and <InlineMath tex={String.raw`c_B`} /> through a single
        autodiff graph; the loss is weighted <InlineMath tex={String.raw`[1,1,10,10,10,10]`} /> over
        <InlineMath tex={String.raw`[\text{eqA},\text{eqB},\text{bcA},\text{bcB},\text{icA},\text{icB}]`} /> (BC/IC
        up-weighted 10x); optimisation is Adam (20 000 steps, lr <InlineMath tex={String.raw`10^{-3}`} />) then L-BFGS.
      </p>

      <h3>The method: why a manufactured solution (and why time, not Péclet)</h3>
      <p>
        A real heap model carries a <em>shrinking-core</em> sink, dual porosity and a variable velocity; the single
        bimolecular reaction <InlineMath tex={String.raw`k_f c_A c_B`} /> is a deliberate, <em>well-posed</em> teaching
        simplification. Because the source <InlineMath tex={String.raw`f_i`} /> is derived to make
        <InlineMath tex={String.raw`c^*`} /> exact, there is <strong>no</strong> closed-form family in a physical
        parameter (Péclet <InlineMath tex={String.raw`\mathrm{Pe}=v_z L/D`} /> or Damköhler
        <InlineMath tex={String.raw`\mathrm{Da}=k_f L/v_z`} />): changing <InlineMath tex={String.raw`D`} />,
        <InlineMath tex={String.raw`v_z`} /> or <InlineMath tex={String.raw`k_f`} /> would require re-deriving
        <em> another</em> source for the <em>same</em> field, and the field would not change: that would be a
        <em> fabricated</em> regime. What <em>is</em> exact across the whole history is the <strong>time
        dynamics</strong>; that is why the honest swept axis is <InlineMath tex={String.raw`t`} />, not an invented
        Péclet.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> two aqueous reactants, uniform downward Darcy percolation, isotropic dispersion, one
        bimolecular reaction <InlineMath tex={String.raw`A+B\to C`} />, a saturated domain. It is
        <em> illustrative-synthetic</em>: parameter ranges <em>relevant</em> to Chilean Cu/REE bioleaching, but
        <strong> not</strong> fit to any column test or plant dataset. Method precedent: reactive-transport PIML for
        critical minerals (arXiv:2506.15960). <strong>Out of scope:</strong> shrinking-core kinetics, dual porosity,
        variable velocity/saturation, multi-species chemistry and precipitation; and a sweep over a physical parameter
        (not honest under MMS, see above).
      </p>

      <p>
        <strong>What each variant shows.</strong> The chips are <em>instants</em>: <em>t=0</em> is the manufactured
        initial state (maximal contrast between species); <em>t=0.2/0.4</em> the fronts have advected downward and
        <InlineMath tex={String.raw`c_A`} /> has decayed faster than <InlineMath tex={String.raw`c_B`} />;
        <em> t=0.6/0.8</em> both relax toward the <InlineMath tex={String.raw`+1.5`} /> baseline with
        <InlineMath tex={String.raw`c_A`} /> nearly flat; <em>t=1.0</em> is late percolation
        (<InlineMath tex={String.raw`c_A`} /> almost uniform, <InlineMath tex={String.raw`c_B`} /> still modulated). The
        unequal decay <InlineMath tex={String.raw`e^{-t}`} /> vs <InlineMath tex={String.raw`e^{-t/2}`} /> is the
        signature of the sweep.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`c_A(x,z)`} /> at an instant shows the sine-cosine pattern modulated by the decay;
        hover to read the exact concentration anywhere, and the <strong>line-cut profiles</strong> in
        <InlineMath tex={String.raw`x`} /> and <InlineMath tex={String.raw`z`} /> give the front lobes (their amplitude =
        how much the species has decayed/reacted). The <strong>chips</strong> jump to each instant; in
        <strong> Live</strong>, drag <InlineMath tex={String.raw`t`} /> like a <em>scrubber</em> and watch the reactants
        percolate, react and relax live in the browser (onnxruntime-web). The plotted primary species is
        <InlineMath tex={String.raw`c_A`} />; the <InlineMath tex={String.raw`c_B`} /> error against the MMS is reported
        alongside <InlineMath tex={String.raw`c_A`} />'s.
      </p>
    </>
  );
}
