import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-darcy-operator (Darcy-flow operator learning with an FNO; discrete held-out
 *  test samples as the workbench variants). */
export function DarcyOperatorContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: aprender el OPERADOR de Darcy: un FNO que resuelve toda una familia</h2>
      <p>
        <strong>El problema.</strong> El flujo de Darcy en un medio poroso (un acuífero, un lecho de roca) cumple
        <InlineMath tex={String.raw`-\nabla\!\cdot\big(a(\mathbf{x})\,\nabla u\big)=1`} />, con presión
        <InlineMath tex={String.raw`u=0`} /> en el borde. El <strong>campo de permeabilidad</strong>
        <InlineMath tex={String.raw`a(\mathbf{x})`} /> es heterogéneo y distinto en cada sitio. Resolver una sola
        instancia con un PINN o un método numérico es rutinario; lo difícil: y lo valioso: es resolver la
        <em> familia entera</em>. Aquí <strong>no</strong> entrenamos una red por instancia: aprendemos el
        <strong> operador solución</strong> <InlineMath tex={String.raw`\mathcal{G}:a\mapsto u`} /> con un
        <strong> Operador Neuronal de Fourier (FNO)</strong>. Un único FNO entrenado mapea <em>cualquier</em> campo de
        permeabilidad nuevo a su campo de presión en <strong>una sola pasada</strong>, sin reentrenar.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> el cuadrado unitario <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, grilla del campo <InlineMath tex={String.raw`32\times32`} />.</li>
        <li><strong>Entrada (la función de coeficiente):</strong> el campo de permeabilidad <InlineMath tex={String.raw`a(\mathbf{x})`} />, un campo aleatorio gaussiano umbralizado a dos valores <InlineMath tex={String.raw`\{3,12\}`} />: interfaces de material afiladas (canales).</li>
        <li><strong>Salida (la solución):</strong> la presión <InlineMath tex={String.raw`u(\mathbf{x})`} /> de la ecuación de Darcy, con <InlineMath tex={String.raw`u=0`} /> en <InlineMath tex={String.raw`\partial\Omega`} />.</li>
        <li><strong>El operador:</strong> <InlineMath tex={String.raw`\mathcal{G}_\theta:a\mapsto u`} />: NO una función de coordenadas, sino un mapa de <em>campo a campo</em> aprendido sobre muchos pares <InlineMath tex={String.raw`(a,u)`} />.</li>
        <li><strong>Variantes (chips):</strong> seis campos <InlineMath tex={String.raw`a`} /> <em>fuera de muestra</em> (held-out) que el FNO nunca vio al entrenar: cada chip es una instancia nueva.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        El problema directo es una EDP elíptica en forma de divergencia. En forma débil, <InlineMath tex={String.raw`u`} />
        es el único minimizador de la energía de Dirichlet ponderada por la permeabilidad:
      </p>
      <Equation tex={String.raw`-\nabla\!\cdot\big(a(\mathbf{x})\nabla u\big)=1\ \text{en}\ \Omega,\quad u|_{\partial\Omega}=0\ \Longleftrightarrow\ u=\arg\min_{v\in H_0^1}\ \int_\Omega\Big(\tfrac12\,a\,|\nabla v|^2 - v\Big)\,d\mathbf{x}.`} />
      <p>
        Esto define un <strong>operador solución</strong> <InlineMath tex={String.raw`\mathcal{G}:a\mapsto u`} /> entre
        espacios de funciones. El FNO aprende <InlineMath tex={String.raw`\mathcal{G}`} /> directamente: lo levanta a un
        canal ancho, aplica varias <strong>capas de Fourier</strong> (convolución espectral sobre los modos de baja
        frecuencia más una conexión <InlineMath tex={String.raw`1\times1`} />) y lo proyecta de vuelta. Por trabajar en
        el espacio de Fourier, una sola capa acopla todo el dominio de golpe: eso es lo que le da el carácter
        <em> global, no local</em> de un operador de Green discreto.
      </p>
      <p>
        Como <InlineMath tex={String.raw`a`} /> es un campo aleatorio a dos valores, <strong>no hay solución en forma
        cerrada</strong>. La <strong>ancla de validación</strong> es una referencia numérica de alta fidelidad
        <InlineMath tex={String.raw`u_{\mathrm{ref}}`} />: un esquema de diferencias finitas de 5 puntos con
        <em> conductividades de cara por media armónica</em> y resolución directa dispersa. La media armónica es la
        conductancia exacta en serie de dos celdas, así que el flujo <InlineMath tex={String.raw`q=-a\,u_x`} /> queda
        continuo a través de las interfaces de material aunque <InlineMath tex={String.raw`a`} /> salte: el balance de
        volumen finito es el teorema de la divergencia discreto, y la matriz es definida positiva, así que
        <InlineMath tex={String.raw`u_{\mathrm{ref}}`} /> es una verdad numérica fiel para cada <InlineMath tex={String.raw`a`} />.
      </p>

      <h3>El método: aprendizaje de OPERADOR, no un PINN por instancia</h3>
      <p>
        Esta es la <em>única</em> ficha del catálogo que <strong>no</strong> entrena un PINN para un problema de
        contorno único. Un PINN aprende <em>una</em> función <InlineMath tex={String.raw`u(\mathbf{x})`} /> para
        <em> un</em> <InlineMath tex={String.raw`a`} /> fijo; cambiar <InlineMath tex={String.raw`a`} /> obliga a
        reentrenar. El FNO aprende el <strong>mapa completo</strong> <InlineMath tex={String.raw`a\mapsto u`} /> sobre
        una distribución de campos de permeabilidad, así que generaliza a <InlineMath tex={String.raw`a`} /> nuevos
        <strong> sin reentrenar</strong> y en una sola evaluación hacia adelante.
      </p>
      <Equation tex={String.raw`\mathcal{G}_\theta = Q\circ \big(\mathcal{L}_L\circ\cdots\circ\mathcal{L}_1\big)\circ P,\qquad \mathcal{L}_\ell(v)=\sigma\Big(W_\ell\,v + \mathcal{F}^{-1}\big(R_\ell\cdot \mathcal{F}v\big)\Big),`} />
      <p>
        donde <InlineMath tex={String.raw`P`} /> levanta el canal, <InlineMath tex={String.raw`Q`} /> proyecta,
        <InlineMath tex={String.raw`\mathcal{F}`} /> es la FFT, <InlineMath tex={String.raw`R_\ell`} /> son pesos
        complejos aprendidos que multiplican solo los <strong>modos de Fourier más bajos</strong> (el resto se trunca) y
        <InlineMath tex={String.raw`W_\ell`} /> es la conexión local <InlineMath tex={String.raw`1\times1`} />. Se entrena
        minimizando la <strong>L2 relativa</strong> en 256 pares <InlineMath tex={String.raw`(a,u_{\mathrm{ref}})`} /> de
        entrenamiento; la métrica titular es la L2 relativa sobre el conjunto de <strong>prueba retenido</strong>: el
        número real de <em>generalización del operador</em>, 5.5% sobre los 64 campos retenidos.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> Darcy estacionario en 2D, fuente unitaria, BC de Dirichlet homogénea, permeabilidad
        a dos valores (canales afilados), un FNO compacto entrenado sobre una familia de campos sintéticos. Es
        <em> sintético</em> en el sentido honesto: el conjunto de coeficientes analíticos de Darcy es el
        <strong> benchmark estándar de FNO</strong> (Li et al., 2021) y la presión de referencia es una resolución
        numérica (diferencias finitas), no datos de campo reales. <strong>Fuera de alcance:</strong> permeabilidad
        continua/log-normal, anisotropía o tensores de permeabilidad, fuentes no unitarias, 3D, transitorios, y
        cuantificación de incertidumbre del operador. Los números son de <strong>generalización de operador</strong>
        (L2 ~5.5% media sobre el conjunto retenido, por muestra mayormente 2-10%), NO los <InlineMath tex={String.raw`10^{-2}`} /> de un PINN ajustado a una instancia: etiquetados
        como tales, sin inflar.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> Cada chip es un campo de permeabilidad <strong>fuera de muestra</strong>
        distinto: el FNO nunca lo vio. Verás tres campos: la <em>entrada</em> <InlineMath tex={String.raw`a`} /> (los
        canales de alta/baja permeabilidad), la <em>predicción</em> del FNO <InlineMath tex={String.raw`u_{\mathrm{pred}}`} />
        y la <em>referencia</em> <InlineMath tex={String.raw`u_{\mathrm{true}}`} /> de diferencias finitas. La presión es
        alta donde la fuente se acumula contra zonas poco permeables y baja a cero en el borde. Lo importante: es
        <strong> el mismo operador congelado</strong> el que resuelve cada instancia nueva: esa es la promesa del
        aprendizaje de operadores. Las seis geometrías (canales tortuosos, bloques gruesos, vetas delgadas) muestran
        que generaliza sobre <em>toda</em> la familia.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> Usa el <strong>selector de salida</strong> para alternar entre
        <InlineMath tex={String.raw`a`} /> (la entrada), <InlineMath tex={String.raw`u_{\mathrm{pred}}`} /> (lo que dice
        el FNO) y <InlineMath tex={String.raw`u_{\mathrm{true}}`} /> (la verdad numérica); compara
        <InlineMath tex={String.raw`u_{\mathrm{pred}}`} /> con <InlineMath tex={String.raw`u_{\mathrm{true}}`} /> para
        ver dónde el operador acierta y dónde se desvía (típicamente en las interfaces más afiladas). Al pasar el cursor para
        leer valores exactos y mira los <strong>perfiles de corte</strong>. Los <strong>chips</strong> saltan entre
        instancias retenidas; cada uno reporta su propia L2 además de la L2 del conjunto de prueba (la métrica titular,
        igual en todos los chips). Como el FNO es un mapa <strong>campo to campo</strong> (no de coordenadas), el tab
        <strong> Live</strong> reproduce los campos precalculados: no hay deslizador de coordenadas, y eso es correcto para
        un operador: el ONNX precalculado es el propio grafo del FNO, verificado por paridad a 1.8e-6 (máx abs).
      </p>
    </>
  ): (
    <>
      <h2>The problem: learning the Darcy OPERATOR: one FNO that solves a whole family</h2>
      <p>
        <strong>The problem.</strong> Darcy flow in a porous medium (an aquifer, a rock bed) satisfies
        <InlineMath tex={String.raw`-\nabla\!\cdot\big(a(\mathbf{x})\,\nabla u\big)=1`} />, with pressure
        <InlineMath tex={String.raw`u=0`} /> on the boundary. The <strong>permeability field</strong>
        <InlineMath tex={String.raw`a(\mathbf{x})`} /> is heterogeneous and different at every site. Solving a single
        instance with a PINN or a numerical method is routine; the hard: and valuable: task is to solve the
        <em> whole family</em>. Here we do <strong>not</strong> train one network per instance: we learn the
        <strong> solution operator</strong> <InlineMath tex={String.raw`\mathcal{G}:a\mapsto u`} /> with a
        <strong> Fourier Neural Operator (FNO)</strong>. A single trained FNO maps <em>any</em> new permeability field to
        its pressure field in <strong>one forward pass</strong>, with no retraining.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> the unit square <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, a <InlineMath tex={String.raw`32\times32`} /> field grid.</li>
        <li><strong>Input (the coefficient function):</strong> the permeability field <InlineMath tex={String.raw`a(\mathbf{x})`} />, a Gaussian random field thresholded to two values <InlineMath tex={String.raw`\{3,12\}`} />: sharp material interfaces (channels).</li>
        <li><strong>Output (the solution):</strong> the Darcy pressure <InlineMath tex={String.raw`u(\mathbf{x})`} />, with <InlineMath tex={String.raw`u=0`} /> on <InlineMath tex={String.raw`\partial\Omega`} />.</li>
        <li><strong>The operator:</strong> <InlineMath tex={String.raw`\mathcal{G}_\theta:a\mapsto u`} />: NOT a function of coordinates, but a learned <em>field-to-field</em> map fit on many <InlineMath tex={String.raw`(a,u)`} /> pairs.</li>
        <li><strong>Variants (chips):</strong> six <em>held-out</em> permeability fields <InlineMath tex={String.raw`a`} /> the FNO never saw at training: each chip is a new instance.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        The forward problem is an elliptic PDE in divergence form. In weak form, <InlineMath tex={String.raw`u`} /> is the
        unique minimiser of the permeability-weighted Dirichlet energy:
      </p>
      <Equation tex={String.raw`-\nabla\!\cdot\big(a(\mathbf{x})\nabla u\big)=1\ \text{in}\ \Omega,\quad u|_{\partial\Omega}=0\ \Longleftrightarrow\ u=\arg\min_{v\in H_0^1}\ \int_\Omega\Big(\tfrac12\,a\,|\nabla v|^2 - v\Big)\,d\mathbf{x}.`} />
      <p>
        This defines a <strong>solution operator</strong> <InlineMath tex={String.raw`\mathcal{G}:a\mapsto u`} /> between
        function spaces. The FNO learns <InlineMath tex={String.raw`\mathcal{G}`} /> directly: it lifts the input to a
        wide channel, applies several <strong>Fourier layers</strong> (a spectral convolution over the low-frequency
        modes plus a <InlineMath tex={String.raw`1\times1`} /> skip), and projects back. Because it works in Fourier
        space, a single layer couples the whole domain at once: that is what gives it the <em>global, non-local</em>
        character of a discrete Green's operator.
      </p>
      <p>
        Because <InlineMath tex={String.raw`a`} /> is a two-value random field, there is <strong>no closed-form
        solution</strong>. The <strong>validation anchor</strong> is a high-fidelity numerical reference
        <InlineMath tex={String.raw`u_{\mathrm{ref}}`} />: a 5-point finite-difference scheme with
        <em> harmonic-mean face conductivities</em> and a sparse direct solve. The harmonic mean is the exact
        series-conductance of two cells, so the flux <InlineMath tex={String.raw`q=-a\,u_x`} /> stays continuous across
        material interfaces even when <InlineMath tex={String.raw`a`} /> jumps: the finite-volume balance is the
        discrete divergence theorem, and the matrix is positive-definite, so <InlineMath tex={String.raw`u_{\mathrm{ref}}`} />
        is a faithful numerical truth for each <InlineMath tex={String.raw`a`} />.
      </p>

      <h3>The method: OPERATOR learning, not a per-instance PINN</h3>
      <p>
        This is the <em>only</em> case in the catalogue that does <strong>not</strong> train a PINN for a single
        boundary-value problem. A PINN learns <em>one</em> function <InlineMath tex={String.raw`u(\mathbf{x})`} /> for
        <em> one</em> fixed <InlineMath tex={String.raw`a`} />; changing <InlineMath tex={String.raw`a`} /> forces a
        retrain. The FNO learns the <strong>whole map</strong> <InlineMath tex={String.raw`a\mapsto u`} /> over a
        distribution of permeability fields, so it generalises to new <InlineMath tex={String.raw`a`} />
        <strong> without retraining</strong> and in a single forward evaluation.
      </p>
      <Equation tex={String.raw`\mathcal{G}_\theta = Q\circ \big(\mathcal{L}_L\circ\cdots\circ\mathcal{L}_1\big)\circ P,\qquad \mathcal{L}_\ell(v)=\sigma\Big(W_\ell\,v + \mathcal{F}^{-1}\big(R_\ell\cdot \mathcal{F}v\big)\Big),`} />
      <p>
        where <InlineMath tex={String.raw`P`} /> lifts the channel, <InlineMath tex={String.raw`Q`} /> projects,
        <InlineMath tex={String.raw`\mathcal{F}`} /> is the FFT, <InlineMath tex={String.raw`R_\ell`} /> are learned
        complex weights that multiply only the <strong>lowest Fourier modes</strong> (the rest are truncated) and
        <InlineMath tex={String.raw`W_\ell`} /> is the local <InlineMath tex={String.raw`1\times1`} /> skip. It is trained
        by minimising the <strong>relative L2</strong> on 256 training <InlineMath tex={String.raw`(a,u_{\mathrm{ref}})`} />
        pairs; the headline metric is the relative L2 on the <strong>held-out test set</strong>: the true
        <em> operator-generalization</em> number, 5.5% over the 64 held-out fields.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> steady 2D Darcy flow, unit source, homogeneous Dirichlet BC, two-value permeability
        (sharp channels), a compact FNO trained over a family of synthetic fields. It is <em>synthetic</em> in the
        honest sense: the analytic-coefficient Darcy set is the <strong>field-standard FNO benchmark</strong> (Li et al.,
        2021) and the reference pressure is a numerical (finite-difference) solve, not real field data.
        <strong> Out of scope:</strong> continuous / log-normal permeability, anisotropy or permeability tensors,
        non-unit sources, 3D, transients, and operator uncertainty quantification. The numbers are
        <strong> operator-generalization</strong> figures (L2 ~5.5% held-out mean, per-sample mostly 2-10%), NOT the <InlineMath tex={String.raw`10^{-2}`} /> of
        a PINN fit to one instance: labeled as such, not inflated.
      </p>

      <p>
        <strong>What each variant shows.</strong> Each chip is a different <strong>held-out</strong> permeability field
      : the FNO never saw it. You get three fields: the <em>input</em> <InlineMath tex={String.raw`a`} /> (the high/low
        permeability channels), the FNO's <em>prediction</em> <InlineMath tex={String.raw`u_{\mathrm{pred}}`} />, and the
        finite-difference <em>reference</em> <InlineMath tex={String.raw`u_{\mathrm{true}}`} />. The pressure is high
        where the source piles up against low-permeability zones and drops to zero on the boundary. The key point: it is
        <strong> the same frozen operator</strong> solving every new instance: that is the promise of operator learning.
        The six geometries (tortuous channels, coarse blocks, thin veins) show it generalises across the <em>whole</em>
        family.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> Use the <strong>output selector</strong> to switch between
        <InlineMath tex={String.raw`a`} /> (the input), <InlineMath tex={String.raw`u_{\mathrm{pred}}`} /> (what the FNO
        says) and <InlineMath tex={String.raw`u_{\mathrm{true}}`} /> (the numerical truth); compare
        <InlineMath tex={String.raw`u_{\mathrm{pred}}`} /> against <InlineMath tex={String.raw`u_{\mathrm{true}}`} /> to
        see where the operator is right and where it drifts (typically at the sharpest interfaces). Hover to read exact
        values and watch the <strong>line-cut profiles</strong>. The <strong>chips</strong> jump between held-out
        instances; each reports its own L2 alongside the test-set L2 (the headline metric, the same on every chip).
        Because the FNO is a <strong>field to field</strong> map (not a coordinate map), the <strong>Live</strong> tab
        replays the baked fields: there is no coordinate slider, and that is correct for an operator: the baked ONNX is
        the FNO's own graph, parity-checked to 1.8e-6 (max abs).
      </p>
    </>
  );
}
