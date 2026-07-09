import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for ind-heat2d-inverse (sparse-data field inverse: recover k(x,y) from noisy T sensors). */
export function Heat2dInverseContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: recuperar un campo de conductividad oculto desde unos pocos sensores: conducción inversa 2D</h2>
      <p>
        <strong>El problema.</strong> Una placa conduce calor con una conductividad térmica <InlineMath tex={String.raw`k(x,y)`} />
        que <em>no conocemos</em> y que varía punto a punto. Conocemos la fuente de calor <InlineMath tex={String.raw`q(x,y)`} />
        y medimos la temperatura <InlineMath tex={String.raw`T`} /> en apenas <strong>un centenar de sensores dispersos</strong>
        (con ruido). La pregunta inversa es: ¿se puede reconstruir el <strong>campo completo</strong>
        <InlineMath tex={String.raw`k(x,y)`} /> en toda la placa: incluso donde no hay ningún sensor? Aquí
        <InlineMath tex={String.raw`k`} /> no es un escalar a ajustar, sino un <em>campo desconocido</em>; ese es el caso
        canónico donde un PINN aventaja a los métodos clásicos FEM/FVM: la <strong>física</strong> (la EDP) rellena los
        huecos entre sensores y regulariza la reconstrucción.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> la placa <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, grilla del campo <InlineMath tex={String.raw`81\times81`} />.</li>
        <li><strong>Incógnita (objetivo inverso):</strong> el campo de conductividad <InlineMath tex={String.raw`k(x,y)>0`} />: lo que se recupera y se puntúa.</li>
        <li><strong>Estado auxiliar:</strong> la temperatura <InlineMath tex={String.raw`T(x,y)`} />, también salida de la red, con <InlineMath tex={String.raw`T=0`} /> en el borde.</li>
        <li><strong>Datos conocidos:</strong> la fuente <InlineMath tex={String.raw`q(x,y)`} /> (entra en la EDP) y <InlineMath tex={String.raw`\sim100`} /> observaciones ruidosas <InlineMath tex={String.raw`T(x_i,y_i)`} />.</li>
        <li><strong>Ruido de sensor:</strong> <InlineMath tex={String.raw`\sigma=0.01`} /> gaussiano sobre las lecturas de <InlineMath tex={String.raw`T`} />: los datos no son perfectos.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        La conducción estacionaria con conductividad variable obedece la EDP en forma de divergencia, con
        <InlineMath tex={String.raw`T`} /> anclada a cero en el contorno:
      </p>
      <Equation tex={String.raw`\nabla\!\cdot\!\big(k(x,y)\,\nabla T\big)=q(x,y)\ \text{en}\ (0,1)^2,\qquad T|_{\partial\Omega}=0.`} />
      <p>
        Para tener una <strong>verdad de terreno exacta</strong> usamos una solución manufacturada (MMS): se fija el par
        <InlineMath tex={String.raw`(T^*,k^*)`} /> y se <em>deriva</em> la fuente <InlineMath tex={String.raw`q`} /> que lo hace
        cumplir la EDP por construcción: 
      </p>
      <Equation tex={String.raw`T^*=\sin\pi x\,\sin\pi y,\qquad k^*=1+\tfrac12\sin\pi x\,\sin\pi y,\qquad q=\nabla\!\cdot\!\big(k^*\nabla T^*\big).`} />
      <p>
        Aquí <InlineMath tex={String.raw`T^*`} /> ya cumple <InlineMath tex={String.raw`T^*|_{\partial\Omega}=0`} />, y
        <InlineMath tex={String.raw`q`} /> sale en forma cerrada (vía cálculo simbólico). Como el par
        <InlineMath tex={String.raw`(T^*,k^*,q)`} /> satisface la EDP <em>idénticamente</em> por sustitución, el
        <InlineMath tex={String.raw`k^*`} /> es una <strong>ancla analítica</strong> legítima: el error reportado es el
        relativo-<InlineMath tex={String.raw`L^2`} /> del <InlineMath tex={String.raw`k`} /> recuperado contra ese
        <InlineMath tex={String.raw`k^*`} /> verdadero.
      </p>

      <h3>El método: inversión de campo con la SEGUNDA salida de la red</h3>
      <p>
        El truco central es que la conductividad desconocida <strong>es una salida de la red</strong>, no una constante a
        ajustar. La red <InlineMath tex={String.raw`\mathcal{N}_\theta(x,y)=(k_\theta,T_\theta)`} /> produce <em>dos</em>
        campos a la vez, y el residual usa la regla del producto sobre <InlineMath tex={String.raw`\nabla\!\cdot(k\nabla T)`} />:
      </p>
      <Equation tex={String.raw`\mathcal{R}=k\,(T_{xx}+T_{yy})+k_x\,T_x+k_y\,T_y-q,\qquad \mathcal{L}=\|\mathcal{R}\|^2+\lambda\!\!\sum_{i}\big(T_\theta(x_i,y_i)-T_i^{\text{obs}}\big)^2.`} />
      <ul>
        <li><strong>Dos salidas, no un escalar:</strong> una red de ramas separadas (PFNN) emite <InlineMath tex={String.raw`(k,T)`} />; el gradiente de <InlineMath tex={String.raw`k`} /> entra en el residual, así que la EDP <em>conecta</em> ambos campos.</li>
        <li><strong>Positividad dura:</strong> la conductividad física es positiva, así que <InlineMath tex={String.raw`k=\mathrm{softplus}(\cdot)+\varepsilon>0`} /> por construcción: nunca un <InlineMath tex={String.raw`k`} /> negativo sin sentido físico.</li>
        <li><strong>Borde exacto:</strong> <InlineMath tex={String.raw`T_\theta=x(1-x)\,y(1-y)\,\tilde T`} /> se anula en <InlineMath tex={String.raw`\partial\Omega`} /> por construcción: la condición de borde es exacta, sin término de pérdida que compita.</li>
        <li><strong>Datos dispersos:</strong> las <InlineMath tex={String.raw`\sim100`} /> lecturas ruidosas entran como una pérdida de observación ponderada (<InlineMath tex={String.raw`\lambda=100`} />); la EDP <strong>regulariza</strong> y propaga la información de los sensores a todo el dominio.</li>
      </ul>
      <p>
        Sin el prior físico, un centenar de puntos no determinaría un campo 2D completo (problema mal puesto). Con él, la
        red rellena los huecos de forma consistente con la conducción: la razón de fondo por la que &ldquo;los PINN son
        para problemas inversos&rdquo;.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> conducción estacionaria 2D con conductividad variable, recuperación de campo desde
        datos dispersos y ruidosos, ancla MMS exacta. Es <em>sintético</em>: no existe un dataset abierto de campo
        térmico inverso 2D, así que la verdad de terreno es manufacturada (no ajustada a una placa real).
        <strong> Fuera de alcance:</strong> conducción transitoria (dependiente del tiempo), conductividad anisótropa o
        dependiente de la temperatura, fuentes desconocidas, y geometrías complejas. El error es naturalmente más alto
        donde <InlineMath tex={String.raw`|\nabla T|`} /> es pequeño (cerca del borde y en los extremos de
        <InlineMath tex={String.raw`T`} />): ahí la EDP es poco sensible a <InlineMath tex={String.raw`k`} />, así que los
        datos informan menos sobre la conductividad: una limitación honesta, no un defecto del método.
      </p>

      <p>
        <strong>Qué muestra el benchmark.</strong> El campo recuperado <InlineMath tex={String.raw`k(x,y)`} /> reproduce el
        <em> domo</em> central de <InlineMath tex={String.raw`k^*`} /> (máximo en el centro, donde
        <InlineMath tex={String.raw`\sin\pi x\sin\pi y=1`} />, y <InlineMath tex={String.raw`k\to1`} /> hacia los bordes),
        a partir de solo <InlineMath tex={String.raw`\sim100`} /> sensores de temperatura ruidosos. Como la conductividad
        es <em>estacionaria</em> y la verdad MMS es un par fijo <InlineMath tex={String.raw`(T^*,k^*)`} />: sin un mando
        físico que admita una familia paramétrica de inversiones bien puestas con el mismo conjunto de sensores: este
        caso se publica como un <strong>benchmark de una sola variante</strong>, no como un barrido (nunca fabricamos
        regímenes para inflar un contador).
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`k(x,y)`} /> es
        el producto: un domo suave centrado, brillante en el medio y descendiendo hacia
        <InlineMath tex={String.raw`k=1`} /> en el borde. Pasa el cursor para leer la conductividad recuperada en cualquier
        punto y compárala mentalmente con <InlineMath tex={String.raw`k^*=1+\tfrac12\sin\pi x\sin\pi y`} />; los
        <strong> perfiles de corte</strong> en <InlineMath tex={String.raw`x`} /> e <InlineMath tex={String.raw`y`} /> dan la
        campana de la conductividad (su altura <InlineMath tex={String.raw`\approx1.5`} /> en el centro). El error es
        visiblemente mayor cerca del contorno, donde <InlineMath tex={String.raw`T`} /> y su gradiente se anulan. Como es un
        benchmark de parámetro fijo, el tab <strong>Live</strong> re-evalúa la red entrenada (la misma física) en tu
        navegador (onnxruntime-web), sin deslizador de parámetro.
      </p>
    </>
  ): (
    <>
      <h2>The problem: recover a hidden conductivity field from a handful of sensors: 2D inverse conduction</h2>
      <p>
        <strong>The problem.</strong> A plate conducts heat with a thermal conductivity <InlineMath tex={String.raw`k(x,y)`} />
        that we <em>do not know</em> and that varies point to point. We know the heat source
        <InlineMath tex={String.raw`q(x,y)`} /> and we measure the temperature <InlineMath tex={String.raw`T`} /> at just
        <strong> a hundred or so scattered sensors</strong> (noisy). The inverse question is: can we reconstruct the
        <strong> whole field</strong> <InlineMath tex={String.raw`k(x,y)`} /> across the plate: even where there is no
        sensor at all? Here <InlineMath tex={String.raw`k`} /> is not a scalar to fit but an <em>unknown field</em>; this is
        the canonical case where a PINN beats classical FEM/FVM: the <strong>physics</strong> (the PDE) fills the gaps
        between sensors and regularises the reconstruction.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> the plate <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, an <InlineMath tex={String.raw`81\times81`} /> field grid.</li>
        <li><strong>Unknown (the inverse target):</strong> the conductivity field <InlineMath tex={String.raw`k(x,y)>0`} />: what is recovered and scored.</li>
        <li><strong>Auxiliary state:</strong> the temperature <InlineMath tex={String.raw`T(x,y)`} />, also a network output, with <InlineMath tex={String.raw`T=0`} /> on the boundary.</li>
        <li><strong>Known data:</strong> the source <InlineMath tex={String.raw`q(x,y)`} /> (enters the PDE) and <InlineMath tex={String.raw`\sim100`} /> noisy observations <InlineMath tex={String.raw`T(x_i,y_i)`} />.</li>
        <li><strong>Sensor noise:</strong> <InlineMath tex={String.raw`\sigma=0.01`} /> Gaussian on the <InlineMath tex={String.raw`T`} /> readings: the data is not perfect.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        Steady conduction with variable conductivity obeys the divergence-form PDE, with <InlineMath tex={String.raw`T`} />
        pinned to zero on the boundary:
      </p>
      <Equation tex={String.raw`\nabla\!\cdot\!\big(k(x,y)\,\nabla T\big)=q(x,y)\ \text{on}\ (0,1)^2,\qquad T|_{\partial\Omega}=0.`} />
      <p>
        To get an <strong>exact ground truth</strong> we use a manufactured solution (MMS): fix the pair
        <InlineMath tex={String.raw`(T^*,k^*)`} /> and <em>derive</em> the source <InlineMath tex={String.raw`q`} /> that makes
        it satisfy the PDE by construction: 
      </p>
      <Equation tex={String.raw`T^*=\sin\pi x\,\sin\pi y,\qquad k^*=1+\tfrac12\sin\pi x\,\sin\pi y,\qquad q=\nabla\!\cdot\!\big(k^*\nabla T^*\big).`} />
      <p>
        Here <InlineMath tex={String.raw`T^*`} /> already satisfies <InlineMath tex={String.raw`T^*|_{\partial\Omega}=0`} />,
        and <InlineMath tex={String.raw`q`} /> follows in closed form (via symbolic algebra). Because the triple
        <InlineMath tex={String.raw`(T^*,k^*,q)`} /> satisfies the PDE <em>identically</em> by substitution,
        <InlineMath tex={String.raw`k^*`} /> is a legitimate <strong>analytic anchor</strong>: the reported error is the
        relative-<InlineMath tex={String.raw`L^2`} /> of the recovered <InlineMath tex={String.raw`k`} /> against that true
        <InlineMath tex={String.raw`k^*`} />.
      </p>

      <h3>The method: field inversion via the SECOND network output</h3>
      <p>
        The central trick is that the unknown conductivity <strong>is a network output</strong>, not a constant to fit. The
        network <InlineMath tex={String.raw`\mathcal{N}_\theta(x,y)=(k_\theta,T_\theta)`} /> produces <em>two</em> fields at
        once, and the residual uses the product rule on <InlineMath tex={String.raw`\nabla\!\cdot(k\nabla T)`} />:
      </p>
      <Equation tex={String.raw`\mathcal{R}=k\,(T_{xx}+T_{yy})+k_x\,T_x+k_y\,T_y-q,\qquad \mathcal{L}=\|\mathcal{R}\|^2+\lambda\!\!\sum_{i}\big(T_\theta(x_i,y_i)-T_i^{\text{obs}}\big)^2.`} />
      <ul>
        <li><strong>Two outputs, not a scalar:</strong> a parallel-branch network (PFNN) emits <InlineMath tex={String.raw`(k,T)`} />; the gradient of <InlineMath tex={String.raw`k`} /> enters the residual, so the PDE <em>couples</em> both fields.</li>
        <li><strong>Hard positivity:</strong> physical conductivity is positive, so <InlineMath tex={String.raw`k=\mathrm{softplus}(\cdot)+\varepsilon>0`} /> by construction: never a physically meaningless negative <InlineMath tex={String.raw`k`} />.</li>
        <li><strong>Exact boundary:</strong> <InlineMath tex={String.raw`T_\theta=x(1-x)\,y(1-y)\,\tilde T`} /> vanishes on <InlineMath tex={String.raw`\partial\Omega`} /> by construction: the boundary condition is exact, with no competing loss term.</li>
        <li><strong>Sparse data:</strong> the <InlineMath tex={String.raw`\sim100`} /> noisy readings enter as a weighted observation loss (<InlineMath tex={String.raw`\lambda=100`} />); the PDE <strong>regularises</strong> and propagates the sensor information across the whole domain.</li>
      </ul>
      <p>
        Without the physics prior, a hundred points would not determine a full 2D field (an ill-posed problem). With it,
        the network fills the gaps consistently with conduction: the underlying reason &ldquo;PINNs are for inverse
        problems&rdquo;.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> steady 2D conduction with variable conductivity, field recovery from sparse noisy data,
        an exact MMS anchor. It is <em>synthetic</em>: no open 2D thermal-field inverse dataset exists, so the ground truth
        is manufactured (not fit to a real plate). <strong>Out of scope:</strong> transient (time-dependent) conduction,
        anisotropic or temperature-dependent conductivity, unknown sources, and complex geometries. The error is naturally
        larger where <InlineMath tex={String.raw`|\nabla T|`} /> is small (near the boundary and at the extrema of
        <InlineMath tex={String.raw`T`} />): there the PDE is weakly sensitive to <InlineMath tex={String.raw`k`} />, so the
        data inform conductivity less: an honest limitation, not a flaw of the method.
      </p>

      <p>
        <strong>What the benchmark shows.</strong> The recovered field <InlineMath tex={String.raw`k(x,y)`} /> reproduces the
        central <em>dome</em> of <InlineMath tex={String.raw`k^*`} /> (peak at the center, where
        <InlineMath tex={String.raw`\sin\pi x\sin\pi y=1`} />, and <InlineMath tex={String.raw`k\to1`} /> toward the edges),
        from only <InlineMath tex={String.raw`\sim100`} /> noisy temperature sensors. Because the conductivity is
        <em> stationary</em> and the MMS truth is a fixed pair <InlineMath tex={String.raw`(T^*,k^*)`} />: with no physical
        knob that admits a parametric family of well-posed inversions sharing the same sensor set: this case ships as a
        <strong> single-variant benchmark</strong>, not a sweep (we never fabricate regimes to inflate a count).
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of <InlineMath tex={String.raw`k(x,y)`} />
        is the product: a smooth centered dome, bright in the middle and falling toward <InlineMath tex={String.raw`k=1`} />
        at the edge. Hover to read the recovered conductivity anywhere and compare it mentally with
        <InlineMath tex={String.raw`k^*=1+\tfrac12\sin\pi x\sin\pi y`} />; the <strong>line-cut profiles</strong> in
        <InlineMath tex={String.raw`x`} /> and <InlineMath tex={String.raw`y`} /> give the conductivity bell (its height
        <InlineMath tex={String.raw`\approx1.5`} /> at the center). The error is visibly larger near the boundary, where
        <InlineMath tex={String.raw`T`} /> and its gradient vanish. Since it is a fixed-parameter benchmark, the
        <strong> Live</strong> tab re-evaluates the trained network (the same physics) in your browser
        (onnxruntime-web), with no parameter slider.
      </p>
    </>
  );
}
