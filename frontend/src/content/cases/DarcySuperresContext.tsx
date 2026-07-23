import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-darcy-superres (FNO zero-shot super-resolution; the variants ARE the
 *  evaluation resolutions). */
export function DarcySuperresContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>Super-resolución cero-disparo: un operador, cualquier grilla</h2>
      <p>
        Esta es la propiedad que separa a un <strong>operador</strong> neuronal de una red de grilla a grilla
        (una CNN, una U-Net): la <strong>invariancia a la discretización</strong>. Un operador neuronal de
        Fourier actúa sobre <em>modos de Fourier</em>, no sobre píxeles, así que los mismos pesos entrenados
        sirven a <em>cualquier</em> resolución. Un operador entrenado solo con datos gruesos puede evaluarse en
        una grilla más fina que <strong>nunca vio</strong>.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Operador:</strong> el mismo FNO de Darcy, entrenado SOLO en <InlineMath tex={String.raw`32\times32`} />.</li>
        <li><strong>Variantes (chips):</strong> las grillas de evaluación: <InlineMath tex={String.raw`32\times32`} /> (entrenamiento), <InlineMath tex={String.raw`64\times64`} /> y <InlineMath tex={String.raw`96\times96`} /> (nunca vistas).</li>
        <li><strong>Los tres campos:</strong> la predicción del operador, la verdad por diferencias finitas en esa grilla, y el error absoluto.</li>
        <li><strong>El campo de coeficiente</strong> usa una longitud de correlación consistente con la resolución (la <InlineMath tex={String.raw`\sigma`} /> del filtro escala con la grilla): el MISMO proceso físico muestreado más fino, no tres familias distintas.</li>
      </ul>

      <h3>Por qué funciona</h3>
      <p>Cada capa de Fourier del operador aplica</p>
      <Equation tex={String.raw`(\mathcal{K}v)(\mathbf{x})=\mathcal{F}^{-1}\big(R\cdot(\mathcal{F}v)\big)(\mathbf{x}),`} />
      <p>
        donde <InlineMath tex={String.raw`R`} /> son pesos aprendidos sobre un conjunto <em>truncado</em> de
        modos de frecuencia baja. Como <InlineMath tex={String.raw`R`} /> indexa MODOS y no posiciones de
        grilla, la misma transformación se aplica a una entrada muestreada en <InlineMath tex={String.raw`32^2`} />,
        <InlineMath tex={String.raw`64^2`} /> o <InlineMath tex={String.raw`96^2`} />: la FFT y su inversa se
        ajustan a la grilla, los pesos no cambian. Por eso el operador simplemente <em>corre</em> a otra
        resolución, mientras que una CNN, cuyos filtros son locales en píxeles, ni siquiera puede evaluarse
        fuera de su grilla de entrenamiento sin cambiar de arquitectura.
      </p>

      <h3>Lo que medimos</h3>
      <ul>
        <li><strong>32x32</strong> (entrenamiento): el error de referencia.</li>
        <li><strong>64x64</strong> (no vista): el mismo operador, cerca de <strong>1.8x</strong> el error base.</li>
        <li><strong>96x96</strong> (no vista): más fina aún, con la degradación esperable al alejarse de la grilla de entrenamiento.</li>
      </ul>

      <h3>Lectura honesta</h3>
      <ul>
        <li><strong>No es precisión gratis.</strong> El error <em>sube</em> con la resolución: las grillas más finas resuelven detalles que el entrenamiento grueso nunca le enseñó. Es "un operador sirve a muchas grillas", no "el operador mejora en grillas finas".</li>
        <li><strong>La referencia en cada grilla es su propia resolución</strong> por diferencias finitas, así que el error es la discrepancia honesta operador-vs-verdad en ese grid.</li>
        <li><strong>El ONNX exportado es el operador 32x32</strong>; las evaluaciones en grillas más finas son artefactos de campo horneados (el mismo operador torch las produjo en cada grilla).</li>
      </ul>
    </>
  ) : (
    <>
      <h2>Zero-shot super-resolution: one operator, any grid</h2>
      <p>
        This is the property that separates a neural <strong>operator</strong> from a grid-to-grid network (a
        CNN, a U-Net): <strong>discretisation invariance</strong>. A Fourier neural operator acts on
        <em> Fourier modes</em>, not pixels, so the same trained weights apply at <em>any</em> resolution. An
        operator trained only on coarse data can be evaluated on a finer grid it <strong>never saw</strong>.
      </p>

      <h3>Components and variables</h3>
      <ul>
        <li><strong>Operator:</strong> the same Darcy FNO, trained ONLY at <InlineMath tex={String.raw`32\times32`} />.</li>
        <li><strong>Variants (chips):</strong> the evaluation grids: <InlineMath tex={String.raw`32\times32`} /> (training), <InlineMath tex={String.raw`64\times64`} /> and <InlineMath tex={String.raw`96\times96`} /> (never seen).</li>
        <li><strong>The three fields:</strong> the operator's prediction, the finite-difference truth at that grid, and the absolute error.</li>
        <li><strong>The coefficient field</strong> uses a resolution-consistent correlation length (the filter <InlineMath tex={String.raw`\sigma`} /> scales with the grid): the SAME physical process sampled more finely, not three different families.</li>
      </ul>

      <h3>Why it works</h3>
      <p>Each Fourier layer of the operator applies</p>
      <Equation tex={String.raw`(\mathcal{K}v)(\mathbf{x})=\mathcal{F}^{-1}\big(R\cdot(\mathcal{F}v)\big)(\mathbf{x}),`} />
      <p>
        where <InlineMath tex={String.raw`R`} /> are learned weights over a <em>truncated</em> set of
        low-frequency modes. Because <InlineMath tex={String.raw`R`} /> indexes MODES, not grid positions, the
        same transformation applies to an input sampled at <InlineMath tex={String.raw`32^2`} />,
        <InlineMath tex={String.raw`64^2`} /> or <InlineMath tex={String.raw`96^2`} />: the FFT and its inverse
        adapt to the grid, the weights do not change. That is why the operator simply <em>runs</em> at another
        resolution, whereas a CNN, whose filters are local in pixels, cannot even be evaluated off its training
        grid without changing architecture.
      </p>

      <h3>What we measured</h3>
      <ul>
        <li><strong>32x32</strong> (training): the baseline error.</li>
        <li><strong>64x64</strong> (unseen): the same operator, about <strong>1.8x</strong> the baseline error.</li>
        <li><strong>96x96</strong> (unseen): finer still, with the degradation expected far from the training grid.</li>
      </ul>

      <h3>Honest reading</h3>
      <ul>
        <li><strong>Not free accuracy.</strong> The error <em>rises</em> with resolution: finer grids resolve features the coarse training never taught it. It is "one operator serves many grids", not "the operator gets better on finer grids".</li>
        <li><strong>The reference at each grid is its own finite-difference solve</strong>, so the error is the honest operator-vs-truth discrepancy there.</li>
        <li><strong>The exported ONNX is the 32x32 operator</strong>; the finer-grid evaluations are baked field artifacts (the same torch operator produced them at each grid).</li>
      </ul>
    </>
  );
}
