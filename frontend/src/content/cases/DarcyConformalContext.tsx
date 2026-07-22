import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-darcy-conformal (split conformal prediction on the Darcy operator; the
 *  variants ARE the coverage targets). */
export function DarcyConformalContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>Predicción conforme: una barra de error honesta para el operador</h2>
      <p>
        Los otros dos casos de Darcy entrenan un operador y reportan un único número de error retenido. Pero
        eso no dice nada sobre el <em>próximo</em> campo. Un sustituto desplegado necesita una
        <strong> garantía sobre la instancia que aún no ha visto</strong>. La predicción conforme por división
        la da: una banda con una probabilidad de cobertura declarada, <strong>sin suponer ninguna
        distribución de error</strong> más allá de la intercambiabilidad, y sin reentrenar.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Operador:</strong> el mismo FNO de Darcy, <InlineMath tex={String.raw`\mathcal{G}:a\mapsto u`} />.</li>
        <li><strong>Conjunto de calibración:</strong> 128 campos que el operador nunca vio al entrenar (aparte del conjunto de prueba de 200).</li>
        <li><strong>Los cuatro campos mostrados:</strong> la predicción <InlineMath tex={String.raw`\mathcal{G}(a)`} />, la verdad <InlineMath tex={String.raw`u`} />, el error absoluto, y la máscara <em>dentro-de-banda</em> (1 donde la banda contiene la verdad).</li>
        <li><strong>Variantes (chips):</strong> los objetivos de cobertura <InlineMath tex={String.raw`1-\alpha`} /> (80%, 90%, 95%).</li>
      </ul>

      <h3>Formalización</h3>
      <p>La receta tiene tres pasos y no toca los pesos:</p>
      <ol>
        <li>En la calibración, puntúa cada instancia por su peor error de píxel: <InlineMath tex={String.raw`s_i=\max_{\mathbf{x}}|\mathcal{G}(a_i)(\mathbf{x})-u_i(\mathbf{x})|`} />.</li>
        <li>Toma el cuantil de muestra finita <InlineMath tex={String.raw`q=\lceil (n+1)(1-\alpha)\rceil/n`} /> de los <InlineMath tex={String.raw`\{s_i\}`} />.</li>
        <li>Para un campo nuevo, la banda <InlineMath tex={String.raw`\mathcal{G}(a)\pm q`} /> contiene el campo verdadero entero con probabilidad al menos <InlineMath tex={String.raw`1-\alpha`} />.</li>
      </ol>
      <Equation tex={String.raw`\mathbb{P}\Big(\max_{\mathbf{x}}\,|\mathcal{G}(a)(\mathbf{x})-u(\mathbf{x})|\le q\Big)\ \ge\ 1-\alpha.`} />
      <p>La garantía es <em>marginal</em> (sobre el sorteo de calibración y prueba) y <strong>libre de distribución</strong>: no supone que los errores sean gaussianos ni nada.</p>

      <h3>Lo que medimos</h3>
      <p>Para cada objetivo, la <strong>cobertura empírica</strong> sobre los 200 campos de prueba no vistos:</p>
      <ul>
        <li>Objetivo 80% → logrado <strong>87.5%</strong></li>
        <li>Objetivo 90% → logrado <strong>96.5%</strong></li>
        <li>Objetivo 95% → logrado <strong>99.5%</strong></li>
      </ul>
      <p>A la altura o por encima del objetivo cada vez, justo como exige la garantía: es una <em>cota inferior</em>, así que la sobre-cobertura es esperable con un score de campo entero (la banda se dimensiona para el píxel más difícil).</p>

      <h3>Límites honestos (el punto del caso)</h3>
      <ul>
        <li><strong>Cobertura marginal bajo intercambiabilidad.</strong> El campo nuevo debe venir de la misma distribución que la calibración. Otra familia de coeficientes rompe la intercambiabilidad y <strong>anula</strong> la garantía: justo el cuidado que un sustituto necesita adosado.</li>
        <li><strong>Un solo ancho, no incertidumbre por píxel.</strong> La banda es lo bastante ancha para el píxel más difícil, así que sobre-cubre el interior fácil. Un score normalizado la ajustaría, a costa de transparencia.</li>
        <li><strong>Cuantifica el error propio del operador.</strong> No hace correcto a un operador equivocado; mide su discrepancia contra el solucionador de referencia.</li>
      </ul>
    </>
  ) : (
    <>
      <h2>Conformal prediction: an honest error bar for the operator</h2>
      <p>
        The other two Darcy cases train an operator and report a single held-out error number. But that says
        nothing about the <em>next</em> field. A deployed surrogate needs a
        <strong> guarantee on the instance it has not yet seen</strong>. Split conformal prediction gives one:
        a band with a stated coverage probability, <strong>assuming no error distribution</strong> beyond
        exchangeability, and with no retraining.
      </p>

      <h3>Components and variables</h3>
      <ul>
        <li><strong>Operator:</strong> the same Darcy FNO, <InlineMath tex={String.raw`\mathcal{G}:a\mapsto u`} />.</li>
        <li><strong>Calibration set:</strong> 128 fields the operator never saw in training (separate from the 200-field test set).</li>
        <li><strong>The four fields shown:</strong> the prediction <InlineMath tex={String.raw`\mathcal{G}(a)`} />, the truth <InlineMath tex={String.raw`u`} />, the absolute error, and the <em>in-band</em> mask (1 where the band contains the truth).</li>
        <li><strong>Variants (chips):</strong> the coverage targets <InlineMath tex={String.raw`1-\alpha`} /> (80%, 90%, 95%).</li>
      </ul>

      <h3>Formalisation</h3>
      <p>The recipe is three steps and touches no weights:</p>
      <ol>
        <li>On calibration, score each instance by its worst pixel error: <InlineMath tex={String.raw`s_i=\max_{\mathbf{x}}|\mathcal{G}(a_i)(\mathbf{x})-u_i(\mathbf{x})|`} />.</li>
        <li>Take the finite-sample quantile <InlineMath tex={String.raw`q=\lceil (n+1)(1-\alpha)\rceil/n`} /> of the <InlineMath tex={String.raw`\{s_i\}`} />.</li>
        <li>For a new field, the band <InlineMath tex={String.raw`\mathcal{G}(a)\pm q`} /> contains the whole true field with probability at least <InlineMath tex={String.raw`1-\alpha`} />.</li>
      </ol>
      <Equation tex={String.raw`\mathbb{P}\Big(\max_{\mathbf{x}}\,|\mathcal{G}(a)(\mathbf{x})-u(\mathbf{x})|\le q\Big)\ \ge\ 1-\alpha.`} />
      <p>The guarantee is <em>marginal</em> (over the draw of calibration and test) and <strong>distribution-free</strong>: it does not assume the errors are Gaussian or anything else.</p>

      <h3>What we measured</h3>
      <p>For each target, the <strong>empirical coverage</strong> over the 200 unseen test fields:</p>
      <ul>
        <li>Target 80% → achieved <strong>87.5%</strong></li>
        <li>Target 90% → achieved <strong>96.5%</strong></li>
        <li>Target 95% → achieved <strong>99.5%</strong></li>
      </ul>
      <p>At or above target every time, exactly as the guarantee requires: it is a <em>lower bound</em>, so over-coverage is expected from a whole-field score (the band is sized for the hardest pixel).</p>

      <h3>Honest limits (the point of the case)</h3>
      <ul>
        <li><strong>Marginal coverage under exchangeability.</strong> The new field must come from the same distribution as calibration. A different coefficient family breaks exchangeability and <strong>voids</strong> the guarantee: exactly the caveat a surrogate needs stapled to it.</li>
        <li><strong>One width, not per-pixel uncertainty.</strong> The band is wide enough for the hardest pixel, so it over-covers the easy interior. A normalised score would tighten it, at the cost of transparency.</li>
        <li><strong>It quantifies the operator's own error.</strong> It cannot make a wrong operator right; it measures its discrepancy against the reference solver.</li>
      </ul>
    </>
  );
}
