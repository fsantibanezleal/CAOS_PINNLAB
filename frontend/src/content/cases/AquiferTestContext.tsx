import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for env-aquifer-test (a confined-aquifer pumping test; Cooper-Jacob recovers T,S;
 *  the variants ARE three aquifer types). An honest "where a PINN is NOT the tool" case. */
export function AquiferTestContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>La prueba de bombeo: recuperar T y S de un acuífero</h2>
      <p>
        La transmisividad <InlineMath tex={String.raw`T`} /> y el coeficiente de almacenamiento
        <InlineMath tex={String.raw`S`} /> de un acuífero <strong>no se pueden medir directamente</strong>. El
        método de campo estándar es una <strong>prueba de bombeo</strong>: se bombea un pozo a caudal constante
        <InlineMath tex={String.raw`Q`} />, se observa cómo baja el nivel del agua (el abatimiento
        <InlineMath tex={String.raw`s`} />) en un pozo de observación con el tiempo, y se infieren
        <InlineMath tex={String.raw`T`} /> y <InlineMath tex={String.raw`S`} /> de la curva.
      </p>
      <p>
        Es también un caso honesto de <strong>saber cuándo un PINN NO es la herramienta</strong>: el problema
        tiene forma cerrada exacta, así que el método clásico gana, y un PINN hace mucho peor.
      </p>

      <h3>La física (acuífero confinado)</h3>
      <p>Flujo radial transitorio hacia el pozo, solución de Theis (1935):</p>
      <Equation tex={String.raw`s(r,t)=\frac{Q}{4\pi T}\,W(u),\qquad u=\frac{r^2 S}{4 T t},\qquad W(u)=E_1(u).`} />
      <p>
        <strong>Confinado no es opcional:</strong> en un acuífero libre el espesor saturado cambia y Theis es
        el modelo equivocado (se necesita Boulton/Neuman). El caso lo declara confinado y lo cumple.
      </p>

      <h3>El método que funciona: Cooper-Jacob</h3>
      <p>
        Para tiempo tardío (<InlineMath tex={String.raw`u`} /> pequeño),
        <InlineMath tex={String.raw`W(u)\approx-0.5772-\ln u`} />, así que el abatimiento es <em>lineal en</em>
        <InlineMath tex={String.raw`\ln t`} />:
      </p>
      <Equation tex={String.raw`s\approx\frac{Q}{4\pi T}\ln\!\frac{2.25\,T t}{r^2 S}.`} />
      <p>
        Se ajusta una recta a <InlineMath tex={String.raw`s`} /> vs <InlineMath tex={String.raw`\ln t`} />: la
        <strong> pendiente</strong> da <InlineMath tex={String.raw`T=Q/(4\pi\,\text{pendiente})`} /> y el
        <strong> intercepto</strong> de abatimiento cero da <InlineMath tex={String.raw`S`} />. Verificado en
        tres acuíferos (arena fina, gruesa, grava): <InlineMath tex={String.raw`T`} /> a 0.2-3.2%,
        <InlineMath tex={String.raw`S`} /> a 2.9-9.4%. El ancla analítica de Theis se comprobó contra un solve
        radial numérico a 0.06%.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Campo:</strong> el cono de abatimiento <InlineMath tex={String.raw`s(x,y)`} /> a <InlineMath tex={String.raw`t=1`} /> día en una vista de planta de 1000x1000 m, más profundo en el pozo central.</li>
        <li><strong>Variantes (chips):</strong> tres acuíferos confinados con <InlineMath tex={String.raw`(T,S)`} /> distintos.</li>
        <li><strong>La respuesta:</strong> <InlineMath tex={String.raw`(T,S)`} /> recuperados por Cooper-Jacob vs los verdaderos.</li>
      </ul>

      <h3>Por qué un PINN NO es la herramienta aquí</h3>
      <p>
        Se probó un PINN con los mismos datos y le fue mucho peor: un PINN radial directo colapsó a
        <strong> 82%</strong> de error (el transitorio rígido cerca del pozo más la frontera de flujo es la
        clásica patología de gradientes del PINN), y un inverso de parámetros con PINN recuperó
        <InlineMath tex={String.raw`T=2335`} /> vs 500 m²/día (<strong>367%</strong> de error). El PINN gana su
        lugar solo cuando el acuífero es <em>acotado o heterogéneo</em>, donde no hay forma cerrada y
        Cooper-Jacob no vale. En este problema, recurrir a un PINN es la decisión equivocada; saberlo es el
        punto.
      </p>
    </>
  ) : (
    <>
      <h2>The pumping test: recovering an aquifer's T and S</h2>
      <p>
        An aquifer's transmissivity <InlineMath tex={String.raw`T`} /> and storativity
        <InlineMath tex={String.raw`S`} /> <strong>cannot be measured directly</strong>. The standard field
        method is a <strong>pumping test</strong>: pump a well at constant rate <InlineMath tex={String.raw`Q`} />,
        watch the water level fall (the drawdown <InlineMath tex={String.raw`s`} />) in an observation well over
        time, and infer <InlineMath tex={String.raw`T`} /> and <InlineMath tex={String.raw`S`} /> from the curve.
      </p>
      <p>
        It is also an honest <strong>"know when a PINN is NOT the tool"</strong> case: the problem has an exact
        closed form, so the classical method wins, and a PINN does far worse.
      </p>

      <h3>The physics (confined aquifer)</h3>
      <p>Radial transient flow to the well, the Theis (1935) solution:</p>
      <Equation tex={String.raw`s(r,t)=\frac{Q}{4\pi T}\,W(u),\qquad u=\frac{r^2 S}{4 T t},\qquad W(u)=E_1(u).`} />
      <p>
        <strong>Confined is not optional:</strong> for an unconfined aquifer the saturated thickness changes and
        Theis is the wrong model (Boulton/Neuman is needed). The case states confined and means it.
      </p>

      <h3>The method that works: Cooper-Jacob</h3>
      <p>
        For late time (small <InlineMath tex={String.raw`u`} />),
        <InlineMath tex={String.raw`W(u)\approx-0.5772-\ln u`} />, so the drawdown is <em>linear in</em>
        <InlineMath tex={String.raw`\ln t`} />:
      </p>
      <Equation tex={String.raw`s\approx\frac{Q}{4\pi T}\ln\!\frac{2.25\,T t}{r^2 S}.`} />
      <p>
        Fit a straight line to <InlineMath tex={String.raw`s`} /> vs <InlineMath tex={String.raw`\ln t`} />: the
        <strong> slope</strong> gives <InlineMath tex={String.raw`T=Q/(4\pi\,\text{slope})`} /> and the
        zero-drawdown <strong>intercept</strong> gives <InlineMath tex={String.raw`S`} />. Verified across three
        aquifers (fine sand, coarse sand, gravel): <InlineMath tex={String.raw`T`} /> to 0.2-3.2%,
        <InlineMath tex={String.raw`S`} /> to 2.9-9.4%. The analytic Theis anchor was checked against a numerical
        radial solve to 0.06%.
      </p>

      <h3>Components and variables</h3>
      <ul>
        <li><strong>Field:</strong> the drawdown cone <InlineMath tex={String.raw`s(x,y)`} /> at <InlineMath tex={String.raw`t=1`} /> day over a 1000x1000 m plan view, deepest at the central well.</li>
        <li><strong>Variants (chips):</strong> three confined aquifers with different <InlineMath tex={String.raw`(T,S)`} />.</li>
        <li><strong>The answer:</strong> <InlineMath tex={String.raw`(T,S)`} /> recovered by Cooper-Jacob vs the true values.</li>
      </ul>

      <h3>Why a PINN is NOT the tool here</h3>
      <p>
        A PINN was spiked on the same data and did far worse: a forward radial PINN collapsed to
        <strong> 82%</strong> error (the stiff near-well transient plus the well-flux boundary is the classic
        PINN gradient pathology), and a PINN parameter-inverse recovered
        <InlineMath tex={String.raw`T=2335`} /> vs 500 m²/day (<strong>367%</strong> error). The PINN earns its
        keep only once the aquifer is <em>bounded or heterogeneous</em>, where no closed form exists and
        Cooper-Jacob is invalid. On this problem, reaching for a PINN is the wrong call; knowing that is the
        point.
      </p>
    </>
  );
}
