import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for env-soil-heat-real (inverse thermal-diffusivity recovery from REAL USCRN soil
 *  temperatures; the single validated-real benchmark — recover alpha, validate out-of-sample). */
export function SoilHeatRealContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: recuperar la difusividad térmica del suelo a partir de temperaturas REALES</h2>
      <p>
        <strong>El problema.</strong> El calor estacional entra por la superficie del suelo y se propaga hacia abajo
        por <em>conducción difusiva</em>. La onda térmica se <strong>amortigua</strong> y se <strong>retrasa en
        fase</strong> con la profundidad: a 100 cm, el verano llega semanas más tarde y mucho más suave que a 5 cm. La
        física es la ecuación del calor 1D, <InlineMath tex={String.raw`T_t=\alpha\,T_{zz}`} />, con una única constante
        material desconocida: la <strong>difusividad térmica</strong> <InlineMath tex={String.raw`\alpha`} />. Este es el
        <strong> único caso entrenado contra un conjunto de datos REAL medido</strong>: temperaturas diarias de suelo de
        la red NOAA USCRN. Planteamos un problema <strong>inverso</strong> — tomamos dos sensores como contornos reales,
        recuperamos <InlineMath tex={String.raw`\alpha`} />, y validamos contra sensores interiores que el optimizador
        nunca vio.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> profundidad <InlineMath tex={String.raw`z`} /> de 5 a 100 cm (normalizada a <InlineMath tex={String.raw`[0,1]`} />) × tiempo <InlineMath tex={String.raw`t`} /> sobre 2019-2021 (estación IL_Champaign_9_SW).</li>
        <li><strong>Incógnita de campo:</strong> la temperatura <InlineMath tex={String.raw`T(z,t)`} /> (normalizada; el heatmap del campo es <InlineMath tex={String.raw`49\times81`} />).</li>
        <li><strong>Incógnita escalar (lo inverso):</strong> la difusividad térmica efectiva <InlineMath tex={String.raw`\alpha`} /> — un <em>parámetro entrenable</em>, no un input.</li>
        <li><strong>Contornos REALES (Dirichlet):</strong> los sensores de 5 cm y 100 cm imponen <InlineMath tex={String.raw`T`} /> en <InlineMath tex={String.raw`z=0`} /> y <InlineMath tex={String.raw`z=1`} /> como series de tiempo medidas.</li>
        <li><strong>Ancla de validación (held-out):</strong> los sensores de 10, 20 y 50 cm — profundidades interiores <em>nunca</em> mostradas al optimizador.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Trabajamos en variables normalizadas: <InlineMath tex={String.raw`z\in[0,1]`} /> mapea 5-100 cm sobre una
        longitud física <InlineMath tex={String.raw`L`} />, y <InlineMath tex={String.raw`t\in[0,1]`} /> mapea el lapso
        total <InlineMath tex={String.raw`\tau`} />. La conducción de calor con un coeficiente normalizado adimensional
        <InlineMath tex={String.raw`\kappa`} /> es:
      </p>
      <Equation tex={String.raw`T_t=\kappa\,T_{zz},\qquad T(0,t)=T_{5\,\mathrm{cm}}(t),\quad T(1,t)=T_{100\,\mathrm{cm}}(t),`} />
      <p>
        donde los dos contornos son los sensores reales. Para mantener <InlineMath tex={String.raw`\kappa>0`} /> de forma
        estable, se entrena <InlineMath tex={String.raw`\log\kappa`} /> como variable de DeepXDE y se penaliza el residual
        de la PDE más el ajuste a los contornos y al perfil inicial. La difusividad física se recupera al final
        deshaciendo la adimensionalización:
      </p>
      <Equation tex={String.raw`\alpha=\kappa\,\frac{L^2}{\tau}\quad[\mathrm{m^2/s}],\qquad \text{reportada en } \mathrm{mm^2/s}.`} />
      <p>
        El rango físico esperado para suelos minerales típicos es
        <InlineMath tex={String.raw`\alpha\approx0.2\text{-}0.8\,\mathrm{mm^2/s}`} />; un valor recuperado fuera de esa
        banda sería una señal de alerta. La métrica honesta NO es el residual sino el <strong>error fuera de
        muestra</strong>: el L2 relativo y el RMSE en grados Celsius contra las temperaturas interiores reales de 10/20/50
        cm.
      </p>

      <h3>El método: inverso con datos reales y validación fuera de muestra</h3>
      <p>
        Un PINN inverso aprende <em>simultáneamente</em> el campo <InlineMath tex={String.raw`T_\theta(z,t)`} /> y el
        coeficiente <InlineMath tex={String.raw`\kappa`} />, minimizando una pérdida compuesta:
      </p>
      <Equation tex={String.raw`\mathcal{L}=w_r\,\big\|T_t-\kappa\,T_{zz}\big\|^2 + w_b\,\big\|T_\theta-T_{\text{bordes}}\big\|^2 + w_0\,\big\|T_\theta-T_{\text{perfil }t=0}\big\|^2.`} />
      <ul>
        <li><strong>Solo Adam (sin L-BFGS):</strong> la inversión de un escalar entrenable es frágil bajo L-BFGS, que puede descartar la variable; Adam mantiene <InlineMath tex={String.raw`\log\kappa`} /> en el grafo todo el entrenamiento.</li>
        <li><strong>Contornos puntuales reales:</strong> los sensores de 5 y 100 cm entran como restricciones en conjuntos de puntos (los instantes medidos), no como funciones analíticas.</li>
        <li><strong>Honestidad por construcción:</strong> los sensores de 10/20/50 cm se excluyen del entrenamiento; el puntaje es la diferencia entre la predicción del campo a esas profundidades y la medición real — una verdadera prueba <em>out-of-sample</em>.</li>
      </ul>
      <p>
        Esto es <strong>validated-real</strong>: no hay verdad fabricada. Tanto los contornos como el ancla son
        mediciones físicas; el único número inventado sería <InlineMath tex={String.raw`\alpha`} />, y precisamente ese es
        el que el método debe <em>descubrir</em> y que luego se contrasta con la banda física conocida.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> conducción de calor 1D vertical con una difusividad efectiva única y constante,
        contornos Dirichlet reales en la cima y el fondo de la columna, y un perfil inicial interpolado de los cinco
        sensores. <strong>Fuera de alcance:</strong> difusividad que varíe con la profundidad o la humedad (perfiles
        estratificados), advección de agua de infiltración, calor latente por congelación/deshielo, y el contenido de
        agua del suelo (que modula <InlineMath tex={String.raw`\alpha`} /> estacionalmente). Por eso <strong>no existe
        una familia paramétrica en forma cerrada</strong>: el forzamiento superficial es una serie temporal real sin
        solución analítica, y <InlineMath tex={String.raw`\alpha`} /> es lo que se recupera, no un mando que se barra.
        Este caso se publica honestamente como un <strong>único benchmark validado-real</strong>, no como un barrido de
        regímenes.
      </p>

      <p>
        <strong>Qué muestra el benchmark.</strong> El campo recuperado <InlineMath tex={String.raw`T(z,t)`} /> es la
        firma clásica de la conducción subterránea: una <strong>onda estacional amortiguada y desfasada</strong>. Cerca
        de la superficie (<InlineMath tex={String.raw`z\to0`} />) la oscilación anual es amplia y sigue de cerca al aire;
        en profundidad (<InlineMath tex={String.raw`z\to1`} />) la misma onda se aplana y se retrasa varias semanas. El
        contraste amplitud-arriba / amplitud-abajo es exactamente lo que codifica <InlineMath tex={String.raw`\alpha`} />:
        el método lo recupera en la banda física correcta (<InlineMath tex={String.raw`\sim0.3\,\mathrm{mm^2/s}`} />) y
        reproduce los sensores interiores ocultos con un error de orden <InlineMath tex={String.raw`1\,^\circ\mathrm{C}`} />
        RMSE — sin haberlos visto nunca.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`T(z,t)`} />
        (profundidad en un eje, tiempo en el otro) muestra bandas cálidas y frías que se inclinan con la profundidad —
        esa inclinación <em>es</em> el retraso de fase, y el desvanecimiento del contraste hacia abajo <em>es</em> el
        amortiguamiento. Pasa el cursor para leer la temperatura exacta en cualquier (profundidad, fecha); mira el
        <strong> perfil de corte</strong> a profundidad fija (la onda anual, cada vez más suave al bajar) y a fecha fija
        (cómo cae la temperatura con la profundidad en invierno y sube en verano). Como es un benchmark de parámetro
        fijo, el tab <strong>Live</strong> re-evalúa la red entrenada (la misma física, vía onnxruntime-web), sin
        deslizador de parámetro; el valor recuperado de <InlineMath tex={String.raw`\alpha`} /> y los RMSE fuera de
        muestra acompañan al campo como las cifras honestas del caso.
      </p>
    </>
  ) : (
    <>
      <h2>The problem: recover the soil's thermal diffusivity from REAL temperatures</h2>
      <p>
        <strong>The problem.</strong> Seasonal heat enters at the soil surface and propagates downward by
        <em> diffusive conduction</em>. The thermal wave is <strong>damped</strong> and <strong>phase-lagged</strong>
        with depth: at 100 cm, summer arrives weeks later and far smoother than at 5 cm. The physics is the 1D heat
        equation, <InlineMath tex={String.raw`T_t=\alpha\,T_{zz}`} />, with a single unknown material constant: the
        <strong> thermal diffusivity</strong> <InlineMath tex={String.raw`\alpha`} />. This is the <strong>only case
        trained against a REAL measured dataset</strong>: daily soil temperatures from NOAA's U.S. Climate Reference
        Network. We pose an <strong>inverse</strong> problem — take two sensors as real boundaries, recover
        <InlineMath tex={String.raw`\alpha`} />, and validate against interior sensors the optimizer never saw.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> depth <InlineMath tex={String.raw`z`} /> from 5 to 100 cm (normalised to <InlineMath tex={String.raw`[0,1]`} />) × time <InlineMath tex={String.raw`t`} /> over 2019-2021 (station IL_Champaign_9_SW).</li>
        <li><strong>Field unknown:</strong> the temperature <InlineMath tex={String.raw`T(z,t)`} /> (normalised; the field heatmap is <InlineMath tex={String.raw`49\times81`} />).</li>
        <li><strong>Scalar unknown (the inverse part):</strong> the effective thermal diffusivity <InlineMath tex={String.raw`\alpha`} /> — a <em>trainable parameter</em>, not an input.</li>
        <li><strong>REAL boundaries (Dirichlet):</strong> the 5 cm and 100 cm sensors impose <InlineMath tex={String.raw`T`} /> at <InlineMath tex={String.raw`z=0`} /> and <InlineMath tex={String.raw`z=1`} /> as measured time series.</li>
        <li><strong>Validation anchor (held-out):</strong> the 10, 20 and 50 cm sensors — interior depths <em>never</em> shown to the optimizer.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        We work in normalised variables: <InlineMath tex={String.raw`z\in[0,1]`} /> maps 5-100 cm over a physical length
        <InlineMath tex={String.raw`L`} />, and <InlineMath tex={String.raw`t\in[0,1]`} /> maps the full span
        <InlineMath tex={String.raw`\tau`} />. Heat conduction with a dimensionless normalised coefficient
        <InlineMath tex={String.raw`\kappa`} /> reads:
      </p>
      <Equation tex={String.raw`T_t=\kappa\,T_{zz},\qquad T(0,t)=T_{5\,\mathrm{cm}}(t),\quad T(1,t)=T_{100\,\mathrm{cm}}(t),`} />
      <p>
        where the two boundaries are the real sensors. To keep <InlineMath tex={String.raw`\kappa>0`} /> stably, we train
        <InlineMath tex={String.raw`\log\kappa`} /> as a DeepXDE variable and penalise the PDE residual plus the fit to
        the boundaries and the initial profile. The physical diffusivity is recovered at the end by undoing the
        non-dimensionalisation:
      </p>
      <Equation tex={String.raw`\alpha=\kappa\,\frac{L^2}{\tau}\quad[\mathrm{m^2/s}],\qquad \text{reported in } \mathrm{mm^2/s}.`} />
      <p>
        The expected physical range for typical mineral soils is
        <InlineMath tex={String.raw`\alpha\approx0.2\text{-}0.8\,\mathrm{mm^2/s}`} />; a recovered value outside that band
        would be a red flag. The honest metric is NOT the residual but the <strong>out-of-sample error</strong>: the
        relative-L2 and the RMSE in degrees Celsius against the real 10/20/50 cm interior temperatures.
      </p>

      <h3>The method: inverse with real data and out-of-sample validation</h3>
      <p>
        An inverse PINN learns the field <InlineMath tex={String.raw`T_\theta(z,t)`} /> and the coefficient
        <InlineMath tex={String.raw`\kappa`} /> <em>simultaneously</em>, minimising a composite loss:
      </p>
      <Equation tex={String.raw`\mathcal{L}=w_r\,\big\|T_t-\kappa\,T_{zz}\big\|^2 + w_b\,\big\|T_\theta-T_{\text{boundaries}}\big\|^2 + w_0\,\big\|T_\theta-T_{\text{profile }t=0}\big\|^2.`} />
      <ul>
        <li><strong>Adam only (no L-BFGS):</strong> inverting a single trainable scalar is fragile under L-BFGS, which can drop the variable; Adam keeps <InlineMath tex={String.raw`\log\kappa`} /> in the graph for the whole run.</li>
        <li><strong>Real point-set boundaries:</strong> the 5 and 100 cm sensors enter as point-set constraints (the measured instants), not as analytic functions.</li>
        <li><strong>Honesty by construction:</strong> the 10/20/50 cm sensors are excluded from training; the score is the gap between the field prediction at those depths and the real measurement — a genuine <em>out-of-sample</em> test.</li>
      </ul>
      <p>
        This is <strong>validated-real</strong>: there is no fabricated truth. Both the boundaries and the anchor are
        physical measurements; the only "invented" number would be <InlineMath tex={String.raw`\alpha`} />, and that is
        precisely the quantity the method must <em>discover</em> and that is then checked against the known physical band.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> 1D vertical heat conduction with a single constant effective diffusivity, real
        Dirichlet boundaries at the top and bottom of the column, and an initial profile interpolated from the five
        sensors. <strong>Out of scope:</strong> depth- or moisture-dependent diffusivity (stratified profiles),
        advection of infiltrating water, latent heat of freeze/thaw, and the soil water content (which modulates
        <InlineMath tex={String.raw`\alpha`} /> seasonally). That is why <strong>no closed-form parametric family
        exists</strong>: the surface forcing is a real time series with no analytic solution, and
        <InlineMath tex={String.raw`\alpha`} /> is what we recover, not a knob to sweep. This case ships honestly as a
        <strong> single validated-real benchmark</strong>, not as a regime sweep.
      </p>

      <p>
        <strong>What the benchmark shows.</strong> The recovered field <InlineMath tex={String.raw`T(z,t)`} /> is the
        classic signature of subsurface conduction: a <strong>damped, phase-lagged seasonal wave</strong>. Near the
        surface (<InlineMath tex={String.raw`z\to0`} />) the annual swing is large and tracks the air closely; at depth
        (<InlineMath tex={String.raw`z\to1`} />) the same wave flattens and lags by several weeks. The top-amplitude /
        bottom-amplitude contrast is exactly what <InlineMath tex={String.raw`\alpha`} /> encodes: the method recovers it
        in the correct physical band (<InlineMath tex={String.raw`\sim0.3\,\mathrm{mm^2/s}`} />) and reproduces the hidden
        interior sensors to about <InlineMath tex={String.raw`1\,^\circ\mathrm{C}`} /> RMSE — having never seen them.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`T(z,t)`} /> (depth on one axis, time on the other) shows warm and cold bands that
        tilt with depth — that tilt <em>is</em> the phase lag, and the fading contrast going downward <em>is</em> the
        damping. Hover to read the exact temperature at any (depth, date); watch the <strong>line-cut profile</strong> at
        fixed depth (the annual wave, ever smoother as you go down) and at a fixed date (how temperature falls with depth
        in winter and rises in summer). Since it is a fixed-parameter benchmark, the <strong>Live</strong> tab
        re-evaluates the trained network (the same physics, via onnxruntime-web), with no parameter slider; the recovered
        <InlineMath tex={String.raw`\alpha`} /> and the out-of-sample RMSEs accompany the field as the honest figures of
        the case.
      </p>
    </>
  );
}
