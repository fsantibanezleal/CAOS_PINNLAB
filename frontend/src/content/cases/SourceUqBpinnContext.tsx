import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for poll-source-uq-bpinn (deep-ensemble Bayesian PINN: pollutant diffusion with epistemic
 *  uncertainty from sparse noisy sensors; single honest UQ benchmark: mean + std emitted as ONE [mean,std] graph). */
export function SourceUqBpinnContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: estimar un contaminante con pocos sensores ruidosos: y reportar la incertidumbre</h2>
      <p>
        <strong>El problema.</strong> Un contaminante disuelto difunde en un canal 1D y se gobierna por la ecuación del
        calor <InlineMath tex={String.raw`c_t = D\,c_{xx}`} />, con paredes limpias <InlineMath tex={String.raw`c=0`} />.
        Pero <em>no</em> conocemos la condición inicial completa: solo disponemos de un <strong>puñado de lecturas de
        sensores dispersas y ruidosas</strong>. Una PINN normal entregaría <em>una</em> respuesta puntual, sin barras de
        error: peligroso para una decisión ambiental. Aquí entrenamos un <strong>ensemble profundo</strong> de
        <InlineMath tex={String.raw`K`} /> PINNs inicializadas de forma independiente: su <em>media</em> estima el campo
        y su <em>desviación estándar</em> es la <strong>incertidumbre epistémica</strong>: pequeña donde hay datos o
        física dura (las paredes), y <strong>creciente donde los datos escasean</strong>.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> canal <InlineMath tex={String.raw`x\in[0,1]`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`61\times61`} />.</li>
        <li><strong>Incógnita:</strong> la concentración <InlineMath tex={String.raw`c(x,t)`} /> del contaminante.</li>
        <li><strong>Difusividad:</strong> <InlineMath tex={String.raw`D=0.1`} />: fija; controla cuán rápido decae el modo.</li>
        <li><strong>Datos:</strong> <InlineMath tex={String.raw`N=24`} /> sensores dispersos con ruido gaussiano <InlineMath tex={String.raw`\sigma=0.02`} /> (no la condición inicial completa).</li>
        <li><strong>Salidas de la red:</strong> dos campos: la <em>media</em> predictiva <InlineMath tex={String.raw`\mu(x,t)`} /> y la <em>desviación</em> del ensemble <InlineMath tex={String.raw`s(x,t)`} /> (la UQ).</li>
        <li><strong>Miembros del ensemble:</strong> <InlineMath tex={String.raw`K=5`} /> redes con inicialización distinta y <em>bagging</em> de sensores.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        El campo de referencia es el <strong>modo fundamental de difusión</strong>: solución exacta del problema con
        <InlineMath tex={String.raw`c(x,0)=\sin(\pi x)`} /> y paredes <InlineMath tex={String.raw`c=0`} />:
      </p>
      <Equation tex={String.raw`c^*(x,t)=e^{-D\pi^2 t}\,\sin(\pi x),\qquad c^*_t = -D\pi^2\,c^*,\quad c^*_{xx} = -\pi^2\,c^* \;\Rightarrow\; c^*_t = D\,c^*_{xx}.`} />
      <p>
        Es nuestra <strong>ancla de validación</strong> exacta: el puntaje primario es el <InlineMath tex={String.raw`L^2`} />
        relativo de la media <InlineMath tex={String.raw`\mu`} /> contra <InlineMath tex={String.raw`c^*`} />. Cada
        miembro <InlineMath tex={String.raw`f_{\theta_k}`} /> minimiza el residual de la PDE en puntos de colocación más
        un término de datos en sus sensores, y las paredes se imponen de forma <em>dura</em> mediante una transformación
        de salida que se anula en <InlineMath tex={String.raw`x=0,1`} />:
      </p>
      <Equation tex={String.raw`f_{\theta_k}(x,t) = x\,(1-x)\,\mathcal{N}_{\theta_k}(x,t)\;\Rightarrow\; f_{\theta_k}\big|_{x=0,1}=0\ \text{(exacto, sin pérdida de BC).}`} />
      <p>
        La media y la incertidumbre del ensemble son simplemente los estadísticos de los <InlineMath tex={String.raw`K`} />
        miembros, evaluados en cualquier punto de consulta:
      </p>
      <Equation tex={String.raw`\mu(x,t)=\frac{1}{K}\sum_{k=1}^{K} f_{\theta_k}(x,t),\qquad s(x,t)=\sqrt{\frac{1}{K}\sum_{k=1}^{K}\big(f_{\theta_k}-\mu\big)^2}.`} />

      <h3>El método: ensemble profundo como aproximación bayesiana</h3>
      <p>
        Inferir la <em>posterior</em> bayesiana exacta sobre los pesos (HMC, SVGD) es caro. El <strong>ensemble
        profundo</strong> (Lakshminarayanan et al., NeurIPS 2017; arXiv:1612.01474,
        doi:10.48550/arXiv.1612.01474) es la
        aproximación reconocida y barata: entrenar <InlineMath tex={String.raw`K`} /> redes con <em>inicializaciones
        distintas</em> y datos remuestreados (<em>bagging</em>) hace que <strong>discrepen más donde no hay datos que las
        limiten</strong>: y esa discrepancia <em>es</em> la incertidumbre epistémica. Cada miembro ve un subconjunto
        bootstrap de los sensores con su propia realización de ruido; las paredes duras y la física compartida los
        anclan donde la teoría manda, y los liberan donde no.
      </p>
      <p>
        Todo el ensemble se exporta como <strong>un único grafo ONNX</strong> que emite <InlineMath tex={String.raw`[\mu,\,s]`} />
        por punto de consulta, así la capa interactiva del navegador carga un solo archivo y re-evalúa media e
        incertidumbre en vivo (onnxruntime-web). La calidad de la UQ se reporta como <strong>calibración a 2σ</strong>:
        la fracción de la grilla donde <InlineMath tex={String.raw`|\mu - c^*|\le 2s`} /> (un ensemble bien calibrado
        cubre <InlineMath tex={String.raw`\sim 95\%`} />).
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> difusión 1D con difusividad fija, paredes Dirichlet duras, un campo de referencia
        analítico, y UQ <em>epistémica</em> por discrepancia de ensemble a partir de sensores dispersos y ruidosos. Es
        <em> ilustrativo-sintético</em>: un demostrador de UQ sobre un campo manufacturado (la <InlineMath tex={String.raw`c^*`} />
        analítica), <strong>no</strong> un dataset medido ni un caso de calibración real. <strong>Fuera de alcance:</strong>
        UQ <em>aleatórica</em> formal (ruido de observación modelado como verosimilitud), la posterior bayesiana completa
        (esto es un proxy de ensemble, no HMC/VI), difusividad <InlineMath tex={String.raw`D`} /> desconocida o un
        problema inverso, y geometría 2-D/3-D. Por eso el caso se publica como un <strong>benchmark único y honesto</strong>:
        no es una familia paramétrica con forma cerrada en un perilla física, sino una demostración de UQ a un nivel de
        ruido y dispersión fijos: inventar regímenes para llenar chips sería deshonesto.
      </p>

      <p>
        <strong>Qué muestra el benchmark.</strong> El campo de la <strong>media</strong> reproduce el modo fundamental:
        una media-onda seno en <InlineMath tex={String.raw`x`} /> que se <em>desvanece exponencialmente</em> en
        <InlineMath tex={String.raw`t`} /> (factor <InlineMath tex={String.raw`e^{-D\pi^2 t}`} />). El campo de la
        <strong> incertidumbre</strong> cuenta la historia complementaria: es <em>casi cero en las paredes</em> (la
        física dura no deja margen) y <em>cerca de los sensores</em>, y <strong>crece en las regiones sin datos</strong>
      : típicamente la franja interior a tiempos intermedios, lejos de cualquier lectura. Ese contraste es la lección:
        <em> dónde confiar</em> en la estimación y <em>dónde colocar el próximo sensor</em>.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> por defecto muestra la media
        <InlineMath tex={String.raw`\mu(x,t)`} /> (x horizontal, t vertical); pasa el cursor para leer el valor exacto y
        ve cómo el seno se aplana al subir <InlineMath tex={String.raw`t`} />. Cambia al campo de
        <strong> incertidumbre</strong> <InlineMath tex={String.raw`s(x,t)`} /> para ver las paredes y los sensores como
        valles oscuros y las zonas ciegas como crestas brillantes. Los <strong>perfiles de corte</strong> en
        <InlineMath tex={String.raw`x`} /> y <InlineMath tex={String.raw`t`} /> permiten comparar media e incertidumbre
        lado a lado: donde la banda <InlineMath tex={String.raw`\mu\pm 2s`} /> envuelve a <InlineMath tex={String.raw`c^*`} />,
        el ensemble está calibrado. Como es un benchmark único, el tab <strong>Live</strong> re-evalúa el ensemble
        entrenado (la misma física y los mismos sensores), emitiendo media e incertidumbre en tu navegador, sin
        deslizador de parámetro.
      </p>
    </>
  ): (
    <>
      <h2>The problem: estimate a pollutant from a few noisy sensors: and report the uncertainty</h2>
      <p>
        <strong>The problem.</strong> A dissolved pollutant diffuses in a 1D channel, governed by the heat equation
        <InlineMath tex={String.raw`c_t = D\,c_{xx}`} />, with clean walls <InlineMath tex={String.raw`c=0`} />. But we
        do <em>not</em> know the full initial condition: we only have a <strong>handful of sparse, noisy sensor
        readings</strong>. An ordinary PINN would return <em>one</em> point answer with no error bars: dangerous for an
        environmental decision. Here we train a <strong>deep ensemble</strong> of <InlineMath tex={String.raw`K`} />
        independently-initialised PINNs: their <em>mean</em> estimates the field and their <em>standard deviation</em> is
        the <strong>epistemic uncertainty</strong>: small where there is data or hard physics (the walls), and
        <strong> growing where data is sparse</strong>.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> channel <InlineMath tex={String.raw`x\in[0,1]`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, a <InlineMath tex={String.raw`61\times61`} /> field grid.</li>
        <li><strong>Unknown:</strong> the pollutant concentration <InlineMath tex={String.raw`c(x,t)`} />.</li>
        <li><strong>Diffusivity:</strong> <InlineMath tex={String.raw`D=0.1`} />: fixed; sets how fast the mode decays.</li>
        <li><strong>Data:</strong> <InlineMath tex={String.raw`N=24`} /> sparse sensors with Gaussian noise <InlineMath tex={String.raw`\sigma=0.02`} /> (not the full initial condition).</li>
        <li><strong>Network outputs:</strong> two fields: the predictive <em>mean</em> <InlineMath tex={String.raw`\mu(x,t)`} /> and the ensemble <em>std</em> <InlineMath tex={String.raw`s(x,t)`} /> (the UQ).</li>
        <li><strong>Ensemble members:</strong> <InlineMath tex={String.raw`K=5`} /> nets with distinct initialisation and sensor <em>bagging</em>.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        The reference field is the <strong>fundamental diffusion mode</strong>: the exact solution for
        <InlineMath tex={String.raw`c(x,0)=\sin(\pi x)`} /> with walls <InlineMath tex={String.raw`c=0`} />:
      </p>
      <Equation tex={String.raw`c^*(x,t)=e^{-D\pi^2 t}\,\sin(\pi x),\qquad c^*_t = -D\pi^2\,c^*,\quad c^*_{xx} = -\pi^2\,c^* \;\Rightarrow\; c^*_t = D\,c^*_{xx}.`} />
      <p>
        It is our exact <strong>validation anchor</strong>: the primary score is the relative
        <InlineMath tex={String.raw`L^2`} /> of the mean <InlineMath tex={String.raw`\mu`} /> against
        <InlineMath tex={String.raw`c^*`} />. Each member <InlineMath tex={String.raw`f_{\theta_k}`} /> minimises the PDE
        residual at collocation points plus a data term at its sensors, and the walls are imposed <em>hard</em> via an
        output transform that vanishes at <InlineMath tex={String.raw`x=0,1`} />:
      </p>
      <Equation tex={String.raw`f_{\theta_k}(x,t) = x\,(1-x)\,\mathcal{N}_{\theta_k}(x,t)\;\Rightarrow\; f_{\theta_k}\big|_{x=0,1}=0\ \text{(exact, no BC loss).}`} />
      <p>
        The ensemble mean and uncertainty are simply the statistics of the <InlineMath tex={String.raw`K`} /> members,
        evaluated at any query point:
      </p>
      <Equation tex={String.raw`\mu(x,t)=\frac{1}{K}\sum_{k=1}^{K} f_{\theta_k}(x,t),\qquad s(x,t)=\sqrt{\frac{1}{K}\sum_{k=1}^{K}\big(f_{\theta_k}-\mu\big)^2}.`} />

      <h3>The method: deep ensemble as a Bayesian approximation</h3>
      <p>
        Inferring the exact Bayesian <em>posterior</em> over the weights (HMC, SVGD) is expensive. The <strong>deep
        ensemble</strong> (Lakshminarayanan et al., NeurIPS 2017; arXiv:1612.01474,
        doi:10.48550/arXiv.1612.01474) is
        the recognised cheap approximation: training <InlineMath tex={String.raw`K`} /> nets from <em>distinct
        initialisations</em> on resampled data (<em>bagging</em>) makes them <strong>disagree most where no data
        constrains them</strong>: and that disagreement <em>is</em> the epistemic uncertainty. Each member sees a
        bootstrap subset of the sensors with its own noise realisation; the hard walls and shared physics anchor them
        where theory dictates and free them where it does not.
      </p>
      <p>
        The whole ensemble is exported as <strong>one ONNX graph</strong> emitting <InlineMath tex={String.raw`[\mu,\,s]`} />
        per query point, so the browser interactive layer loads a single file and re-evaluates mean and uncertainty live
        (onnxruntime-web). UQ quality is reported as <strong>2σ calibration</strong>: the fraction of the grid where
        <InlineMath tex={String.raw`|\mu - c^*|\le 2s`} /> (a well-calibrated ensemble covers
        <InlineMath tex={String.raw`\sim 95\%`} />).
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> 1D diffusion with fixed diffusivity, hard Dirichlet walls, an analytic reference field,
        and <em>epistemic</em> UQ from ensemble disagreement over sparse, noisy sensors. It is
        <em> illustrative-synthetic</em>: a UQ demonstrator on a manufactured field (the analytic
        <InlineMath tex={String.raw`c^*`} />), <strong>not</strong> a measured dataset or a real calibration case.
        <strong> Out of scope:</strong> formal <em>aleatoric</em> UQ (observation noise modelled as a likelihood), the
        full Bayesian posterior (this is an ensemble proxy, not HMC/VI), unknown diffusivity
        <InlineMath tex={String.raw`D`} /> or an inverse problem, and 2-D/3-D geometry. That is why this case ships as a
        <strong> single honest benchmark</strong>: it is not a closed-form parametric family in a physical knob, but a UQ
        demonstration at one fixed noise and sparsity level: fabricating regimes to fill chips would be dishonest.
      </p>

      <p>
        <strong>What the benchmark shows.</strong> The <strong>mean</strong> field reproduces the fundamental mode: a
        sine half-wave in <InlineMath tex={String.raw`x`} /> that <em>decays exponentially</em> in
        <InlineMath tex={String.raw`t`} /> (factor <InlineMath tex={String.raw`e^{-D\pi^2 t}`} />). The
        <strong> uncertainty</strong> field tells the complementary story: it is <em>near zero at the walls</em> (the
        hard physics leaves no slack) and <em>near the sensors</em>, and it <strong>grows in the data-sparse
        regions</strong>: typically the interior band at intermediate times, away from any reading. That contrast is the
        lesson: <em>where to trust</em> the estimate and <em>where to place the next sensor</em>.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The default <strong>heatmap</strong> shows the mean
        <InlineMath tex={String.raw`\mu(x,t)`} /> (x horizontal, t vertical); hover to read the exact value and watch the
        sine flatten as <InlineMath tex={String.raw`t`} /> rises. Switch to the <strong>uncertainty</strong> field
        <InlineMath tex={String.raw`s(x,t)`} /> to see the walls and sensors as dark valleys and the blind zones as
        bright ridges. The <strong>line-cut profiles</strong> in <InlineMath tex={String.raw`x`} /> and
        <InlineMath tex={String.raw`t`} /> let you compare mean and uncertainty side by side: where the band
        <InlineMath tex={String.raw`\mu\pm 2s`} /> brackets <InlineMath tex={String.raw`c^*`} />, the ensemble is
        calibrated. Since it is a single benchmark, the <strong>Live</strong> tab re-evaluates the trained ensemble (the
        same physics and the same sensors), emitting mean and uncertainty in your browser, with no parameter slider.
      </p>
    </>
  );
}
