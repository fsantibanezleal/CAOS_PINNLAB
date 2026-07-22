import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-darcy-pino (PINO: the physics-informed neural operator on the Darcy
 *  family; the workbench variants are LABEL BUDGETS, so the data-efficiency question is the case). */
export function DarcyPinoContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>PINO: el operador que además conoce la ecuación</h2>
      <p>
        El caso hermano, <strong>bench-darcy-operator</strong>, aprende el operador de Darcy solo a partir de
        <em> pares resueltos</em>. Pero generar esos datos significa correr el solver clásico una vez por
        instancia, y en un problema real esa es justamente la parte cara. PINO agrega el
        <strong> residuo de la EDP</strong> a la pérdida del propio operador, así la ecuación puede sustituir
        etiquetas. Este caso mide <strong>cuánto</strong>, en nuestra propia grilla, incluyendo dónde
        <em> no</em> funciona.
      </p>
      <p>
        La pregunta que decide si algo de esto sirve en la práctica:
        <strong> ¿cuántas instancias resueltas necesitas, si el operador también conoce la ecuación?</strong>
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> el cuadrado unitario <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, grilla <InlineMath tex={String.raw`32\times32`} />.</li>
        <li><strong>Entrada:</strong> el campo de permeabilidad <InlineMath tex={String.raw`a(\mathbf{x})`} />, campo aleatorio gaussiano umbralizado a dos valores <InlineMath tex={String.raw`\{3,12\}`} /> (interfaces afiladas).</li>
        <li><strong>Salida:</strong> la presión <InlineMath tex={String.raw`u(\mathbf{x})`} />, con <InlineMath tex={String.raw`u=0`} /> en el borde.</li>
        <li><strong>Variantes (chips):</strong> <em>presupuestos de etiquetas</em>: 0, 8, 32 y 128 instancias resueltas. Ese es el eje del experimento.</li>
        <li><strong>Las dos vías comparadas:</strong> FNO solo-datos y PINO (datos + ecuación), con la <em>misma</em> arquitectura, semilla, épocas y conjunto de prueba. Lo único que cambia es la pérdida.</li>
      </ul>

      <h3>Formalización</h3>
      <p>El problema directo es el mismo Darcy elíptico en forma de divergencia:</p>
      <Equation tex={String.raw`-\nabla\!\cdot\big(a(\mathbf{x})\nabla u\big)=1\ \text{en}\ \Omega=(0,1)^2,\qquad u|_{\partial\Omega}=0.`} />
      <p>
        El FNO solo-datos minimiza únicamente el error contra soluciones ya calculadas. PINO minimiza la suma
        de ese término y del residuo de la ecuación evaluado sobre <em>su propia salida</em>:
      </p>
      <Equation tex={String.raw`\min_\theta\ \underbrace{\mathbb{E}\big\|\mathcal{G}_\theta(a)-u\big\|^2}_{\text{datos (caros)}}\;+\;\lambda\,\underbrace{\mathbb{E}_{a'\sim\mu}\big\|\mathcal{R}\big(a',\mathcal{G}_\theta(a')\big)\big\|^2}_{\text{ecuación (gratis)}}`} />
      <p>
        El segundo término es el mecanismo que importa: un campo nuevo <InlineMath tex={String.raw`a'\sim\mu`} />
        cuesta un filtro gaussiano y un umbral, <strong>sin resolver nada</strong>. Es el Algoritmo 1 del
        paper: "tenemos acceso a un conjunto de datos ilimitado muestreando nuevos
        <InlineMath tex={String.raw`a_j`} /> en cada iteración". Después del preentrenamiento hay una segunda
        fase, la <strong>optimización en tiempo de consulta</strong>: se afina el operador sobre una única
        instancia con el residuo más una <em>pérdida ancla</em>
        <InlineMath tex={String.raw`\mathcal{L}_{op}=\|\mathcal{G}_{\theta_i}(a)-\mathcal{G}_{\theta_0}(a)\|^2`} />
        que lo mantiene cerca del operador preentrenado.
      </p>

      <h3>Dos detalles de implementación que deciden si esto funciona</h3>
      <p>
        <strong>El residuo sobre una salida en grilla.</strong> Un operador emite <InlineMath tex={String.raw`u`} />
        sobre una grilla, así que no hay autograd en <InlineMath tex={String.raw`x`} />. Usamos la forma de
        volúmenes finitos con conductividades de cara por <em>media armónica</em> (la media correcta cuando el
        coeficiente es discontinuo). Se verificó antes de construir nada encima: alimentado con la solución de
        referencia devuelve rms <InlineMath tex={String.raw`3.6\times10^{-14}`} />; alimentado con
        <InlineMath tex={String.raw`u=0`} /> devuelve exactamente <InlineMath tex={String.raw`f=1`} />. La vía
        espectral (FFT) <em>no</em> es la opción por defecto acá: el problema no es periódico y el coeficiente
        es discontinuo, y medido sobre un campo de referencia el laplaciano espectral yerra por ~6 veces el
        término fuente, justo la falla que el paper advierte.
      </p>
      <p>
        <strong>La condición de borde debe ser dura.</strong> El residuo es solo interior, y la solución de
        Darcy es única únicamente <em>junto con</em> su condición de borde. Con una penalización blanda, un
        peso fuerte sobre la física lleva el residuo interior a cero alrededor de valores de borde
        <em> equivocados</em>. Medido: con 32 etiquetas las configuraciones blandas dieron -0.5% y -68.9%, y
        al imponer <InlineMath tex={String.raw`u|_{\partial\Omega}=0`} /> de forma exacta (multiplicando la
        salida por <InlineMath tex={String.raw`16\,x(1-x)\,y(1-y)`} />) el mismo presupuesto pasó a
        <strong> +45.3%</strong>.
      </p>

      <h3>Lo que medimos, incluyendo dónde falla</h3>
      <p>Error relativo L2 sobre 32 campos no vistos:</p>
      <ul>
        <li><strong>0 etiquetas:</strong> PINO 0.689 contra un operador solo-datos que por definición está sin entrenar (1.045). Es un punto cualitativo, no un buen número absoluto.</li>
        <li><strong>8 etiquetas:</strong> PINO 0.471 contra 0.170. <strong>PINO pierde, y feo.</strong> Aquí <InlineMath tex={String.raw`\lambda=1.9`} />, así que no es un peso excesivo sobre la física: con solo 8 etiquetas el término de datos no alcanza a fijar la solución y manda un término físico que aún no converge en 80 épocas.</li>
        <li><strong>32 etiquetas:</strong> PINO 0.078 contra 0.143, <strong>45% mejor</strong>.</li>
        <li><strong>128 etiquetas:</strong> PINO 0.037 contra 0.072, <strong>48% mejor</strong>.</li>
      </ul>
      <p>
        La falla con 8 etiquetas se publica, no se esconde: es exactamente la objeción de siempre a lo
        físico-informado (los tiempos de entrenamiento largos), reproducida y medida dentro de nuestro propio
        operador. La ecuación es la señal correcta, pero <strong>lenta</strong>.
      </p>

      <h3>Lo que este caso NO afirma</h3>
      <ul>
        <li><strong>No es un resultado de precisión de discretización.</strong> El residuo usa el mismo esténcil que el solver de referencia, así que llevarlo a cero <em>es</em> resolver ese mismo sistema discreto. Lo que compra el término físico, y lo único que afirmamos, es eficiencia en datos.</li>
        <li><strong>No son los números del paper.</strong> Los titulares de PINO (20x menos error y 25x más rápido que un PINN en flujo de Kolmogorov; 400x contra un solver pseudo-espectral en GPU) son del paper, en sus problemas y a su escala.</li>
        <li><strong>No es una afirmación de velocidad.</strong> Medido en esta máquina: el solver clásico de referencia a <InlineMath tex={String.raw`32\times32`} /> tarda 42.4 ms y la inferencia ONNX del operador 35-39 ms. A esta grilla el operador <em>no</em> tiene ventaja de velocidad.</li>
        <li><strong>No es más barato de entrenar.</strong> La vía PINO cuesta cerca del doble (766 s contra 377 s con 128 etiquetas). El ahorro está en <em>llamadas al solver evitadas</em>, nunca en tiempo de entrenamiento.</li>
      </ul>
    </>
  ) : (
    <>
      <h2>PINO: the operator that also knows the equation</h2>
      <p>
        Its sibling case, <strong>bench-darcy-operator</strong>, learns the Darcy operator from
        <em> solved pairs</em> alone. But producing that data means running the classical solver once per
        instance, and on a real problem that is exactly the expensive part. PINO adds the
        <strong> PDE residual</strong> to the operator's own loss, so the equation can substitute for labels.
        This case measures <strong>by how much</strong>, on our own grid, including where it does
        <em> not</em> work.
      </p>
      <p>
        The question that decides whether any of this is useful in practice:
        <strong> how many solved instances do you need, if the operator also knows the equation?</strong>
      </p>

      <h3>Components and variables</h3>
      <ul>
        <li><strong>Domain:</strong> the unit square <InlineMath tex={String.raw`(x,y)\in[0,1]^2`} />, a <InlineMath tex={String.raw`32\times32`} /> grid.</li>
        <li><strong>Input:</strong> the permeability field <InlineMath tex={String.raw`a(\mathbf{x})`} />, a Gaussian random field thresholded to two values <InlineMath tex={String.raw`\{3,12\}`} /> (sharp interfaces).</li>
        <li><strong>Output:</strong> the pressure <InlineMath tex={String.raw`u(\mathbf{x})`} />, with <InlineMath tex={String.raw`u=0`} /> on the boundary.</li>
        <li><strong>Variants (chips):</strong> <em>label budgets</em>: 0, 8, 32 and 128 solved instances. That is the axis of the experiment.</li>
        <li><strong>The two lanes compared:</strong> data-only FNO and PINO (data + equation), with the <em>same</em> architecture, seed, epochs and test set. Only the loss changes.</li>
      </ul>

      <h3>Formalisation</h3>
      <p>The forward problem is the same elliptic Darcy equation in divergence form:</p>
      <Equation tex={String.raw`-\nabla\!\cdot\big(a(\mathbf{x})\nabla u\big)=1\ \text{on}\ \Omega=(0,1)^2,\qquad u|_{\partial\Omega}=0.`} />
      <p>
        The data-only FNO minimises only the error against already-computed solutions. PINO minimises that
        term plus the equation's residual evaluated on <em>its own output</em>:
      </p>
      <Equation tex={String.raw`\min_\theta\ \underbrace{\mathbb{E}\big\|\mathcal{G}_\theta(a)-u\big\|^2}_{\text{data (expensive)}}\;+\;\lambda\,\underbrace{\mathbb{E}_{a'\sim\mu}\big\|\mathcal{R}\big(a',\mathcal{G}_\theta(a')\big)\big\|^2}_{\text{equation (free)}}`} />
      <p>
        The second term is the mechanism that matters: a fresh field <InlineMath tex={String.raw`a'\sim\mu`} />
        costs a Gaussian filter and a threshold, with <strong>no solve at all</strong>. This is the paper's
        Algorithm 1: "we have access to the unlimited dataset by sampling new
        <InlineMath tex={String.raw`a_j`} /> in each iteration". After pre-training there is a second phase,
        <strong> test-time optimization</strong>: the operator is fine-tuned on a single instance using the
        residual plus an <em>anchor loss</em>
        <InlineMath tex={String.raw`\mathcal{L}_{op}=\|\mathcal{G}_{\theta_i}(a)-\mathcal{G}_{\theta_0}(a)\|^2`} />
        that keeps it near the pre-trained operator.
      </p>

      <h3>Two implementation facts that decide whether this works</h3>
      <p>
        <strong>The residual on a grid output.</strong> An operator emits <InlineMath tex={String.raw`u`} /> on
        a grid, so there is no autograd in <InlineMath tex={String.raw`x`} />. We use the finite-volume
        divergence form with <em>harmonic-mean</em> face conductivities (the correct face average for a
        discontinuous coefficient). It was verified before anything was built on it: fed the reference
        solution it returns rms <InlineMath tex={String.raw`3.6\times10^{-14}`} />; fed
        <InlineMath tex={String.raw`u=0`} /> it returns exactly <InlineMath tex={String.raw`f=1`} />. The
        spectral (FFT) route is <em>not</em> the default here: this problem is non-periodic with a
        discontinuous coefficient, and measured on a reference field the spectral Laplacian misses by ~6x the
        source term, exactly the failure the paper warns about.
      </p>
      <p>
        <strong>The boundary condition must be hard.</strong> The residual is interior-only, and the Darcy
        solution is unique only <em>together with</em> its boundary condition. With a soft penalty, a strong
        physics weight drives the interior residual to zero around the <em>wrong</em> boundary values.
        Measured: at 32 labels the soft configurations scored -0.5% and -68.9%, and enforcing
        <InlineMath tex={String.raw`u|_{\partial\Omega}=0`} /> exactly (multiplying the output by
        <InlineMath tex={String.raw`16\,x(1-x)\,y(1-y)`} />) moved the same budget to <strong>+45.3%</strong>.
      </p>

      <h3>What we measured, including where it fails</h3>
      <p>Relative L2 error over 32 unseen fields:</p>
      <ul>
        <li><strong>0 labels:</strong> PINO 0.689 against a data-only operator that is untrained by definition (1.045). A qualitative point, not a good absolute number.</li>
        <li><strong>8 labels:</strong> PINO 0.471 against 0.170. <strong>PINO loses, badly.</strong> Here <InlineMath tex={String.raw`\lambda=1.9`} />, so this is not an over-weighted physics term: with only 8 labels the data term cannot pin the solution, so the outcome is dominated by a physics term that has not converged in 80 epochs.</li>
        <li><strong>32 labels:</strong> PINO 0.078 against 0.143, <strong>45% better</strong>.</li>
        <li><strong>128 labels:</strong> PINO 0.037 against 0.072, <strong>48% better</strong>.</li>
      </ul>
      <p>
        The 8-label failure is published, not hidden: it is exactly the standing objection to
        physics-informed training (long training times), reproduced and measured inside our own operator. The
        equation is the right signal, but a <strong>slow</strong> one.
      </p>

      <h3>What this case does NOT claim</h3>
      <ul>
        <li><strong>Not a discretisation-accuracy result.</strong> The residual uses the same stencil as the reference solver, so driving it to zero <em>is</em> solving that same discrete system. What the physics term buys, and all we claim, is data efficiency.</li>
        <li><strong>Not the paper's numbers.</strong> PINO's headline results (20x lower error and 25x speedup vs a PINN on Kolmogorov flow; 400x vs a GPU pseudo-spectral solver) are the paper's, on the paper's problems, at the paper's scale.</li>
        <li><strong>Not a speed claim.</strong> Measured on this machine: the classical reference solve at <InlineMath tex={String.raw`32\times32`} /> takes 42.4 ms and the operator's ONNX inference 35-39 ms. At this grid the operator has <em>no</em> speed advantage.</li>
        <li><strong>Not cheaper to train.</strong> The PINO lane costs about twice as much (766 s against 377 s at 128 labels). The saving is in <em>solver calls avoided</em>, never in training time.</li>
      </ul>
    </>
  );
}
