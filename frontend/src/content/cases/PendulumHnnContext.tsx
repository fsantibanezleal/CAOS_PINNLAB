import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for dyn-pendulum-hnn (structure-preserving learning: a Hamiltonian network vs an
 *  unstructured one; the variants ARE the two models). */
export function PendulumHnnContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>Aprendizaje que preserva estructura: aprender la energía, no el campo</h2>
      <p>
        El caso hermano, <strong>dyn-double-pendulum</strong>, pregunta cuánto puede una red seguir una
        trayectoria caótica. Este pregunta algo distinto y más filoso: dos modelos que
        <strong> ajustan la dinámica igual de bien</strong>, ¿alguno respeta la energía que el sistema
        conserva? Es la familia que al catálogo le faltaba por completo: antes de este caso, el único sistema
        mecánico era un PINN de residuo, así que nada conservaba energía por construcción.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Sistema:</strong> el mismo péndulo doble, pero soltado desde el reposo a un ángulo pequeño (40°), así el movimiento es <em>acotado y casi-periódico</em>, no caótico.</li>
        <li><strong>Estado:</strong> las coordenadas canónicas <InlineMath tex={String.raw`(q_1,q_2,p_1,p_2)`} />. Los momentos <InlineMath tex={String.raw`p_i`} /> <strong>no</strong> son las velocidades angulares: <InlineMath tex={String.raw`p_1=2\dot\theta_1+\dot\theta_2\cos(\theta_1-\theta_2)`} />.</li>
        <li><strong>Variantes (chips):</strong> los dos <em>modelos</em>. Una red hamiltoniana (estructurada) y una red sin estructura (referencia), idénticas en tamaño, datos, semilla, pasos e integrador.</li>
        <li><strong>La métrica:</strong> la deriva relativa de la energía total <InlineMath tex={String.raw`|E(t)-E(0)|/|E(0)|`} /> a lo largo de la trayectoria integrada.</li>
      </ul>

      <h3>Formalización</h3>
      <p>Una red sin estructura mapea el estado a su derivada temporal directamente, así que puede producir cualquier campo vectorial, incluidos los que ningún sistema hamiltoniano generaría. Una red hamiltoniana en cambio entrega un solo escalar <InlineMath tex={String.raw`H(q,p)`} /> y toma el <strong>gradiente simpléctico</strong>:</p>
      <Equation tex={String.raw`\dot q=\frac{\partial H}{\partial p},\quad \dot p=-\frac{\partial H}{\partial q}\qquad\Longleftrightarrow\qquad \dot z = J\,\nabla H,\ \ J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}.`} />
      <p>Cualquier campo de esa forma conserva <InlineMath tex={String.raw`H`} /> exactamente a lo largo de su propio flujo, porque <InlineMath tex={String.raw`\tfrac{dH}{dt}=\nabla H\cdot J\,\nabla H=0`} /> (<InlineMath tex={String.raw`J`} /> es antisimétrica). Por eso la conservación de energía <strong>no</strong> es algo que el optimizador deba acertar: es una propiedad de la parametrización.</p>

      <h3>Lo que medimos</h3>
      <p>Ambos modelos se integran con el <strong>mismo</strong> RK4 desde la misma condición inicial, así que el integrador no puede explicar ninguna diferencia. Sobre 8 segundos:</p>
      <ul>
        <li><strong>Red hamiltoniana:</strong> pérdida de derivada <InlineMath tex={String.raw`1.2\times10^{-5}`} />, deriva de energía <strong>0.07%</strong>.</li>
        <li><strong>Red sin estructura:</strong> pérdida de derivada <InlineMath tex={String.raw`1.0\times10^{-4}`} />, deriva de energía <strong>7.44%</strong>: unas <strong>100 veces</strong> más.</li>
      </ul>
      <p>Los dos ajustan la derivada al mismo orden, y aun así solo el estructurado mantiene la energía. La diferencia es el invariante, no el ajuste.</p>

      <h3>Límites honestos</h3>
      <ul>
        <li><strong>Este es el régimen de BAJA energía.</strong> En la condición caótica de alta energía la trayectoria abandona los datos de entrenamiento en menos de un segundo, y allí hasta la <InlineMath tex={String.raw`H`} /> aprendida queda sin restringir y deriva (el modelo estructurado igual gana por ~27x, pero ambos son malos en términos absolutos). Ese régimen caótico es trabajo de dyn-double-pendulum.</li>
        <li><strong>0.07% no es cero.</strong> La red conserva su PROPIA <InlineMath tex={String.raw`H`} /> aprendida, no la verdadera, e integrada con RK4, que no es simpléctico. Un integrador leapfrog lo ajustaría más.</li>
        <li><strong>Coordenadas canónicas obligatorias.</strong> Alimentar a la red <InlineMath tex={String.raw`(\theta,\dot\theta)`} /> y llamarla hamiltoniana es un error común y silencioso.</li>
      </ul>
    </>
  ) : (
    <>
      <h2>Structure-preserving learning: learn the energy, not the field</h2>
      <p>
        The sibling case, <strong>dyn-double-pendulum</strong>, asks how long a network can track a chaotic
        trajectory. This one asks something different and sharper: two models that
        <strong> fit the dynamics equally well</strong>, does either respect the energy the system conserves?
        It is the family the catalogue was missing entirely: before this case the only mechanical system was a
        residual PINN, so nothing conserved energy by construction.
      </p>

      <h3>Components and variables</h3>
      <ul>
        <li><strong>System:</strong> the same double pendulum, but released from rest at a small angle (40°) so the motion is <em>bounded and quasi-periodic</em>, not chaotic.</li>
        <li><strong>State:</strong> the canonical coordinates <InlineMath tex={String.raw`(q_1,q_2,p_1,p_2)`} />. The momenta <InlineMath tex={String.raw`p_i`} /> are <strong>not</strong> the angular velocities: <InlineMath tex={String.raw`p_1=2\dot\theta_1+\dot\theta_2\cos(\theta_1-\theta_2)`} />.</li>
        <li><strong>Variants (chips):</strong> the two <em>models</em>. A Hamiltonian network (structured) and an unstructured network (baseline), identical in size, data, seed, steps and integrator.</li>
        <li><strong>The metric:</strong> the relative drift of the total energy <InlineMath tex={String.raw`|E(t)-E(0)|/|E(0)|`} /> along the rolled-out trajectory.</li>
      </ul>

      <h3>Formalisation</h3>
      <p>An unstructured network maps the state to its time derivative directly, so it can produce any vector field, including ones no Hamiltonian system would generate. A Hamiltonian network instead outputs a single scalar <InlineMath tex={String.raw`H(q,p)`} /> and takes the <strong>symplectic gradient</strong>:</p>
      <Equation tex={String.raw`\dot q=\frac{\partial H}{\partial p},\quad \dot p=-\frac{\partial H}{\partial q}\qquad\Longleftrightarrow\qquad \dot z = J\,\nabla H,\ \ J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}.`} />
      <p>Any field of that form conserves <InlineMath tex={String.raw`H`} /> exactly along its own flow, because <InlineMath tex={String.raw`\tfrac{dH}{dt}=\nabla H\cdot J\,\nabla H=0`} /> (<InlineMath tex={String.raw`J`} /> is antisymmetric). So energy conservation is <strong>not</strong> something the optimiser has to get right: it is a property of the parameterisation.</p>

      <h3>What we measured</h3>
      <p>Both models are rolled out with the <strong>same</strong> RK4 from the same initial condition, so the integrator cannot explain any difference. Over 8 seconds:</p>
      <ul>
        <li><strong>Hamiltonian network:</strong> derivative loss <InlineMath tex={String.raw`1.2\times10^{-5}`} />, energy drift <strong>0.07%</strong>.</li>
        <li><strong>Unstructured network:</strong> derivative loss <InlineMath tex={String.raw`1.0\times10^{-4}`} />, energy drift <strong>7.44%</strong>: about <strong>100x</strong> more.</li>
      </ul>
      <p>The two fit the derivative to the same order, and still only the structured one holds the energy. The difference is the invariant, not the fit.</p>

      <h3>Honest limits</h3>
      <ul>
        <li><strong>This is the LOW-energy regime.</strong> At the high-energy chaotic initial condition the trajectory leaves the training data within a second, and there even the learned <InlineMath tex={String.raw`H`} /> becomes unconstrained and drifts (the structured model still wins by ~27x, but both are bad in absolute terms). That chaotic regime is dyn-double-pendulum's job.</li>
        <li><strong>0.07% is not zero.</strong> The network conserves its OWN learned <InlineMath tex={String.raw`H`} />, not the true one, integrated with RK4, which is not symplectic. A leapfrog integrator would tighten it.</li>
        <li><strong>Canonical coordinates are mandatory.</strong> Feeding the network <InlineMath tex={String.raw`(\theta,\dot\theta)`} /> and calling it Hamiltonian is a common, silent error.</li>
      </ul>
    </>
  );
}
