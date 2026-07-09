import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for poll-tailings-seepage (1D unsaturated Richards/Gardner seepage, parametric in the
 *  sorptive number α via an exact Kirchhoff-transform family). */
export function TailingsSeepageContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: agua que percola por un depósito de relaves no saturado: la ecuación de Richards</h2>
      <p>
        <strong>El problema.</strong> En un tranque de relaves, el agua se infiltra hacia abajo por un material
        <em> parcialmente saturado</em>: los poros contienen aire y agua a la vez. La presión del agua es{" "}
        <em>negativa</em> (succión), <InlineMath tex={String.raw`\psi<0`} />, y tanto la capacidad de almacenamiento
        como la conductividad del suelo <strong>dependen fuertemente de esa succión</strong>. La ecuación de{" "}
        <strong>Richards</strong> <InlineMath tex={String.raw`C(\psi)\,\psi_t=\partial_z[K(\psi)(\psi_z+1)]`} /> gobierna
        el perfil vertical de carga de presión <InlineMath tex={String.raw`\psi(z,t)`} />. Es fuertemente{" "}
        <em>no lineal</em> porque <InlineMath tex={String.raw`K`} /> y <InlineMath tex={String.raw`C`} /> son funciones
        de la propia incógnita. Aquí el parámetro de la curva de retención: el <strong>número sortivo</strong> de
        Gardner <InlineMath tex={String.raw`\alpha`} />: es ajustable: una sola red aprende toda la familia{" "}
        <InlineMath tex={String.raw`\psi(z,t;\alpha)`} />, y en el tab <strong>Live</strong> mover el deslizador de{" "}
        <InlineMath tex={String.raw`\alpha`} /> muestra perfiles de succión más profundos (poros amplios) o más someros
        (poros finos).
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> profundidad <InlineMath tex={String.raw`z\in[0,1]`} /> (z hacia arriba) × tiempo <InlineMath tex={String.raw`t\in[0,1]`} /> (escalados), grilla del campo <InlineMath tex={String.raw`101\times51`} />.</li>
        <li><strong>Incógnita:</strong> la carga de presión <InlineMath tex={String.raw`\psi(z,t)<0`} /> (succión); cuanto más negativa, más seco el material.</li>
        <li><strong>Parámetro de control:</strong> el <em>número sortivo</em> de Gardner <InlineMath tex={String.raw`\alpha\in[1.0,2.5]`} />: un input de la red. Fija cuán rápido cae la conductividad con la succión.</li>
        <li><strong>Conductividad:</strong> <InlineMath tex={String.raw`K(\psi)=K_s\,e^{\alpha\psi}`} /> (modelo exponencial de Gardner), con <InlineMath tex={String.raw`K_s=0.25`} /> saturada.</li>
        <li><strong>Contenido de agua y capacidad:</strong> <InlineMath tex={String.raw`\theta(\psi)=\theta_r+(\theta_s-\theta_r)e^{\alpha\psi}`} /> y <InlineMath tex={String.raw`C(\psi)=\theta'(\psi)=(\theta_s-\theta_r)\,\alpha\,e^{\alpha\psi}`} />, con <InlineMath tex={String.raw`\theta_s=0.43,\ \theta_r=0.078`} />.</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        Expandiendo la divergencia con <InlineMath tex={String.raw`K'(\psi)=\alpha K`} />, la ecuación de Richards con
        cierre de Gardner queda
      </p>
      <Equation tex={String.raw`C(\psi)\,\psi_t \;=\; K(\psi)\,\psi_{zz} \;+\; K'(\psi)\,(\psi_z^2+\psi_z),\qquad K=K_s e^{\alpha\psi},\ C=(\theta_s-\theta_r)\alpha\,e^{\alpha\psi}.`} />
      <p>
        El truco que la hace tratable es la <strong>transformada de Kirchhoff</strong>{" "}
        <InlineMath tex={String.raw`m:=e^{\alpha\psi}`} /> (de modo que <InlineMath tex={String.raw`\psi=\tfrac1\alpha\ln m`} />
        y <InlineMath tex={String.raw`0<m<1\Leftrightarrow\psi<0`} />, siempre no saturado). Sustituyendo, el operador
        no lineal <strong>se linealiza exactamente</strong> a coeficientes constantes en <InlineMath tex={String.raw`m`} />:
      </p>
      <Equation tex={String.raw`C\,\psi_t-K\,\psi_{zz}-K'(\psi_z^2+\psi_z)\;=\;(\theta_s-\theta_r)\,m_t-\frac{K_s}{\alpha}\,m_{zz}-K_s\,m_z.`} />
      <p>
        Una solución <em>exacta</em> (sin fuente manufacturada) de esa ecuación lineal en{" "}
        <InlineMath tex={String.raw`m`} /> es el modo separable, que devuelto a <InlineMath tex={String.raw`\psi`} /> da
        nuestra <strong>ancla de validación</strong> en forma cerrada para <em>cualquier</em>{" "}
        <InlineMath tex={String.raw`\alpha`} />:
      </p>
      <Equation tex={String.raw`\psi^*(z,t;\alpha)=\frac1\alpha\ln\!\Big(M_0+A\,e^{-\lambda(\alpha)\,t}\,e^{-\kappa z}\Big),\qquad \lambda(\alpha)=\frac{K_s}{\theta_s-\theta_r}\,\frac{\kappa(\alpha-\kappa)}{\alpha}.`} />
      <p>
        La <strong>relación de dispersión</strong> <InlineMath tex={String.raw`\lambda(\alpha)`} /> es exactamente la
        condición que anula el residual lineal en <InlineMath tex={String.raw`m`} /> (por eso es solución{" "}
        <em>exacta</em>, no por una fuente añadida). Con <InlineMath tex={String.raw`\kappa=0.9<\alpha`} /> se tiene{" "}
        <InlineMath tex={String.raw`\lambda>0`} />: un transitorio de <em>secado</em> (la succión se profundiza en el
        tiempo). La PINN <InlineMath tex={String.raw`\psi_\theta(z,t,\alpha)`} /> minimiza el residual de Richards en
        puntos de colocación, con la IC y las condiciones de borde en <InlineMath tex={String.raw`z=0,1`} /> impuestas de
        forma <em>blanda</em> (iguales a <InlineMath tex={String.raw`\psi^*`} />), de modo que el L2 reportado es el
        error real del PINN frente a la solución exacta.
      </p>

      <h3>El método: cierre de Gardner y por qué la familia es honesta</h3>
      <p>
        Gardner (1958) propuso <InlineMath tex={String.raw`K(\psi)=K_s e^{\alpha\psi}`} /> precisamente porque mantiene
        el problema <em>analíticamente tratable</em>: es el cierre clásico para soluciones cerradas de la zona no
        saturada. El número sortivo <InlineMath tex={String.raw`\alpha`} /> codifica la distribución de tamaño de poro:{" "}
        <InlineMath tex={String.raw`\alpha`} /> chico (poros amplios, arenoso) drena fácil y deja perfiles de succión{" "}
        <em>profundos y estratificados</em>; <InlineMath tex={String.raw`\alpha`} /> grande (poros finos, arcilloso)
        retiene agua y da succión <em>somera</em>. Como <InlineMath tex={String.raw`\psi^*`} /> depende{" "}
        <em>genuinamente</em> de <InlineMath tex={String.raw`\alpha`} /> (la profundidad de la succión casi se duplica a
        lo largo del barrido), el deslizador mueve física real, no un parámetro cosmético.
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> flujo 1D vertical no saturado, cierre exponencial de Gardner, succión estrictamente
        negativa (<InlineMath tex={String.raw`\psi<0`} /> en todo el dominio y todo <InlineMath tex={String.raw`\alpha`} />),
        familia exacta sin fuente. Es <em>ilustrativo-sintético</em>: físicamente fiel pero NO ajustado a mediciones: {" "}
        <strong>no existe un dataset abierto de <InlineMath tex={String.raw`\psi(z,t)`} /> en la zona no saturada de un
        tranque</strong>; la lane no saturada está modelada, no medida. <strong>Fuera de alcance:</strong> el cierre más
        completo de van Genuchten-Mualem (documentado como extensión), histéresis de la curva de retención, flujo 2D/3D
        y heterogeneidad por capas, y la entrada en saturación (<InlineMath tex={String.raw`\psi\to0`} />, donde Gardner
        deja de ser válido). El inverso de carga de Darcy <em>saturada</em> sí podría usar datos reales del USGS: se
        documenta como la extensión de Tier-C, separada de esta lane.
      </p>

      <p>
        <strong>Qué muestra cada variante.</strong> El barrido del número sortivo recorre la familia de suelos:{" "}
        <em>α=1.0</em> (poros amplios): la succión más profunda y estratificada; <em>α=1.3/1.6/1.9</em> la van haciendo
        progresivamente más somera y secando más rápido; <em>α=2.2/2.5</em> (poros finos) dan un perfil somero con el
        tope casi saturado. En todas, el perfil <strong>se seca en el tiempo</strong> (<InlineMath tex={String.raw`\lambda>0`} />)
        y la succión crece con la profundidad medida desde el tope.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`\psi(z,t)`} />{" "}
        (z vertical, t horizontal) muestra una banda de succión que se <em>profundiza</em> hacia la derecha (secado): el
        color codifica cuán negativa es la carga. Pasa el cursor para leer la succión exacta en cualquier punto, y mira
        los <strong>perfiles de corte</strong> en <InlineMath tex={String.raw`z`} /> (el perfil vertical de succión, su
        pendiente = el gradiente hidráulico) y en <InlineMath tex={String.raw`t`} /> (cómo se seca un punto fijo). Los{" "}
        <strong>chips</strong> cargan cada número sortivo; en <strong>Live</strong>, desliza{" "}
        <InlineMath tex={String.raw`\alpha`} /> y ve el perfil de succión profundizarse o aplanarse en vivo en tu
        navegador (onnxruntime-web).
      </p>
    </>
  ): (
    <>
      <h2>The problem: water seeping through an unsaturated tailings deposit: the Richards equation</h2>
      <p>
        <strong>The problem.</strong> In a tailings dam, water infiltrates downward through a{" "}
        <em>partially saturated</em> material: the pores hold both air and water. The water pressure is{" "}
        <em>negative</em> (suction), <InlineMath tex={String.raw`\psi<0`} />, and both the storage capacity and the
        soil conductivity <strong>depend strongly on that suction</strong>. The <strong>Richards</strong> equation{" "}
        <InlineMath tex={String.raw`C(\psi)\,\psi_t=\partial_z[K(\psi)(\psi_z+1)]`} /> governs the vertical
        pressure-head profile <InlineMath tex={String.raw`\psi(z,t)`} />. It is strongly <em>nonlinear</em> because{" "}
        <InlineMath tex={String.raw`K`} /> and <InlineMath tex={String.raw`C`} /> are functions of the unknown itself.
        Here the retention-curve parameter: Gardner's <strong>sorptive number</strong>{" "}
        <InlineMath tex={String.raw`\alpha`} />: is tunable: a single network learns the whole family{" "}
        <InlineMath tex={String.raw`\psi(z,t;\alpha)`} />, and in the <strong>Live</strong> tab moving the{" "}
        <InlineMath tex={String.raw`\alpha`} /> slider shows deeper suction profiles (coarse pores) or shallower ones
        (fine pores).
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> depth <InlineMath tex={String.raw`z\in[0,1]`} /> (z up) × time <InlineMath tex={String.raw`t\in[0,1]`} /> (scaled), a <InlineMath tex={String.raw`101\times51`} /> field grid.</li>
        <li><strong>Unknown:</strong> the pressure head <InlineMath tex={String.raw`\psi(z,t)<0`} /> (suction); the more negative, the drier the material.</li>
        <li><strong>Control parameter:</strong> Gardner's <em>sorptive number</em> <InlineMath tex={String.raw`\alpha\in[1.0,2.5]`} />: a network input. It sets how fast conductivity drops with suction.</li>
        <li><strong>Conductivity:</strong> <InlineMath tex={String.raw`K(\psi)=K_s\,e^{\alpha\psi}`} /> (Gardner exponential model), with saturated <InlineMath tex={String.raw`K_s=0.25`} />.</li>
        <li><strong>Water content &amp; capacity:</strong> <InlineMath tex={String.raw`\theta(\psi)=\theta_r+(\theta_s-\theta_r)e^{\alpha\psi}`} /> and <InlineMath tex={String.raw`C(\psi)=\theta'(\psi)=(\theta_s-\theta_r)\,\alpha\,e^{\alpha\psi}`} />, with <InlineMath tex={String.raw`\theta_s=0.43,\ \theta_r=0.078`} />.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        Expanding the divergence with <InlineMath tex={String.raw`K'(\psi)=\alpha K`} />, the Richards equation with the
        Gardner closure reads
      </p>
      <Equation tex={String.raw`C(\psi)\,\psi_t \;=\; K(\psi)\,\psi_{zz} \;+\; K'(\psi)\,(\psi_z^2+\psi_z),\qquad K=K_s e^{\alpha\psi},\ C=(\theta_s-\theta_r)\alpha\,e^{\alpha\psi}.`} />
      <p>
        The trick that makes it tractable is the <strong>Kirchhoff transform</strong>{" "}
        <InlineMath tex={String.raw`m:=e^{\alpha\psi}`} /> (so <InlineMath tex={String.raw`\psi=\tfrac1\alpha\ln m`} />
        and <InlineMath tex={String.raw`0<m<1\Leftrightarrow\psi<0`} />, always unsaturated). Substituting, the
        nonlinear operator <strong>linearises exactly</strong> to constant coefficients in{" "}
        <InlineMath tex={String.raw`m`} />:
      </p>
      <Equation tex={String.raw`C\,\psi_t-K\,\psi_{zz}-K'(\psi_z^2+\psi_z)\;=\;(\theta_s-\theta_r)\,m_t-\frac{K_s}{\alpha}\,m_{zz}-K_s\,m_z.`} />
      <p>
        An <em>exact</em> solution (no manufactured source) of that linear equation in{" "}
        <InlineMath tex={String.raw`m`} /> is the separable mode, which mapped back to{" "}
        <InlineMath tex={String.raw`\psi`} /> gives our <strong>validation anchor</strong> in closed form for{" "}
        <em>any</em> <InlineMath tex={String.raw`\alpha`} />:
      </p>
      <Equation tex={String.raw`\psi^*(z,t;\alpha)=\frac1\alpha\ln\!\Big(M_0+A\,e^{-\lambda(\alpha)\,t}\,e^{-\kappa z}\Big),\qquad \lambda(\alpha)=\frac{K_s}{\theta_s-\theta_r}\,\frac{\kappa(\alpha-\kappa)}{\alpha}.`} />
      <p>
        The <strong>dispersion relation</strong> <InlineMath tex={String.raw`\lambda(\alpha)`} /> is exactly the
        condition that zeroes the linear residual in <InlineMath tex={String.raw`m`} /> (which is why it is an{" "}
        <em>exact</em> solution, not one propped up by an added source). With{" "}
        <InlineMath tex={String.raw`\kappa=0.9<\alpha`} /> we get <InlineMath tex={String.raw`\lambda>0`} />: a{" "}
        <em>drying</em> transient (the suction deepens in time). The PINN{" "}
        <InlineMath tex={String.raw`\psi_\theta(z,t,\alpha)`} /> minimises the Richards residual at collocation points,
        with the IC and the boundary conditions at <InlineMath tex={String.raw`z=0,1`} /> imposed <em>softly</em> (equal
        to <InlineMath tex={String.raw`\psi^*`} />), so the reported L2 is the true PINN error against the exact
        solution.
      </p>

      <h3>The method: the Gardner closure and why the family is honest</h3>
      <p>
        Gardner (1958) proposed <InlineMath tex={String.raw`K(\psi)=K_s e^{\alpha\psi}`} /> precisely because it keeps
        the problem <em>analytically tractable</em>: it is the classic closure for closed-form unsaturated-zone
        solutions. The sorptive number <InlineMath tex={String.raw`\alpha`} /> encodes the pore-size distribution:
        small <InlineMath tex={String.raw`\alpha`} /> (coarse, sandy pores) drains easily and leaves{" "}
        <em>deep, stratified</em> suction profiles; large <InlineMath tex={String.raw`\alpha`} /> (fine, clayey pores)
        retains water and gives <em>shallow</em> suction. Because <InlineMath tex={String.raw`\psi^*`} /> depends{" "}
        <em>genuinely</em> on <InlineMath tex={String.raw`\alpha`} /> (the suction depth nearly doubles across the
        sweep), the slider moves real physics, not a cosmetic parameter.
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> 1D vertical unsaturated flow, the Gardner exponential closure, strictly negative
        suction (<InlineMath tex={String.raw`\psi<0`} /> everywhere and for every <InlineMath tex={String.raw`\alpha`} />),
        a source-free exact family. It is <em>illustrative-synthetic</em>: physically faithful but NOT fit to
        measurements: <strong>no open <InlineMath tex={String.raw`\psi(z,t)`} /> dataset exists for the unsaturated
        zone of a tailings deposit</strong>; the unsaturated lane is modeled, not measured. <strong>Out of scope:</strong>{" "}
        the fuller van Genuchten-Mualem closure (documented as an extension), retention-curve hysteresis, 2D/3D flow and
        layered heterogeneity, and the onset of saturation (<InlineMath tex={String.raw`\psi\to0`} />, where Gardner
        ceases to hold). The <em>saturated</em> Darcy-head inverse could use real USGS data: documented as the Tier-C
        extension, separate from this lane.
      </p>

      <p>
        <strong>What each variant shows.</strong> The sorptive-number sweep walks the soil family: <em>α=1.0</em> (coarse
        pores): the deepest, most stratified suction; <em>α=1.3/1.6/1.9</em> make it progressively shallower and dry
        faster; <em>α=2.2/2.5</em> (fine pores) give a shallow profile with a near-saturated top. In all of them the
        profile <strong>dries in time</strong> (<InlineMath tex={String.raw`\lambda>0`} />) and the suction grows with
        depth measured down from the top.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of{" "}
        <InlineMath tex={String.raw`\psi(z,t)`} /> (z vertical, t horizontal) shows a suction band that{" "}
        <em>deepens</em> to the right (drying): the colour encodes how negative the head is. Hover to read the exact
        suction anywhere, and watch the <strong>line-cut profiles</strong> in <InlineMath tex={String.raw`z`} /> (the
        vertical suction profile, its slope = the hydraulic gradient) and in <InlineMath tex={String.raw`t`} /> (how a
        fixed point dries out). The <strong>chips</strong> load each sorptive number; in <strong>Live</strong>, slide{" "}
        <InlineMath tex={String.raw`\alpha`} /> and watch the suction profile deepen or flatten live in your browser
        (onnxruntime-web).
      </p>
    </>
  );
}
