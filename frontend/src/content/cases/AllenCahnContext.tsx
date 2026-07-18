import { Equation, InlineMath } from "../../components/Equation";

/** Deep bilingual Context for bench-allencahn (stiff bistable reaction-diffusion, fixed-parameter benchmark). */
export function AllenCahnContext({ lang }: { lang: "en" | "es" }) {
  const es = lang === "es";
  return es ? (
    <>
      <h2>El problema: separación de fases con interfaces afiladas: la ecuación de Allen-Cahn</h2>
      <p>
        <strong>El problema.</strong> La ecuación de Allen-Cahn
        <InlineMath tex={String.raw`u_t = d\,u_{xx} + 5\,(u-u^3)`} /> modela la <em>separación de fases</em>: el campo
        <InlineMath tex={String.raw`u`} /> es atraído hacia dos estados estables <InlineMath tex={String.raw`u=\pm1`} />
        (los pozos del potencial de doble pozo <InlineMath tex={String.raw`W(u)=\tfrac54(1-u^2)^2`} />), y la pequeña
        difusión <InlineMath tex={String.raw`d=0.001`} /> solo suaviza las <strong>interfaces</strong> entre fases. El
        resultado son mesetas planas en <InlineMath tex={String.raw`\pm1`} /> separadas por <em>capas de transición
        afiladas</em> que se mueven lentamente. Es el caso <strong>rígido</strong> (stiff) canónico: la reacción rápida
        contra la difusión lenta hace que un PINN ingenuo <strong>fracase</strong> (colapsa a un estado metastable
        incorrecto). Aquí mostramos la receta que <em>sí</em> funciona.
      </p>

      <h3>Componentes y variables</h3>
      <ul>
        <li><strong>Dominio:</strong> espacio <InlineMath tex={String.raw`x\in[-1,1]`} /> × tiempo <InlineMath tex={String.raw`t\in[0,1]`} />, grilla del campo <InlineMath tex={String.raw`201\times101`} />.</li>
        <li><strong>Incógnita:</strong> el parámetro de orden <InlineMath tex={String.raw`u(x,t)\in[-1,1]`} /> (qué fase, y cuán cerca de saturar).</li>
        <li><strong>Difusión:</strong> <InlineMath tex={String.raw`d=0.001`} />: muy pequeña, así las interfaces son delgadas (<InlineMath tex={String.raw`\sim\sqrt{d}`} />).</li>
        <li><strong>Reacción bistable:</strong> <InlineMath tex={String.raw`5(u-u^3)`} />: empuja hacia <InlineMath tex={String.raw`\pm1`} /> y desestabiliza <InlineMath tex={String.raw`u=0`} />.</li>
        <li><strong>Condición inicial:</strong> <InlineMath tex={String.raw`u(x,0)=x^2\cos(\pi x)`} />, con bordes acoplados (periódicos).</li>
      </ul>

      <h3>Formalización</h3>
      <p>
        La ecuación es el <em>flujo gradiente</em> en <InlineMath tex={String.raw`L^2`} /> de la energía de
        Ginzburg-Landau, que penaliza tanto los gradientes como estar lejos de los pozos:
      </p>
      <Equation tex={String.raw`E[u]=\int \Big(\tfrac{d}{2}\,u_x^2 + W(u)\Big)\,dx,\qquad u_t=-\frac{\delta E}{\delta u}=d\,u_{xx}+5(u-u^3).`} />
      <p>
        El ancho de interfaz de equilibrio sale de balancear los dos términos:
        <InlineMath tex={String.raw`\ell\sim\sqrt{d/5}`} />. Con <InlineMath tex={String.raw`d=0.001`} /> eso es
        <InlineMath tex={String.raw`\ell\approx0.014`} />: una capa finísima frente al dominio de ancho 2. No hay
        solución en forma cerrada para esta IC; la <strong>ancla de validación</strong> es una referencia espectral de
        alta precisión <InlineMath tex={String.raw`u_{\mathrm{ref}}(x,t)`} /> (DeepXDE/Raissi).
      </p>

      <h3>El método: por qué el PINN ingenuo falla, y la receta que funciona</h3>
      <p>
        Un PINN con IC/BC <em>blandas</em> (como términos de pérdida) colapsa: el mínimo trivial
        <InlineMath tex={String.raw`u\equiv0`} /> o un estado metastable tiene residual pequeño y la optimización se
        queda atascada ahí (<strong>95,4%</strong> de L2-relativo frente a la referencia espectral; las vistas Compare y
        Training lo muestran). La receta robusta combina dos ideas:
      </p>
      <Equation tex={String.raw`u_\theta = \underbrace{x^2\cos(\pi x)}_{=\,u(x,0)} + \underbrace{t\,(1-x^2)}_{\text{se anula en }t=0,\ x=\pm1}\,\mathcal{N}_\theta(x,t).`} />
      <ul>
        <li><strong>Restricción dura (hard constraint):</strong> la IC se incorpora en el ansatz, así <InlineMath tex={String.raw`u_\theta(x,0)=u(x,0)`} /> <em>exactamente</em> y los bordes quedan acoplados: sin términos de pérdida de IC/BC que compitan.</li>
        <li><strong>RAR (refinamiento adaptativo por residual):</strong> tras el ajuste base Adam y luego L-BFGS, <strong>4 rondas</strong> de refinamiento voraz: cada ronda muestrea un pool de 100.000 puntos, añade los <strong>600 puntos</strong> de mayor residual como anclas, y reajusta (5.000 iteraciones de Adam); un <strong>L-BFGS final único</strong> pule tras el bucle, concentrando la densidad de colocación justo sobre la capa delgada móvil.</li>
      </ul>
      <p>
        El techo SOTA (PirateNets, <InlineMath tex={String.raw`\sim2\times10^{-5}`} />, jaxpi) se <em>cita</em>, no se
        reclama; esta es la receta DeepXDE en CPU.
      </p>
      <p>
        Arquitectura: una FNN tanh <InlineMath tex={String.raw`[2,64,64,64,1]`} /> con 8.000 puntos de dominio / 400 de
        borde / 800 iniciales, Adam (lr 1e-3, 20.000 iteraciones) y luego L-BFGS antes del bucle RAR. El RAR recortado
        (4 rondas más un L-BFGS final único en vez de 6 L-BFGS por ronda) precalcula mucho más rápido para la misma
        precisión honesta sub-1%.
      </p>

      <h3>Resultado medido (semilla 42)</h3>
      <p>
        Validado contra la referencia espectral <InlineMath tex={String.raw`u_{\mathrm{ref}}(x,t)`} /> (<code>Allen_Cahn.npz</code>,
        DeepXDE/Raissi) en la grilla del campo <InlineMath tex={String.raw`201\times101`} />, la red con restricción dura
        más RAR alcanza <strong>0,41% de L2-relativo</strong> (0.004106): sub-1%, superando la banda esperada (por
        debajo de <InlineMath tex={String.raw`10^{-2}`} />) por unas 2,4x, así la receta resuelve genuinamente las capas
        de transición en vez de colapsar al estado metastable. El PINN suave ingenuo, en cambio, difumina las capas
        afiladas <InlineMath tex={String.raw`\pm1`} /> con <strong>95,4% de L2-relativo</strong>: las vistas Compare y
        Training muestran ambas lado a lado con sus mapas de error. El modelo entrenado se publica <strong>en vivo</strong>
        (ONNX de 49 KB, opset 18; paridad ONNX 1.16e-06 máx. abs).
      </p>

      <h3>Alcances y supuestos</h3>
      <p>
        <strong>Se modela:</strong> Allen-Cahn 1D rígido con difusión y reacción fijas, IC suave única, referencia
        espectral como ancla. <strong>Fuera de alcance:</strong> Allen-Cahn 2-D/3-D (curvatura de interfaz), conservación
        de masa (eso es Cahn-Hilliard, 4º orden), y la familia paramétrica en <InlineMath tex={String.raw`d`} />: el
        frente simétrico de Allen-Cahn es <em>estacionario</em>, así que no admite una familia viajera en forma cerrada
        como Burgers; por eso este caso se publica como un <strong>benchmark de parámetro fijo</strong> (una variante),
        no como un barrido.
      </p>

      <p>
        <strong>Qué muestra el benchmark.</strong> Desde la IC suave, la reacción rápidamente satura las regiones hacia
        <InlineMath tex={String.raw`\pm1`} /> formando <strong>mesetas metastables</strong>, separadas por interfaces
        delgadas que luego migran muy lentamente. El heatmap revela ese contraste: grandes bloques planos de color
        casi-constante y líneas afiladas entre ellos. Es el sello de la dinámica rígida: <em>relajación rápida</em>
        local seguida de <em>evolución lenta</em> de las interfaces.
      </p>
      <p>
        <strong>Cómo leer y usar la viz.</strong> El <strong>heatmap</strong> de <InlineMath tex={String.raw`u(x,t)`} />
        (x horizontal, t vertical) muestra las mesetas <InlineMath tex={String.raw`\pm1`} /> como bloques uniformes y
        las interfaces como líneas nítidas. Al pasar el cursor se lee el valor exacto y se ve lo plano que es el interior
        de cada fase; se muestra el <strong>perfil de corte</strong> en <InlineMath tex={String.raw`x`} /> (mesetas + saltos
        afilados) y en <InlineMath tex={String.raw`t`} /> (cuándo pasa una interfaz por un punto fijo). Como es un
        benchmark de parámetro fijo, el tab <strong>Live</strong> re-evalúa la red entrenada (la misma física), sin
        deslizador de parámetro.
      </p>
    </>
  ): (
    <>
      <h2>The problem: phase separation with sharp interfaces: the Allen-Cahn equation</h2>
      <p>
        <strong>The problem.</strong> The Allen-Cahn equation
        <InlineMath tex={String.raw`u_t = d\,u_{xx} + 5\,(u-u^3)`} /> models <em>phase separation</em>: the field
        <InlineMath tex={String.raw`u`} /> is pulled toward two stable states <InlineMath tex={String.raw`u=\pm1`} />
        (the wells of the double-well potential <InlineMath tex={String.raw`W(u)=\tfrac54(1-u^2)^2`} />), and the small
        diffusion <InlineMath tex={String.raw`d=0.001`} /> only smooths the <strong>interfaces</strong> between phases.
        The result is flat plateaus at <InlineMath tex={String.raw`\pm1`} /> separated by <em>sharp transition
        layers</em> that move slowly. It is the canonical <strong>stiff</strong> case: fast reaction versus slow
        diffusion makes a naive PINN <strong>fail</strong> (it collapses to a wrong metastable state). Here we show the
        recipe that <em>does</em> work.
      </p>

      <h3>Components &amp; variables</h3>
      <ul>
        <li><strong>Domain:</strong> space <InlineMath tex={String.raw`x\in[-1,1]`} /> × time <InlineMath tex={String.raw`t\in[0,1]`} />, a <InlineMath tex={String.raw`201\times101`} /> field grid.</li>
        <li><strong>Unknown:</strong> the order parameter <InlineMath tex={String.raw`u(x,t)\in[-1,1]`} /> (which phase, and how saturated).</li>
        <li><strong>Diffusion:</strong> <InlineMath tex={String.raw`d=0.001`} />: very small, so interfaces are thin (<InlineMath tex={String.raw`\sim\sqrt{d}`} />).</li>
        <li><strong>Bistable reaction:</strong> <InlineMath tex={String.raw`5(u-u^3)`} />: pushes toward <InlineMath tex={String.raw`\pm1`} /> and destabilises <InlineMath tex={String.raw`u=0`} />.</li>
        <li><strong>Initial condition:</strong> <InlineMath tex={String.raw`u(x,0)=x^2\cos(\pi x)`} />, with endpoint-coupled (periodic) ends.</li>
      </ul>

      <h3>Formalization</h3>
      <p>
        The equation is the <em>gradient flow</em> in <InlineMath tex={String.raw`L^2`} /> of the Ginzburg-Landau energy,
        which penalises both gradients and being away from the wells:
      </p>
      <Equation tex={String.raw`E[u]=\int \Big(\tfrac{d}{2}\,u_x^2 + W(u)\Big)\,dx,\qquad u_t=-\frac{\delta E}{\delta u}=d\,u_{xx}+5(u-u^3).`} />
      <p>
        The equilibrium interface width follows from balancing the two terms:
        <InlineMath tex={String.raw`\ell\sim\sqrt{d/5}`} />. With <InlineMath tex={String.raw`d=0.001`} /> that is
        <InlineMath tex={String.raw`\ell\approx0.014`} />: a razor-thin layer against a domain of width 2. There is no
        closed-form solution for this IC; the <strong>validation anchor</strong> is a high-accuracy spectral reference
        <InlineMath tex={String.raw`u_{\mathrm{ref}}(x,t)`} /> (DeepXDE/Raissi).
      </p>

      <h3>The method: why the naive PINN fails, and the recipe that works</h3>
      <p>
        A PINN with <em>soft</em> IC/BC (as loss terms) collapses: the trivial minimum
        <InlineMath tex={String.raw`u\equiv0`} /> or a metastable state has small residual and the optimiser gets stuck
        there (<strong>95.4%</strong> relative-L2 against the spectral reference; the Compare and Training views show it).
        The robust recipe combines two ideas:
      </p>
      <Equation tex={String.raw`u_\theta = \underbrace{x^2\cos(\pi x)}_{=\,u(x,0)} + \underbrace{t\,(1-x^2)}_{\text{vanishes at }t=0,\ x=\pm1}\,\mathcal{N}_\theta(x,t).`} />
      <ul>
        <li><strong>Hard constraint:</strong> the IC is baked into the ansatz, so <InlineMath tex={String.raw`u_\theta(x,0)=u(x,0)`} /> <em>exactly</em> and the ends are coupled: no competing IC/BC loss terms.</li>
        <li><strong>RAR (residual-based adaptive refinement):</strong> after the Adam then L-BFGS base fit, <strong>4 rounds</strong> of greedy refinement: each round draws a 100,000-point pool, adds the <strong>top-600</strong> highest-residual points as anchors, and re-fits (5,000 Adam iters); a <strong>single final L-BFGS</strong> polishes after the loop, concentrating collocation density exactly on the thin moving layer.</li>
      </ul>
      <p>
        The SOTA ceiling (PirateNets, <InlineMath tex={String.raw`\sim2\times10^{-5}`} />, jaxpi) is <em>cited</em>, not
        claimed; this is the DeepXDE recipe on CPU.
      </p>
      <p>
        Architecture: a <InlineMath tex={String.raw`[2,64,64,64,1]`} /> tanh FNN with 8,000 domain / 400 boundary /
        800 initial points, Adam (lr 1e-3, 20,000 iters) then L-BFGS before the RAR loop. The trimmed RAR (4 rounds
        plus one final L-BFGS instead of 6 per-round L-BFGS) bakes far faster for the same honest sub-1% accuracy.
      </p>

      <h3>Measured result (seed 42)</h3>
      <p>
        Validated against the spectral reference <InlineMath tex={String.raw`u_{\mathrm{ref}}(x,t)`} /> (<code>Allen_Cahn.npz</code>,
        DeepXDE/Raissi) on the <InlineMath tex={String.raw`201\times101`} /> field grid, the hard-constraint plus RAR
        network reaches <strong>0.41% relative-L2</strong> (0.004106): sub-1%, clearing the expected band (below
        <InlineMath tex={String.raw`10^{-2}`} />) by about 2.4x, so the recipe genuinely resolves the transition layers
        rather than collapsing to the metastable state. The naive soft PINN, by contrast, smears the sharp
        <InlineMath tex={String.raw`\pm1`} /> layers at <strong>95.4% relative-L2</strong>: the Compare and Training
        views show the two side by side with their error maps. The trained model ships <strong>live</strong> (49 KB
        ONNX, opset 18; ONNX parity 1.16e-06 max abs).
      </p>

      <h3>Scope &amp; assumptions</h3>
      <p>
        <strong>Modeled:</strong> stiff 1D Allen-Cahn with fixed diffusion and reaction, a single smooth IC, a spectral
        reference as anchor. <strong>Out of scope:</strong> 2-D/3-D Allen-Cahn (interface curvature), mass conservation
        (that is Cahn-Hilliard, 4th order), and the parametric-in-<InlineMath tex={String.raw`d`} /> family: the
        symmetric Allen-Cahn front is <em>stationary</em>, so it admits no closed-form traveling family like Burgers;
        that is why this case ships as a <strong>fixed-parameter benchmark</strong> (one variant), not a sweep.
      </p>

      <p>
        <strong>What the benchmark shows.</strong> From the smooth IC, the reaction quickly saturates regions toward
        <InlineMath tex={String.raw`\pm1`} /> forming <strong>metastable plateaus</strong>, separated by thin interfaces
        that then migrate very slowly. The heatmap reveals that contrast: large flat blocks of near-constant colour and
        sharp lines between them. It is the hallmark of stiff dynamics: fast local <em>relaxation</em> followed by slow
        <em> interface evolution</em>.
      </p>
      <p>
        <strong>How to read &amp; use the viz.</strong> The <strong>heatmap</strong> of
        <InlineMath tex={String.raw`u(x,t)`} /> (x horizontal, t vertical) shows the <InlineMath tex={String.raw`\pm1`} />
        plateaus as uniform blocks and the interfaces as crisp lines. Hover to read the exact value and see how flat
        each phase interior is; watch the <strong>line-cut profile</strong> in <InlineMath tex={String.raw`x`} />
        (plateaus + sharp jumps) and in <InlineMath tex={String.raw`t`} /> (when an interface passes a fixed point).
        Since it is a fixed-parameter benchmark, the <strong>Live</strong> tab re-evaluates the trained network (the
        same physics), with no parameter slider.
      </p>
    </>
  );
}
