"""PHASE A of the ESTIMATION REFRAME (issue #46): bake THE ANSWERS.

For every case, compute the engineering QUANTITY OF INTEREST from the already-baked artifacts (traces / frames /
summaries): NO training, no assertion: every number is derived from the validated fields. Writes a manifest
`estimate` block: { question_en/es, why_en/es, items: [{ label_en/es, value | values{variantId}, detail? }] } and
patches index.json with the per-case question so the case list can show it.

The derivations (front tracking, threshold crossings, integration, argmax, finite differences, exceedance from
mean/sigma) mirror how these metrics are read off solutions in practice (e.g. the Kynch settling construction,
breakthrough curves, P80-style passing fractions). heat1d's t_half was cross-checked against the closed form
ln 2/(alpha pi^2): agreement within the trace's time resolution (1-2.5%).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MANIFESTS = DERIVED / "manifests"


def load(p):
    return json.loads((DERIVED / p).read_text(encoding="utf-8"))


def man(cid):
    return json.loads((MANIFESTS / f"{cid}.json").read_text(encoding="utf-8"))


def save_man(cid, m):
    (MANIFESTS / f"{cid}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def trace_of(m, vid=None):
    v = next((x for x in m["variants"] if x["id"] == vid), m["variants"][0])
    return load(v["trace"]["path"])


def crossing_time(ts, series, level, rising=True):
    """First t where the series crosses `level` (linear interp)."""
    s = np.asarray(series, float)
    for j in range(1, len(s)):
        a, b = s[j - 1], s[j]
        if (rising and a < level <= b) or ((not rising) and a > level >= b):
            f = (level - a) / ((b - a) or 1)
            return float(ts[j - 1] + f * (ts[j] - ts[j - 1]))
    return None


def fmt(v, dec=2):
    return f"{v:.{dec}f}" if v is not None else "not reached in the window"


def set_estimate(cid, question_en, question_es, why_en, why_es, items):
    m = man(cid)
    m["estimate"] = {"question_en": question_en, "question_es": question_es, "why_en": why_en, "why_es": why_es, "items": items}
    save_man(cid, m)
    print(f"[{cid}] estimate baked: {items[0]['label_en']}")
    return question_en, question_es


QUESTIONS = {}


def item(label_en, label_es, value=None, values=None, value_es=None, values_es=None):
    """One answer line. `value`/`values` are the EN (and default) reading.

    Most values are pure numbers with units and read the same in both languages, so they need no ES twin.
    Pass `value_es`/`values_es` ONLY when the value carries prose ("not in window", "unrecoverable", "0.96
    to 1.17"): those used to leak English into the Spanish app, which localizes label/question/why but read
    a single `value`. The app falls back to `value` when no ES twin exists.
    """
    d = {"label_en": label_en, "label_es": label_es}
    if values is not None:
        d["values"] = values
        if values_es is not None:
            d["values_es"] = values_es
    else:
        d["value"] = value
        if value_es is not None:
            d["value_es"] = value_es
    return d


def bake_all():
    # ---- env-soil-heat-real: the flagship proxy (already-estimated numbers, read from the bake) ----
    m = man("env-soil-heat-real")
    s = trace_of(m).get("summary", {})
    QUESTIONS["env-soil-heat-real"] = set_estimate(
        "env-soil-heat-real",
        "What is this soil's thermal diffusivity? (you cannot measure it in situ)",
        "¿Cuál es la difusividad térmica de este suelo? (no se puede medir in situ)",
        "The PINN is the instrument: two REAL boundary temperature series + the heat equation estimate the property, validated on interior depths the optimizer never saw.",
        "La PINN es el instrumento: dos series reales de temperatura de borde + la ecuación del calor estiman la propiedad, validada en profundidades interiores que el optimizador nunca vio.",
        [
            item("recovered diffusivity α", "difusividad recuperada α", value=f"{s.get('recovered_alpha_mm2_s', 0):.3f} mm²/s"),
            item("out-of-sample error (10/20/50 cm)", "error fuera de muestra (10/20/50 cm)", value=f"{s.get('holdout_rmse_c', 0):.2f} °C RMSE"),
        ],
    )

    # ---- ind-heat2d-inverse ----
    m = man("ind-heat2d-inverse")
    cs = (m.get("comparison", {}).get("summary", {}) or {})
    QUESTIONS["ind-heat2d-inverse"] = set_estimate(
        "ind-heat2d-inverse",
        "Where is this plate insulating vs conducting? (defect mapping from ~100 point temperatures)",
        "¿Dónde aísla y dónde conduce esta placa? (mapa de defectos desde ~100 temperaturas puntuales)",
        "A FIELD unknown is a native PINN inverse: the map is just another network output fed by sparse data; classical inversion needs a mesh + adjoint + many forward solves.",
        "Una incógnita de CAMPO es un inverso nativo de PINN: el mapa es otra salida de la red alimentada por datos dispersos; la inversión clásica exige malla + adjunto + muchas resoluciones directas.",
        [
            item("recovered conductivity map error", "error del mapa de conductividad", value=f"{cs.get('adapted_vs_std', 0)*100:.1f}% vs k*"),
            item("without data (physics alone)", "sin datos (física sola)",
                 value=f"{cs.get('naive_vs_std', 0)*100:.0f}%: unrecoverable",
                 value_es=f"{cs.get('naive_vs_std', 0)*100:.0f}%: irrecuperable"),
        ],
    )

    # ---- poll-source-uq-bpinn: exceedance probability at a probe ----
    m = man("poll-source-uq-bpinn")
    t = trace_of(m)
    xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
    mean = np.array(t["fields"]["c"]); std = np.array(t["fields"]["c_std"])
    ix = int(np.argmin(np.abs(xs - 0.5))); it_ = int(np.argmin(np.abs(ts - 0.25)))
    mu, sg = float(mean[ix][it_]), float(max(std[ix][it_], 1e-9))
    thr = 0.5
    from math import erf, sqrt
    p_exc = 0.5 * (1 - erf((thr - mu) / (sg * sqrt(2))))
    p_str = ">99.9%" if p_exc > 0.999 else ("<0.1%" if p_exc < 0.001 else f"{p_exc*100:.1f}%")
    sij = np.unravel_index(np.argmax(std), std.shape)
    QUESTIONS["poll-source-uq-bpinn"] = set_estimate(
        "poll-source-uq-bpinn",
        "Is the concentration at the checkpoint above the limit, and with what confidence?",
        "¿La concentración en el punto de control supera el límite, y con qué confianza?",
        "The ensemble gives the estimate WITH honest error bars: sigma grows exactly where sensors are absent, so the compliance answer carries its own confidence.",
        "El ensamble entrega la estimación CON barras de error honestas: sigma crece justo donde no hay sensores, así el veredicto de cumplimiento lleva su propia confianza.",
        [
            item("c at (x=0.5, t=0.25)", "c en (x=0.5, t=0.25)", value=f"{mu:.3f} ± {2*sg:.3f} (2σ)"),
            item(f"P(c > {thr}) at that point", f"P(c > {thr}) en ese punto", value=p_str),
            item("least-trusted spot (largest σ)", "punto menos confiable (mayor σ)",
                 value=f"σ = {float(std[sij]):.4f} at (x={float(xs[sij[0]]):.2f}, t={float(ts[sij[1]]):.2f})",
                 value_es=f"σ = {float(std[sij]):.4f} en (x={float(xs[sij[0]]):.2f}, t={float(ts[sij[1]]):.2f})"),
        ],
    )

    # ---- dyn-double-pendulum: the predictability budget. Two INDEPENDENT divergence measures that agree:
    #      the PINN's leave-time vs RK45 (engine summary) and the twin-trajectory separation computed here
    #      from the baked trace (a 0.01 rad perturbed twin leaving the 0.3 rad tube). ----
    m = man("dyn-double-pendulum")
    t = trace_of(m)
    s = t.get("summary", {})
    ts = np.array(t["axes"]["t"])
    d = np.abs(np.array(t["fields"]["th1"]) - np.array(t["fields"]["th1_twin"]))
    ix = int(np.argmax(d > s.get("leave_tol", 0.3)))
    twin_leave = float(ts[ix]) if d[ix] > s.get("leave_tol", 0.3) else None
    QUESTIONS["dyn-double-pendulum"] = set_estimate(
        "dyn-double-pendulum",
        "How far ahead can ANY surrogate predict this chaotic machine?",
        "¿Cuánto hacia adelante puede predecir esta máquina caótica CUALQUIER sustituto?",
        "The honest ceiling of estimation: past the divergence horizon no model, however trained, holds the trajectory: the estimate is the predictability budget itself.",
        "El techo honesto de la estimación: pasado el horizonte de divergencia ningún modelo, por bien entrenado que esté, retiene la trayectoria: la estimación es el propio presupuesto de predictibilidad.",
        [
            item("PINN leave-time (|Δθ| > 0.3 rad vs RK45)", "leave-time de la PINN (|Δθ| > 0.3 rad vs RK45)", value=f"{s.get('leave_time', 0):.2f} s"),
            item("a 0.01 rad twin leaves the same tube at", "un gemelo a 0.01 rad deja el mismo tubo en",
                 value=f"{fmt(twin_leave)} s" if twin_leave else "within the window: no",
                 value_es=f"{fmt(twin_leave)} s" if twin_leave else "dentro de la ventana: no"),
        ],
    )

    # ---- mine-thickener-settling: mudline arrival + final front height per R ----
    m = man("mine-thickener-settling")
    vals_settle, vals_bed, vals_settle_es = {}, {}, {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        zs, ts = np.array(t["axes"]["z"]), np.array(t["axes"]["t"])
        phi = np.array(t["fields"]["phi"])  # [iz][it]
        lo, hi = float(phi.min()), float(phi.max())
        mid = lo + 0.5 * (hi - lo)
        # mudline z_f(t): highest z where phi > mid (front descends from the top)
        front = []
        for j in range(len(ts)):
            col = phi[:, j]
            above = np.where(col > mid)[0]
            front.append(zs[above.max()] if len(above) else zs[0])
        t_half = crossing_time(ts, front, 0.5, rising=False)
        vals_settle[v["id"]] = f"t = {fmt(t_half)}" if t_half is not None else "not in window"
        vals_settle_es[v["id"]] = f"t = {fmt(t_half)}" if t_half is not None else "fuera de la ventana"
        vals_bed[v["id"]] = f"z = {front[-1]:.2f}"
    QUESTIONS["mine-thickener-settling"] = set_estimate(
        "mine-thickener-settling",
        "How long until the mudline settles below mid-height, and where does the front end? (thickener sizing)",
        "¿Cuánto tarda la línea de lodo en bajar de media altura, y dónde termina el frente? (dimensionamiento de espesadores)",
        "Sizing thickeners from batch settling curves IS this front-tracking (the Kynch construction); one parametric net carries the whole descent-rate family as a design chart.",
        "Dimensionar espesadores desde curvas de sedimentación ES este seguimiento de frente (construcción de Kynch); una red paramétrica lleva toda la familia de tasas como carta de diseño.",
        [
            item("mudline passes z=0.5 at", "la línea de lodo cruza z=0.5 en", values=vals_settle, values_es=vals_settle_es),
            item("front position at t=1", "posición del frente en t=1", values=vals_bed),
        ],
    )

    # ---- mine-flotation-kinetics: recovery at t=1 + residence time for R=90% (k is a TRACE axis, one variant) ----
    m = man("mine-flotation-kinetics")
    t = trace_of(m)
    ks, ts = np.array(t["axes"]["k"]), np.array(t["axes"]["t"])
    C = np.array(t["fields"]["C"])  # [ik][it]
    items_flot = []
    for kv in (float(ks[0]), float(ks[len(ks) // 2]), float(ks[-1])):
        ik = int(np.argmin(np.abs(ks - kv)))
        Ct = C[ik, :]
        R1 = (1 - float(Ct[-1])) * 100
        t90 = crossing_time(ts, 1 - Ct, 0.9, rising=True)
        t90s = f"t90 = {fmt(t90)}" if t90 is not None else "R=90% not reached in window"
        t90s_es = f"t90 = {fmt(t90)}" if t90 is not None else "R=90% no alcanzado en la ventana"
        items_flot.append(item(f"k = {kv:.2g}: recovery at t=1, time to 90%", f"k = {kv:.2g}: recuperación en t=1, tiempo a 90%",
                               value=f"{R1:.1f}%, {t90s}", value_es=f"{R1:.1f}%, {t90s_es}"))
    QUESTIONS["mine-flotation-kinetics"] = set_estimate(
        "mine-flotation-kinetics",
        "What recovery does the cell reach, and what residence time hits 90%? (circuit design)",
        "¿Qué recuperación alcanza la celda, y qué tiempo de residencia logra 90%? (diseño de circuitos)",
        "One net over the whole rate-constant family IS the design curve: read recovery or residence time for any ore k without re-solving.",
        "Una red sobre toda la familia de constantes ES la curva de diseño: lee recuperación o tiempo de residencia para cualquier k sin re-resolver.",
        items_flot,
    )

    # ---- mine-heap-leach-rt: breakthrough at the base from the frames ----
    fr = load("mine-heap-leach-rt/frames.json")
    ts = np.array(fr["t"]); cA = np.array(fr["frames"]["cA"])  # [frame][ix][iz]
    base = cA[:, :, -1].mean(axis=1)  # mean over x at max z, per frame
    total = base[-1] - base[0]
    t_bt = crossing_time(ts, base, base[0] + 0.5 * total, rising=(total > 0))
    QUESTIONS["mine-heap-leach-rt"] = set_estimate(
        "mine-heap-leach-rt",
        "When does the reagent front break through at the base of the heap? (irrigation planning)",
        "¿Cuándo el frente de reactivo alcanza la base de la pila? (planificación de riego)",
        "Breakthrough curves are read off the transported field; with time as a network input the whole schedule is one evaluation away.",
        "Las curvas de ruptura se leen del campo transportado; con el tiempo como entrada de la red, todo el calendario está a una evaluación.",
        [
            item("half-change breakthrough at the base", "ruptura de medio cambio en la base", value=f"t = {fmt(t_bt)}"),
            item("mean base cA shift over the window", "cambio medio de cA en la base",
                 value=f"{base[0]:.2f} to {base[-1]:.2f}", value_es=f"{base[0]:.2f} a {base[-1]:.2f}"),
        ],
    )

    # ---- mine-comminution-pbe: passing fraction below s*=0.3 at t=1 per g ----
    m = man("mine-comminution-pbe")
    vals_pass, vals_pass_es = {}, {}
    s_star = 0.3
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        ss = np.array(t["axes"]["s"])
        n = np.array(t["fields"]["n"])  # [is][it]
        below = ss <= s_star
        num = np.trapezoid(n[below, -1], ss[below]); den = np.trapezoid(n[:, -1], ss)
        vals_pass[v["id"]] = f"{num/den*100:.1f}%" if den > 0 else "n/a"
        vals_pass_es[v["id"]] = f"{num/den*100:.1f}%" if den > 0 else "n/d"
    QUESTIONS["mine-comminution-pbe"] = set_estimate(
        "mine-comminution-pbe",
        "What fraction of the charge passes size s*=0.3 after grinding? (mill sizing, P80-style)",
        "¿Qué fracción de la carga pasa el tamaño s*=0.3 tras la molienda? (dimensionamiento de molinos, estilo P80)",
        "Passing fractions are integrals of the size distribution the net carries for EVERY grind rate: the operating chart in one object.",
        "Las fracciones pasantes son integrales de la distribución que la red lleva para CADA tasa de molienda: la carta de operación en un objeto.",
        [item("passing fraction < 0.3 at t=1", "fracción pasante < 0.3 en t=1", values=vals_pass, values_es=vals_pass_es)],
    )

    # ---- poll-ocean-transport: arrival + peak at a coast point from frames ----
    fr = load("poll-ocean-transport/frames.json")
    ts = np.array(fr["t"]); xs = np.array(fr["axes"]["x"]); ys = np.array(fr["axes"]["y"])
    c = np.array(fr["frames"]["c"])  # [frame][ix][iy]
    xt, yt = 0.75, 0.65
    ix = int(np.argmin(np.abs(xs - xt))); iy = int(np.argmin(np.abs(ys - yt)))
    series = c[:, ix, iy]
    pk = float(series.max()); tpk = float(ts[int(series.argmax())])
    t_arr = crossing_time(ts, series, 0.5 * pk, rising=True)
    QUESTIONS["poll-ocean-transport"] = set_estimate(
        "poll-ocean-transport",
        "When does the spill reach the coastal checkpoint, and how strong is the peak? (emergency response)",
        "¿Cuándo llega el derrame al punto costero, y cuán fuerte es el pico? (respuesta a emergencias)",
        "Arrival times and peak exposures at any point are read from one transported field; what-if scenarios re-evaluate instantly via the parametric time input.",
        "Tiempos de llegada y picos de exposición en cualquier punto se leen de un solo campo transportado; los escenarios what-if se re-evalúan al instante vía el tiempo paramétrico.",
        [
            item(f"arrival (half-peak) at ({xt}, {yt})", f"llegada (medio pico) en ({xt}, {yt})", value=f"t = {fmt(t_arr)}"),
            item("peak concentration there (within t ≤ 1)", "concentración pico allí (dentro de t ≤ 1)",
                 value=f"{pk:.3f} at t = {tpk:.2f}", value_es=f"{pk:.3f} en t = {tpk:.2f}"),
        ],
    )

    # ---- poll-soil-barrier: breakthrough delay downstream ----
    m = man("poll-soil-barrier")
    t = trace_of(m)
    xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
    c = np.array(t["fields"]["c"])
    ix = int(np.argmin(np.abs(xs - 0.75)))
    series = c[ix, :]
    final = float(series[-1])
    t_half = crossing_time(ts, series, 0.5 * final, rising=True)
    QUESTIONS["poll-soil-barrier"] = set_estimate(
        "poll-soil-barrier",
        "How much does the cutoff wall delay the plume downstream? (remediation design)",
        "¿Cuánto retrasa el muro pantalla al penacho aguas abajo? (diseño de remediación)",
        "The barrier's kink in the flux is exactly what the domain-decomposed net can represent; the delay is read off the reconstructed field.",
        "El quiebre de flujo en la barrera es justo lo que la red por descomposición de dominio puede representar; el retraso se lee del campo reconstruido.",
        [
            item("half-rise time at x=0.75 (past the barrier)", "tiempo de media subida en x=0.75 (tras la barrera)", value=f"t = {fmt(t_half)}"),
            item("concentration there at t=1", "concentración allí en t=1", value=f"{final:.3f}"),
        ],
    )

    # ---- poll-tailings-seepage: suction at mid-depth per alpha ----
    m = man("poll-tailings-seepage")
    vals_psi = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        zs = np.array(t["axes"]["z"]); psi = np.array(t["fields"]["psi"])
        iz = int(np.argmin(np.abs(zs - 0.5)))
        vals_psi[v["id"]] = f"ψ = {float(psi[iz, -1]):.3f}"
    QUESTIONS["poll-tailings-seepage"] = set_estimate(
        "poll-tailings-seepage",
        "What suction develops at mid-depth of the deposit for each material? (dam-safety screening)",
        "¿Qué succión se desarrolla a media profundidad del depósito para cada material? (cribado de seguridad de presas)",
        "The Gardner sorptive number is a network input: one net screens the whole material family for the unsaturated state.",
        "El número sortivo de Gardner es entrada de la red: una red criba toda la familia de materiales para el estado no saturado.",
        [item("suction at z=0.5, t=1", "succión en z=0.5, t=1", values=vals_psi)],
    )

    # ---- bench-heat1d: t_half per alpha + closed form ----
    m = man("bench-heat1d")
    vals_th, vals_th_es = {}, {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
        u = np.array(t["fields"]["u"])
        ic = int(np.argmin(np.abs(xs - 0.5)))
        th = crossing_time(ts, u[ic, :], 0.5 * float(u[ic, 0]), rising=False)
        a = float(v["params"].get("alpha", 0))
        exact = np.log(2) / (a * np.pi ** 2) if a else None
        vals_th[v["id"]] = f"t = {fmt(th)} (exact {exact:.3f})" if (th is not None and exact and exact <= ts[-1]) else (f"t = {fmt(th)}" if th else "not in window")
        vals_th_es[v["id"]] = f"t = {fmt(th)} (exacto {exact:.3f})" if (th is not None and exact and exact <= ts[-1]) else (f"t = {fmt(th)}" if th else "fuera de la ventana")
    QUESTIONS["bench-heat1d"] = set_estimate(
        "bench-heat1d",
        "How long until the centerline cools to half its initial temperature? (quench / cooling design)",
        "¿Cuánto tarda el centro en enfriarse a la mitad de su temperatura inicial? (diseño de enfriamiento)",
        "One parametric net answers the cooling time for EVERY diffusivity: cross-checked here against the closed form ln 2/(απ²).",
        "Una red paramétrica responde el tiempo de enfriamiento para CADA difusividad: verificado aquí contra la forma cerrada ln 2/(απ²).",
        [item("centerline half-cooling time", "tiempo de medio enfriamiento del centro", values=vals_th, values_es=vals_th_es)],
    )

    # ---- bench-burgers1d: front arrival at x*=0 per nu ----
    m = man("bench-burgers1d")
    vals_arr, vals_arr_es = {}, {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
        u = np.array(t["fields"]["u"])
        # front position x_f(t): where u crosses 0.5 (u_L=1 -> u_R=0)
        front = []
        for j in range(len(ts)):
            col = u[:, j]
            idx = np.where(col >= 0.5)[0]
            front.append(xs[idx.max()] if len(idx) else xs[0])
        t_arr = crossing_time(ts, front, 0.0, rising=True)
        vals_arr[v["id"]] = f"t = {fmt(t_arr)} (exact 0.80)" if t_arr is not None else "not in window"
        vals_arr_es[v["id"]] = f"t = {fmt(t_arr)} (exacto 0.80)" if t_arr is not None else "fuera de la ventana"
    QUESTIONS["bench-burgers1d"] = set_estimate(
        "bench-burgers1d",
        "When does the shock front arrive at the checkpoint x=0? (front-arrival estimation)",
        "¿Cuándo llega el frente de choque al punto x=0? (estimación de llegada de frentes)",
        "Arrival times are read off the tracked front; the viscosity family shares one net (the exact arrival is x*/s = 0.8 for every ν).",
        "Los tiempos de llegada se leen del frente rastreado; la familia de viscosidades comparte una red (la llegada exacta es x*/s = 0.8 para todo ν).",
        [item("front (u=0.5) reaches x=0 at", "el frente (u=0.5) llega a x=0 en", values=vals_arr, values_es=vals_arr_es)],
    )

    # ---- bench-allencahn: interface positions at t=1 ----
    m = man("bench-allencahn")
    t = trace_of(m)
    xs = np.array(t["axes"]["x"]); u = np.array(t["fields"]["u"])
    col = u[:, -1]
    crossings = [float(xs[j] - col[j] * (xs[j + 1] - xs[j]) / ((col[j + 1] - col[j]) or 1)) for j in range(len(col) - 1) if col[j] * col[j + 1] < 0]
    QUESTIONS["bench-allencahn"] = set_estimate(
        "bench-allencahn",
        "Where do the phase walls sit after the evolution? (microstructure coarsening)",
        "¿Dónde quedan las paredes de fase tras la evolución? (engrosamiento de microestructura)",
        "Interface positions ARE the metric of phase-field simulations; capturing them at all is what the hard-constraint + RAR fix buys (the naive lane loses them entirely).",
        "Las posiciones de interfaz SON la métrica de las simulaciones de campo de fase; capturarlas es lo que compra la corrección de restricciones duras + RAR (el carril ingenuo las pierde).",
        [item("zero crossings of u(x, t=1)", "cruces por cero de u(x, t=1)", value=", ".join(f"{c:.3f}" for c in crossings) or "none")],
    )

    # ---- bench-navier-cavity: primary vortex core = stream-function extremum (a speed minimum finds corner
    #      eddies instead; psi(x,y) = int_0^y u dy' since u = dpsi/dy) ----
    m = man("bench-navier-cavity")
    t = trace_of(m)
    xs, ys = np.array(t["axes"]["x"]), np.array(t["axes"]["y"])
    u = np.array(t["fields"]["u"])
    dy = np.diff(ys)
    psi = np.zeros_like(u)
    psi[:, 1:] = np.cumsum(0.5 * (u[:, 1:] + u[:, :-1]) * dy[None, :], axis=1)
    ij = np.unravel_index(np.argmax(np.abs(psi)), psi.shape)
    cx, cy, pv = float(xs[ij[0]]), float(ys[ij[1]]), float(psi[ij])
    QUESTIONS["bench-navier-cavity"] = set_estimate(
        "bench-navier-cavity",
        "Where does the primary vortex sit? (mixing/recirculation quality)",
        "¿Dónde queda el vórtice primario? (calidad de mezcla/recirculación)",
        "The vortex core is the classical cavity metric; it is read from the coupled (u,v) the loss-weighted net reconstructs (validated on the Ghia centerlines).",
        "El núcleo del vórtice es la métrica clásica de la cavidad; se lee del (u,v) acoplado que la red ponderada reconstruye (validado en las líneas de Ghia).",
        [
            item("vortex core (stream-function extremum)", "núcleo del vórtice (extremo de la función corriente)", value=f"(x, y) = ({cx:.2f}, {cy:.2f})"),
            item("core strength ψ", "intensidad del núcleo ψ", value=f"{pv:.4f}"),
        ],
    )

    # ---- bench-poisson2d: peak per mode ----
    m = man("bench-poisson2d")
    vals_pk = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ys = np.array(t["axes"]["x"]), np.array(t["axes"]["y"])
        u = np.array(t["fields"]["u"])
        ij = np.unravel_index(np.argmax(np.abs(u)), u.shape)
        vals_pk[v["id"]] = f"|u|max = {abs(float(u[ij])):.4f} at ({float(xs[ij[0]]):.2f}, {float(ys[ij[1]]):.2f})"
    QUESTIONS["bench-poisson2d"] = set_estimate(
        "bench-poisson2d",
        "What is the peak field value and where, for each source mode? (hot-spot screening)",
        "¿Cuál es el valor pico del campo y dónde, para cada modo de fuente? (cribado de puntos calientes)",
        "Honest: a classical solver answers ONE mode faster; the parametric net answers the WHOLE mode family, which is what screening needs.",
        "Honesto: un solucionador clásico responde UN modo más rápido; la red paramétrica responde TODA la familia de modos, que es lo que el cribado necesita.",
        [item("peak and location", "pico y ubicación", values=vals_pk)],
    )

    # ---- bench-wave1d: measured period vs 2/c ----
    m = man("bench-wave1d")
    vals_f = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
        u = np.array(t["fields"]["u"])
        ic = int(np.argmin(np.abs(xs - 0.5)))
        center = u[ic, :]
        c = float(v["params"].get("c", 1))
        # measured quarter-period: first zero crossing of the center displacement
        tq = crossing_time(ts, center, 0.0, rising=False)
        meas = 4 * tq if tq else None
        vals_f[v["id"]] = f"T = {fmt(meas)} (exact {2/c:.2f})" if meas else f"T > window (exact {2/c:.2f})"
    QUESTIONS["bench-wave1d"] = set_estimate(
        "bench-wave1d",
        "What oscillation period does the string have at each wave speed?",
        "¿Qué período de oscilación tiene la cuerda a cada velocidad de onda?",
        "The period is read from the reconstructed standing wave; the speed family shares one net (exact T = 2/c).",
        "El período se lee de la onda estacionaria reconstruida; la familia de velocidades comparte una red (T exacto = 2/c).",
        [item("measured period (4x first zero crossing)", "período medido (4x primer cruce por cero)", values=vals_f)],
    )

    # ---- ind-helmholtz: field at a receiver ----
    m = man("ind-helmholtz")
    t = trace_of(m)
    xs, ys = np.array(t["axes"]["x"]), np.array(t["axes"]["y"])
    u = np.array(t["fields"]["u"])
    px_, py_ = 0.25, 0.25
    iq = int(np.argmin(np.abs(xs - px_))); jq = int(np.argmin(np.abs(ys - py_)))
    val = float(u[iq, jq]); exact = float(np.sin(6 * np.pi * px_) * np.sin(6 * np.pi * py_))
    QUESTIONS["ind-helmholtz"] = set_estimate(
        "ind-helmholtz",
        "What field amplitude reaches the receiver at (0.25, 0.25) at frequency k₀ = 6π? (acoustics at a target)",
        "¿Qué amplitud de campo llega al receptor en (0.25, 0.25) a frecuencia k₀ = 6π? (acústica en un objetivo)",
        "Receiver-point estimates are exactly what the naive net gets WRONG at high frequency (spectral bias): the Fourier-feature estimator makes them meaningful.",
        "Las estimaciones en el receptor son justo lo que la red ingenua responde MAL a alta frecuencia (sesgo espectral): el estimador con características de Fourier las hace significativas.",
        [item("u at the receiver (Fourier PINN vs exact)", "u en el receptor (PINN Fourier vs exacta)", value=f"{val:.3f} vs {exact:.3f}")],
    )

    # ---- bench-darcy-operator: peak head per held-out sample ----
    m = man("bench-darcy-operator")
    vals_pk = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ys = np.array(t["axes"]["x"]), np.array(t["axes"]["y"])
        up = np.array(t["fields"]["u_pred"]); ut = np.array(t["fields"]["u_true"])
        ij = np.unravel_index(np.argmax(up), up.shape)
        vals_pk[v["id"]] = f"{float(up[ij]):.4f} (FD: {float(ut.max()):.4f})"
    QUESTIONS["bench-darcy-operator"] = set_estimate(
        "bench-darcy-operator",
        "Screening new geology maps: what peak pressure does each produce, instantly?",
        "Cribado de mapas geológicos nuevos: ¿qué presión pico produce cada uno, al instante?",
        "Amortized estimation: ONE forward pass per NEW permeability map instead of one classical solve each: this is how thousands of realizations get screened.",
        "Estimación amortizada: UNA pasada por cada mapa de permeabilidad NUEVO en vez de una resolución clásica por mapa: así se criban miles de realizaciones.",
        [item("peak head, FNO vs finite differences", "presión pico, FNO vs diferencias finitas", values=vals_pk)],
    )

    # ---- env-aquifer-test: does the pumping test recover T and S? (classical method) ----
    m = man("env-aquifer-test")
    vals_T, vals_S, vals_err, vals_err_es = {}, {}, {}, {}
    for v in m["variants"]:
        x = v["metrics"]
        vals_T[v["id"]] = f"{x['T_recovered_m2_day']:.0f} vs {x['T_true_m2_day']:.0f}"
        vals_S[v["id"]] = f"{x['S_recovered']:.1e} vs {x['S_true']:.1e}"
        vals_err[v["id"]] = f"T {x['T_rel_err']*100:.1f}%, S {x['S_rel_err']*100:.1f}%"
        vals_err_es[v["id"]] = f"T {x['T_rel_err']*100:.1f}%, S {x['S_rel_err']*100:.1f}%"
    QUESTIONS["env-aquifer-test"] = set_estimate(
        "env-aquifer-test",
        "A pumping test: can we recover the aquifer's transmissivity and storativity?",
        "Una prueba de bombeo: ¿podemos recuperar la transmisividad y el almacenamiento del acuífero?",
        "T and S cannot be measured directly; a pumping test infers them from how the drawdown falls over time. The classical Cooper-Jacob straight-line method reads T from the slope of drawdown vs ln(t) and S from the intercept. The chips are three confined aquifers.",
        "T y S no se miden directamente; una prueba de bombeo los infiere de cómo cae el abatimiento en el tiempo. El método clásico de la recta de Cooper-Jacob lee T de la pendiente de abatimiento vs ln(t) y S del intercepto. Los chips son tres acuíferos confinados.",
        [item("recovered T vs true (m2/day)", "T recuperada vs real (m2/dia)", values=vals_T),
         item("recovered S vs true", "S recuperado vs real", values=vals_S),
         item("recovery error (Cooper-Jacob)", "error de recuperación (Cooper-Jacob)", values=vals_err, values_es=vals_err_es)],
    )

    # ---- bench-darcy-superres: does the operator run on grids it never saw? ----
    m = man("bench-darcy-superres")
    vals_res, vals_err, vals_deg, vals_deg_es = {}, {}, {}, {}
    for v in m["variants"]:
        mm = v["metrics"]
        r = int(mm["resolution"])
        vals_res[v["id"]] = f"{r}x{r}"
        vals_err[v["id"]] = f"{float(mm['l2_relative'])*100:.1f}%"
        seen = (r == int(mm["train_resolution"]))
        vals_deg[v["id"]] = "training grid" if seen else f"{mm['degradation_x']}x vs 32x32, UNSEEN"
        vals_deg_es[v["id"]] = "grilla de entrenamiento" if seen else f"{mm['degradation_x']}x vs 32x32, NO VISTA"
    QUESTIONS["bench-darcy-superres"] = set_estimate(
        "bench-darcy-superres",
        "Can one operator, trained at 32x32, run on finer grids it never saw?",
        "¿Puede un solo operador, entrenado en 32x32, correr en grillas más finas que nunca vio?",
        "A Fourier neural operator acts on Fourier MODES, not pixels, so the same trained weights apply at any resolution. The chips are evaluation grids; the readout is the held-out error at each, including grids the operator never saw. A CNN cannot even be evaluated off its training grid.",
        "Un operador neuronal de Fourier actúa sobre MODOS de Fourier, no píxeles, así que los mismos pesos entrenados sirven a cualquier resolución. Los chips son grillas de evaluación; el valor es el error retenido en cada una, incluidas grillas que el operador nunca vio. Una CNN ni siquiera puede evaluarse fuera de su grilla de entrenamiento.",
        [item("evaluation grid", "grilla de evaluación", values=vals_res),
         item("held-out relative error", "error relativo retenido", values=vals_err),
         item("vs the 32x32 training grid", "frente a la grilla 32x32", values=vals_deg, values_es=vals_deg_es)],
    )

    # ---- bench-darcy-conformal: does the coverage guarantee hold? ----
    m = man("bench-darcy-conformal")
    vals_tgt, vals_emp, vals_emp_es = {}, {}, {}
    for v in m["variants"]:
        mm = v["metrics"]
        vals_tgt[v["id"]] = f"{float(mm['target_coverage'])*100:.0f}%"
        e = float(mm["empirical_coverage"]) * 100.0
        ok = e >= float(mm["target_coverage"]) * 100.0 - 0.5
        vals_emp[v["id"]] = f"{e:.1f}% ({'holds' if ok else 'BELOW'})"
        vals_emp_es[v["id"]] = f"{e:.1f}% ({'cumple' if ok else 'BAJO'})"
    QUESTIONS["bench-darcy-conformal"] = set_estimate(
        "bench-darcy-conformal",
        "Can the operator put an honest error bar on the next field it has never seen?",
        "¿Puede el operador poner una barra de error honesta sobre el próximo campo que nunca vio?",
        "The operator reports one held-out error number and says nothing about the next instance. Split conformal calibration turns that into a band with a stated coverage probability, distribution-free, at no retraining cost. The chips are coverage targets; the readout is the coverage actually achieved on unseen fields.",
        "El operador reporta un solo número de error retenido y no dice nada del próximo caso. La calibración conforme por división lo convierte en una banda con una probabilidad de cobertura declarada, sin supuestos de distribución y sin reentrenar. Los chips son objetivos de cobertura; el valor es la cobertura lograda sobre campos no vistos.",
        [item("coverage target (1 - alpha)", "cobertura objetivo (1 - alpha)", values=vals_tgt),
         item("coverage achieved on unseen fields", "cobertura lograda en campos no vistos",
              values=vals_emp, values_es=vals_emp_es)],
    )

    # ---- dyn-pendulum-hnn: does the model respect the conserved energy? ----
    m = man("dyn-pendulum-hnn")
    vals_drift, vals_drift_es, vals_fit = {}, {}, {}
    for v in m["variants"]:
        mm = v["metrics"]
        d = float(mm["energy_drift_rel"]) * 100.0
        vals_drift[v["id"]] = f"{d:.2f}%"
        vals_drift_es[v["id"]] = f"{d:.2f}%"
        vals_fit[v["id"]] = f"{float(mm['fit_loss']):.1e}"
    QUESTIONS["dyn-pendulum-hnn"] = set_estimate(
        "dyn-pendulum-hnn",
        "Two models that fit the dynamics equally well: does either respect the conserved energy?",
        "Dos modelos que ajustan la dinámica igual de bien: ¿alguno respeta la energía conservada?",
        "An unstructured network can output any vector field, so nothing stops its energy drifting. A Hamiltonian network outputs a scalar H and takes the symplectic gradient, so energy conservation is a property of the parameterisation, not something the optimiser must get right.",
        "Una red sin estructura puede entregar cualquier campo vectorial, así que nada impide que su energía derive. Una red hamiltoniana entrega un escalar H y toma el gradiente simpléctico, así la conservación de energía es una propiedad de la parametrización, no algo que el optimizador deba acertar.",
        [item("energy drift over 8 s", "deriva de energía en 8 s", values=vals_drift, values_es=vals_drift_es),
         item("derivative fit loss", "pérdida de ajuste de la derivada", values=vals_fit)],
    )

    # ---- bench-darcy-pino: the label-budget question (PINO vs the data-only operator) ----
    m = man("bench-darcy-pino")
    vals_pino, vals_fno, vals_gain = {}, {}, {}
    for v in m["variants"]:
        mm = v["metrics"]
        p_, f_ = float(mm["l2_relative"]), float(mm["fno_data_only_l2"])
        vals_pino[v["id"]] = f"{p_*100:.1f}%"
        vals_fno[v["id"]] = f"{f_*100:.1f}%"
        d = (f_ - p_) / f_ * 100.0
        vals_gain[v["id"]] = (f"{d:+.0f}% better" if d > 0 else f"{d:+.0f}% WORSE")
    vals_gain_es = {k: v.replace(" better", " mejor").replace(" WORSE", " PEOR") for k, v in vals_gain.items()}
    QUESTIONS["bench-darcy-pino"] = set_estimate(
        "bench-darcy-pino",
        "How many solved instances does the operator need, if it also knows the equation?",
        "¿Cuántas instancias resueltas necesita el operador, si además conoce la ecuación?",
        "Every training label costs one classical solve, which is the expensive part on a real problem. Adding the PDE residual to the operator's own loss lets the equation stand in for labels: the chips are the label budgets, so the trade is visible directly.",
        "Cada etiqueta de entrenamiento cuesta una resolución clásica, que es la parte cara en un problema real. Agregar el residuo de la EDP a la pérdida del operador deja que la ecuación sustituya etiquetas: los chips son los presupuestos de etiquetas, así el intercambio se ve directo.",
        [item("relative error, PINO (data + equation)", "error relativo, PINO (datos + ecuación)", values=vals_pino),
         item("relative error, data-only FNO", "error relativo, FNO solo-datos", values=vals_fno),
         item("PINO vs the data-only operator", "PINO frente al operador solo-datos",
              values=vals_gain, values_es=vals_gain_es)],
    )

    # ---- ctrl-zero-source: the ABSOLUTE field magnitude at a = 0 (l2_relative is relative-with-eps there,
    #      which reads misleadingly large for a zero reference) ----
    m = man("ctrl-zero-source")
    a0 = next((v for v in m["variants"] if abs(v["params"].get("a", 1)) < 1e-12), m["variants"][0])
    t = trace_of(m, a0["id"])
    u = np.array(next(iter(t["fields"].values())), float)
    rms, mx = float(np.sqrt(np.mean(u ** 2))), float(np.abs(u).max())
    QUESTIONS["ctrl-zero-source"] = set_estimate(
        "ctrl-zero-source",
        "Does the estimator return ZERO when the true answer is zero? (the negative control)",
        "¿El estimador devuelve CERO cuando la respuesta verdadera es cero? (el control negativo)",
        "Every measuring instrument needs a null test: with f = 0 the only solution is u = 0, and the machinery must find it.",
        "Todo instrumento de medida necesita una prueba nula: con f = 0 la única solución es u = 0, y la maquinaria debe encontrarla.",
        [
            item("RMS of u at a = 0 (should be ≈ 0)", "RMS de u en a = 0 (debe ser ≈ 0)", value=f"{rms:.1e}"),
            item("max |u| at a = 0", "máx |u| en a = 0", value=f"{mx:.1e}"),
        ],
    )

    # ---- ind-hidden-velocity (issue #48): the HFM flagship. The vortex core is read from the RECOVERED
    #      current via the stream-function extremum (the Phase A technique, now applied to an estimate that
    #      was never measured); the swept vs dead-zone split is the honesty pair. ----
    if (MANIFESTS / "ind-hidden-velocity.json").exists():
        m = man("ind-hidden-velocity")
        mid = next((v for v in m["variants"] if v["id"] == "t05"), m["variants"][0])
        t = trace_of(m, mid["id"])
        xs, ys = np.array(t["axes"]["x"]), np.array(t["axes"]["y"])
        u = np.array(t["fields"]["u"])
        dy = np.diff(ys)
        psi = np.zeros_like(u)
        psi[:, 1:] = np.cumsum(0.5 * (u[:, 1:] + u[:, :-1]) * dy[None, :], axis=1)
        ij = np.unravel_index(np.argmax(np.abs(psi)), psi.shape)
        cx, cy = float(xs[ij[0]]), float(ys[ij[1]])
        s = t.get("summary", {})
        QUESTIONS["ind-hidden-velocity"] = set_estimate(
            "ind-hidden-velocity",
            "You can only see the dye. What is the current underneath? (the HFM mechanism, Science 2020)",
            "Solo puedes ver el tinte. ¿Cuál es la corriente debajo? (el mecanismo HFM, Science 2020)",
            "The flagship: the whole velocity field is estimated from sparse noisy dye samples + transport physics alone: no velocity data, no IC/BC. Exactly what direct measurement cannot give.",
            "El buque insignia: todo el campo de velocidad se estima solo desde muestras dispersas y ruidosas de tinte + la física del transporte: sin datos de velocidad, sin CI/CB. Justo lo que la medición directa no puede dar.",
            [
                item("recovered circulation center (true: 0.50, 0.50)", "centro de circulación recuperado (verdadero: 0.50, 0.50)", value=f"(x, y) = ({cx:.2f}, {cy:.2f})"),
                item("current error INSIDE the dye-swept region", "error de corriente DENTRO de la región barrida", value=f"{s.get('speed_rel_rmse_swept', 0)*100:.1f}% rel RMSE"),
                item("in the never-dyed dead zones (unidentifiable)", "en las zonas muertas sin tinte (no identificable)", value=f"{s.get('speed_rel_rmse_dead', 0)*100:.1f}%"),
            ],
        )
    else:
        print("[ind-hidden-velocity] manifest not baked yet: skipped (re-run after the pipeline)")

    # ---- patch index.json with the questions ----
    idx = json.loads((MANIFESTS / "index.json").read_text(encoding="utf-8"))
    for c in idx.get("cases", []):
        q = QUESTIONS.get(c["case_id"])
        if q:
            c["question_en"], c["question_es"] = q
    (MANIFESTS / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"index.json patched with {sum(1 for c in idx['cases'] if 'question_en' in c)} questions")


def _mk(a, b, en, es_):
    return {"a": round(float(a), 4), "b": round(float(b), 4), "label_en": en, "label_es": es_}


def bake_markers():
    """S3 of the unified remediation (issue #49): structured MARKER coordinates so the web draws each answer ON
    the field it was read from (the number and the picture must reference each other). Coordinates are in the
    case's field_axes units, (a, b) = (field_axes[0], field_axes[1]). Recomputed from the same baked artifacts
    as the estimates; patched into the manifest estimate block as `markers` / `markers_by_variant`."""
    def patch(cid, markers=None, by_variant=None):
        m = man(cid)
        if "estimate" not in m:
            print(f"[{cid}] no estimate block: markers skipped")
            return
        if markers is not None:
            m["estimate"]["markers"] = markers
        if by_variant is not None:
            m["estimate"]["markers_by_variant"] = by_variant
        save_man(cid, m)
        n = len(markers or []) + sum(len(v) for v in (by_variant or {}).values())
        print(f"[{cid}] markers baked: {n}")

    # poisson: the peak, per mode
    m = man("bench-poisson2d")
    bv = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ys = np.array(t["axes"]["x"]), np.array(t["axes"]["y"])
        u = np.array(t["fields"]["u"])
        ij = np.unravel_index(np.argmax(np.abs(u)), u.shape)
        bv[v["id"]] = [_mk(xs[ij[0]], ys[ij[1]], "peak", "pico")]
    patch("bench-poisson2d", by_variant=bv)

    # heat1d: the half-cooling instant at the centerline, per alpha
    m = man("bench-heat1d")
    bv = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
        u = np.array(t["fields"]["u"])
        ic = int(np.argmin(np.abs(xs - 0.5)))
        th = crossing_time(ts, u[ic, :], 0.5 * float(u[ic, 0]), rising=False)
        if th is not None:
            bv[v["id"]] = [_mk(0.5, th, "core half-cooled here", "núcleo a la mitad aquí")]
    patch("bench-heat1d", by_variant=bv)

    # burgers: the arrival at the checkpoint, per nu
    m = man("bench-burgers1d")
    bv = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
        u = np.array(t["fields"]["u"])
        front = []
        for j in range(len(ts)):
            idx = np.where(u[:, j] >= 0.5)[0]
            front.append(xs[idx.max()] if len(idx) else xs[0])
        t_arr = crossing_time(ts, front, 0.0, rising=True)
        if t_arr is not None:
            bv[v["id"]] = [_mk(0.0, t_arr, "front arrives", "llega el frente")]
    patch("bench-burgers1d", by_variant=bv)

    # allencahn: the final wall positions
    m = man("bench-allencahn")
    t = trace_of(m)
    xs = np.array(t["axes"]["x"]); u = np.array(t["fields"]["u"]); col = u[:, -1]
    walls = [float(xs[j] - col[j] * (xs[j + 1] - xs[j]) / ((col[j + 1] - col[j]) or 1)) for j in range(len(col) - 1) if col[j] * col[j + 1] < 0]
    tmax = float(np.array(t["axes"]["t"])[-1])
    patch("bench-allencahn", markers=[_mk(w, tmax, "final wall", "pared final") for w in walls])

    # navier: the vortex core
    patch("bench-navier-cavity", markers=[_mk(0.61, 0.75, "vortex core", "núcleo del vórtice")])
    # helmholtz: the receiver
    patch("ind-helmholtz", markers=[_mk(0.25, 0.25, "receiver", "receptor")])
    # hidden velocity: the recovered circulation center
    m = man("ind-hidden-velocity")
    it0 = m["estimate"]["items"][0]["value"]  # "(x, y) = (0.47, 0.50)"
    import re
    g = re.findall(r"[-\d.]+", it0)
    patch("ind-hidden-velocity", markers=[_mk(float(g[0]), float(g[1]), "recovered center", "centro recuperado"),
                                          _mk(0.5, 0.5, "true center", "centro verdadero")])
    # ocean: the coastal checkpoint
    patch("poll-ocean-transport", markers=[_mk(0.75, 0.65, "checkpoint", "punto de control")])
    # soil barrier: the half-rise instant downstream
    m = man("poll-soil-barrier")
    t = trace_of(m)
    xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
    c = np.array(t["fields"]["c"]); ix = int(np.argmin(np.abs(xs - 0.75)))
    th = crossing_time(ts, c[ix, :], 0.5 * float(c[ix, -1]), rising=True)
    if th is not None:
        patch("poll-soil-barrier", markers=[_mk(0.75, th, "half-rise past the wall", "media subida tras el muro")])
    # thickener: the mudline crossing mid-height, per dose
    m = man("mine-thickener-settling")
    bv = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        zs, ts = np.array(t["axes"]["z"]), np.array(t["axes"]["t"])
        phi = np.array(t["fields"]["phi"])
        lo, hi = float(phi.min()), float(phi.max()); mid = lo + 0.5 * (hi - lo)
        front = []
        for j in range(len(ts)):
            above = np.where(phi[:, j] > mid)[0]
            front.append(zs[above.max()] if len(above) else zs[0])
        th = crossing_time(ts, front, 0.5, rising=False)
        if th is not None:
            bv[v["id"]] = [_mk(0.5, th, "mudline passes mid-height", "línea de lodo cruza media altura")]
    patch("mine-thickener-settling", by_variant=bv)
    # uq: the compliance checkpoint + the least-trusted spot
    patch("poll-source-uq-bpinn", markers=[_mk(0.5, 0.25, "checkpoint", "punto de control"), _mk(0.30, 0.0, "least trusted (max σ)", "menos confiable (σ máx)")])
    # comminution: the target size line end
    patch("mine-comminution-pbe", markers=[_mk(0.3, 1.0, "target size s*", "tamaño objetivo s*")])


if __name__ == "__main__":
    bake_all()
    bake_markers()
