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
import io
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


def item(label_en, label_es, value=None, values=None):
    d = {"label_en": label_en, "label_es": label_es}
    if values is not None:
        d["values"] = values
    else:
        d["value"] = value
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
            item("without data (physics alone)", "sin datos (física sola)", value=f"{cs.get('naive_vs_std', 0)*100:.0f}%: unrecoverable"),
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
            item(f"c at (x=0.5, t=0.25)", "c en (x=0.5, t=0.25)", value=f"{mu:.3f} ± {2*sg:.3f} (2σ)"),
            item(f"P(c > {thr}) at that point", f"P(c > {thr}) en ese punto", value=p_str),
            item("least-trusted spot (largest σ)", "punto menos confiable (mayor σ)", value=f"σ = {float(std[sij]):.4f} at (x={float(xs[sij[0]]):.2f}, t={float(ts[sij[1]]):.2f})"),
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
            item("a 0.01 rad twin leaves the same tube at", "un gemelo a 0.01 rad deja el mismo tubo en", value=f"{fmt(twin_leave)} s" if twin_leave else "within the window: no"),
        ],
    )

    # ---- mine-thickener-settling: mudline arrival + final front height per R ----
    m = man("mine-thickener-settling")
    vals_settle, vals_bed = {}, {}
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
        vals_bed[v["id"]] = f"z = {front[-1]:.2f}"
    QUESTIONS["mine-thickener-settling"] = set_estimate(
        "mine-thickener-settling",
        "How long until the mudline settles below mid-height, and where does the front end? (thickener sizing)",
        "¿Cuánto tarda la línea de lodo en bajar de media altura, y dónde termina el frente? (dimensionamiento de espesadores)",
        "Sizing thickeners from batch settling curves IS this front-tracking (the Kynch construction); one parametric net carries the whole descent-rate family as a design chart.",
        "Dimensionar espesadores desde curvas de sedimentación ES este seguimiento de frente (construcción de Kynch); una red paramétrica lleva toda la familia de tasas como carta de diseño.",
        [
            item("mudline passes z=0.5 at", "la línea de lodo cruza z=0.5 en", values=vals_settle),
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
        items_flot.append(item(f"k = {kv:.2g}: recovery at t=1, time to 90%", f"k = {kv:.2g}: recuperación en t=1, tiempo a 90%", value=f"{R1:.1f}%, {t90s}"))
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
            item("mean base cA shift over the window", "cambio medio de cA en la base", value=f"{base[0]:.2f} to {base[-1]:.2f}"),
        ],
    )

    # ---- mine-comminution-pbe: passing fraction below s*=0.3 at t=1 per g ----
    m = man("mine-comminution-pbe")
    vals_pass = {}
    s_star = 0.3
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        ss, ts_ = np.array(t["axes"]["s"]), np.array(t["axes"]["t"])
        n = np.array(t["fields"]["n"])  # [is][it]
        below = ss <= s_star
        num = np.trapezoid(n[below, -1], ss[below]); den = np.trapezoid(n[:, -1], ss)
        vals_pass[v["id"]] = f"{num/den*100:.1f}%" if den > 0 else "n/a"
    QUESTIONS["mine-comminution-pbe"] = set_estimate(
        "mine-comminution-pbe",
        "What fraction of the charge passes size s*=0.3 after grinding? (mill sizing, P80-style)",
        "¿Qué fracción de la carga pasa el tamaño s*=0.3 tras la molienda? (dimensionamiento de molinos, estilo P80)",
        "Passing fractions are integrals of the size distribution the net carries for EVERY grind rate: the operating chart in one object.",
        "Las fracciones pasantes son integrales de la distribución que la red lleva para CADA tasa de molienda: la carta de operación en un objeto.",
        [item("passing fraction < 0.3 at t=1", "fracción pasante < 0.3 en t=1", values=vals_pass)],
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
            item("peak concentration there (within t ≤ 1)", "concentración pico allí (dentro de t ≤ 1)", value=f"{pk:.3f} at t = {tpk:.2f}"),
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
    vals_th = {}
    for v in m["variants"]:
        t = trace_of(m, v["id"])
        xs, ts = np.array(t["axes"]["x"]), np.array(t["axes"]["t"])
        u = np.array(t["fields"]["u"])
        ic = int(np.argmin(np.abs(xs - 0.5)))
        th = crossing_time(ts, u[ic, :], 0.5 * float(u[ic, 0]), rising=False)
        a = float(v["params"].get("alpha", 0))
        exact = np.log(2) / (a * np.pi ** 2) if a else None
        vals_th[v["id"]] = f"t = {fmt(th)} (exact {exact:.3f})" if (th is not None and exact and exact <= ts[-1]) else (f"t = {fmt(th)}" if th else "not in window")
    QUESTIONS["bench-heat1d"] = set_estimate(
        "bench-heat1d",
        "How long until the centerline cools to half its initial temperature? (quench / cooling design)",
        "¿Cuánto tarda el centro en enfriarse a la mitad de su temperatura inicial? (diseño de enfriamiento)",
        "One parametric net answers the cooling time for EVERY diffusivity: cross-checked here against the closed form ln 2/(απ²).",
        "Una red paramétrica responde el tiempo de enfriamiento para CADA difusividad: verificado aquí contra la forma cerrada ln 2/(απ²).",
        [item("centerline half-cooling time", "tiempo de medio enfriamiento del centro", values=vals_th)],
    )

    # ---- bench-burgers1d: front arrival at x*=0 per nu ----
    m = man("bench-burgers1d")
    vals_arr = {}
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
    QUESTIONS["bench-burgers1d"] = set_estimate(
        "bench-burgers1d",
        "When does the shock front arrive at the checkpoint x=0? (front-arrival estimation)",
        "¿Cuándo llega el frente de choque al punto x=0? (estimación de llegada de frentes)",
        "Arrival times are read off the tracked front; the viscosity family shares one net (the exact arrival is x*/s = 0.8 for every ν).",
        "Los tiempos de llegada se leen del frente rastreado; la familia de viscosidades comparte una red (la llegada exacta es x*/s = 0.8 para todo ν).",
        [item("front (u=0.5) reaches x=0 at", "el frente (u=0.5) llega a x=0 en", values=vals_arr)],
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

    # ---- patch index.json with the questions ----
    idx = json.loads((MANIFESTS / "index.json").read_text(encoding="utf-8"))
    for c in idx.get("cases", []):
        q = QUESTIONS.get(c["case_id"])
        if q:
            c["question_en"], c["question_es"] = q
    (MANIFESTS / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"index.json patched with {sum(1 for c in idx['cases'] if 'question_en' in c)} questions")


if __name__ == "__main__":
    bake_all()
