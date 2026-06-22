export const meta = {
  name: 'pinnlab-migrate-remaining-cases',
  description: 'Design the workbench migration (honest parametrization + deep bilingual Context + bake recipe) for the 12 remaining PINN-Lab cases, in parallel. Agents write the Context .tsx and a design note to disk; they do NOT bake or touch registry/index.',
  phases: [{ title: 'Design', detail: 'one agent per remaining case' }],
};

const REPO = 'd:/_Repos/Research_Caos/CAOS_PINNLAB';

// case_id, python module file, React component name, Context .tsx path
const CASES = [
  ['ind-helmholtz', 'ind_helmholtz.py', 'HelmholtzContext'],
  ['bench-navier-cavity', 'bench_navier_cavity.py', 'NavierCavityContext'],
  ['bench-darcy-operator', 'bench_darcy_operator.py', 'DarcyOperatorContext'],
  ['ind-heat2d-inverse', 'ind_heat2d_inverse.py', 'Heat2dInverseContext'],
  ['env-soil-heat-real', 'env_soil_heat_real.py', 'SoilHeatRealContext'],
  ['poll-source-uq-bpinn', 'poll_source_uq_bpinn.py', 'SourceUqBpinnContext'],
  ['poll-soil-barrier', 'poll_soil_barrier.py', 'SoilBarrierContext'],
  ['poll-tailings-seepage', 'poll_tailings_seepage.py', 'TailingsSeepageContext'],
  ['mine-heap-leach-rt', 'mine_heap_leach_rt.py', 'HeapLeachContext'],
  ['mine-thickener-settling', 'mine_thickener_settling.py', 'ThickenerContext'],
  ['mine-comminution-pbe', 'mine_comminution_pbe.py', 'ComminutionContext'],
  ['ctrl-zero-source', 'ctrl_zero_source.py', 'ZeroSourceContext'],
];

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['case_id', 'decision', 'n_variants', 'analytic_anchor', 'bake_recipe', 'context_file', 'component_name', 'design_note', 'risks'],
  properties: {
    case_id: { type: 'string' },
    decision: { type: 'string', enum: ['parametric', 'single', 'discrete'] },
    param_key: { type: 'string', description: 'the swept network-input knob, or "" if single/none' },
    n_variants: { type: 'integer' },
    analytic_anchor: { type: 'string', description: 'the exact u* / reference and a one-line proof it solves the PDE (or "dataset/FEM reference" if no closed form)' },
    bake_recipe: { type: 'string', description: 'net layers, activation, adam iters, RAR? lbfgs?, expected relative-L2 band' },
    context_file: { type: 'string', description: 'the path written' },
    component_name: { type: 'string' },
    design_note: { type: 'string', description: 'the path of the wip/case-designs/<id>.md written' },
    risks: { type: 'string' },
  },
};

const sharedRules = `
PROJECT: PINN-Lab — a catalogue of PDE cases, each a SimLab-style WORKBENCH. You are migrating ONE case to the
workbench standard. The web shell/components/pipeline already exist; you produce (a) the deep bilingual Context .tsx
and (b) a design note with the exact case .py edits + bake recipe. You do NOT run any bake, you do NOT edit
registry.tsx or index.json, you do NOT edit the case .py directly (leave that to the orchestrator).

READ FIRST (all under ${REPO}):
- The playbook: wip/workbench-migration-state.md (the proven recipe + bake-speed lessons + per-case notes).
- base spec: data-pipeline/pinnlab/cases/base.py (CaseSpec, ParamSpec, Variant, field_axes/param_specs/axes rules).
- param grid: data-pipeline/pinnlab/model/analytic.py (how param_grid builds the field + fills param axes).
- PARAMETRIC exemplar: data-pipeline/pinnlab/cases/bench_burgers1d.py + frontend/src/content/cases/Burgers1dContext.tsx.
- SINGLE-variant exemplar: data-pipeline/pinnlab/cases/bench_allencahn.py + frontend/src/content/cases/AllenCahnContext.tsx.
- TIME-SCRUBBER exemplar (x,y,t): data-pipeline/pinnlab/cases/poll_ocean_transport.py + OceanTransportContext.tsx.
- YOUR case .py (read it fully).

DECIDE THE HONEST MIGRATION (this is the crux — be rigorous and conservative):
- PARAMETRIC (>=6 variants, the knob is a NETWORK INPUT in inputs + param_specs, NOT in field_axes) ONLY IF there is a
  ROCK-SOLID closed-form / MMS / analytically-verifiable family in a physical knob. You MUST prove the anchor solves
  the PDE for ALL parameter values by substitution (show it in the design note). If you cannot prove it cleanly, do
  NOT go parametric.
- TIME-SCRUBBER: for an (x,y,t) field with a clean exact solution, set field_axes=(x,y), the swept param = t, variants
  = time snapshots (like ocean). Live becomes a time scrubber.
- SINGLE honest benchmark variant: for stiff/dataset-anchored/inverse/operator/real-data cases with NO meaningful
  closed-form parametric family. This is CORRECT and expected for several of these cases — NEVER fabricate regimes to
  hit a chip count (ADR-0016 §9.A: faking regimes is the "toy" failure to avoid). Honesty over chip count.
- DISCRETE: if the natural variants are a small discrete set (e.g. FNO held-out test samples, B-PINN noise levels),
  use those — but only if each is genuinely meaningful.
ADVERSARIALLY CHECK YOURSELF: would this parametrization actually train and validate? Is the anchor truly exact? If in
doubt, choose the more conservative (single) option and say why.

WRITE (to disk, both files):
1) The Context: frontend/src/content/cases/<ComponentName>.tsx — a deep BILINGUAL (es + en) React component
   "export function <ComponentName>({ lang }: { lang: 'en' | 'es' })", importing
   'import { Equation, InlineMath } from "../../components/Equation";', matching the EXEMPLAR depth and the EXACT
   bilingual section order: The problem -> Components & variables -> Formalization (KaTeX, display + inline) ->
   [The method, if distinctive] -> Scope & assumptions -> What each variant shows (or "What the benchmark shows" for
   single) -> How to read & USE the viz. Graduate level, honest labels (synthetic / illustrative-synthetic /
   validated-real), real DOI-backed framing where relevant. ZERO internal repo paths, file names, or slugs in any
   rendered string. Mirror the exemplar's KaTeX usage (String.raw).
2) The design note: wip/case-designs/<case-id>.md — the decision + rationale, the analytic anchor WITH the
   substitution proof, the EXACT case .py edits (a clear code block the orchestrator can apply: field_axes,
   param_specs, variants(), and any analytic/build/output-transform changes — preserve the case's existing method),
   the bake recipe (layers, activation, adam, RAR? lbfgs?, expected relative-L2 band), the registry line
   ("<case-id>": <ComponentName>), and risks/fallback.

CONSTRAINTS: English code/comments; bilingual UI strings only. Keep the case's existing SOTA method/engine. Do not
weaken honesty. Your final message IS the structured return (the schema) — keep it short; the real work is the two
files you wrote.
`;

phase('Design');
const results = await parallel(CASES.map(([id, pyfile, comp]) => () =>
  agent(
    `${sharedRules}\n\nYOUR CASE: case_id="${id}", python file="data-pipeline/pinnlab/cases/${pyfile}", ` +
    `Context component name="${comp}", Context file="frontend/src/content/cases/${comp}.tsx", ` +
    `design note="wip/case-designs/${id}.md". Read the files, decide the honest migration, write BOTH files, return the schema.`,
    { label: `design:${id}`, phase: 'Design', schema: SCHEMA }
  )
));

const ok = results.filter(Boolean);
log(`designed ${ok.length}/${CASES.length} cases`);
return {
  designed: ok.length,
  total: CASES.length,
  summary: ok.map(r => ({ case: r.case_id, decision: r.decision, variants: r.n_variants, param: r.param_key || '-', risk: r.risks?.slice(0, 80) })),
};
