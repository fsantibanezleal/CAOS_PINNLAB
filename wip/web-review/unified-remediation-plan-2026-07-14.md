# UNIFIED REMEDIATION PLAN (2026-07-14): from solver gallery to investigation instrument

Owner order: "review all, propose a plan unified with all the plans and disaggregated content, then implement
all." This plan SUPERSEDES and absorbs: ux-flow-structure-plan-2026-07-13.md, ux-redesign-2026-07-09.md,
dynamics-evaluation-2026-07-10.md (findings), fundamentals-review-2026-07-10.md (the five questions),
estimation-reframe-plan-2026-07-10.md (question->estimate, shipped parts kept), scenarios.ts (drafted content),
and the uncommitted structural edits (stepper, stage tabs, search, scenario card). One plan, one execution.

## 1. The review (owner's enumeration, verified against the live app 2026-07-14)

MEASURED on live (1500x950, allencahn): the left rail overflows its own height (scrollHeight 1017 vs 798
visible: 220px of controls hidden); the stage starts at y=404 (42% of the screen consumed before the
instrument); the stage gets 466px of 950. The owner's verdicts, each confirmed:

- R1 LAYOUT: no UX/UI design; vertical scroll "like a bad math book"; space wasted; rail overflows; the view
  switch lives at the BOTTOM of the scrolling rail, far from the stage it controls.
- R2 STRUCTURE: default view (Compare) drops the visitor into reference-vs-PINN fields with zero framing;
  the pieces are disconnected (no thread from question to evidence).
- R3 CONTENT: cases are "an ODE/PDE and a PINN side by side": a solver demo, not a case. The five questions
  (why is the PDE relevant / how does it arise from the mechanism / how does the PINN help and under what
  assumptions / what side information is required for validity / how good or bad is the result) are NOT the
  organizing structure of any case. Answer cards are insider jargon ("hard-constraint + RAR", "naive lane");
  estimates are naked code-speak numbers ("zero crossings of u(x,t=1): -0.494, 0.494") never marked on the
  field, with no units, no significance, no judgment.
- R4 INTERACTION: viz effectively non-interactive by the rubric (no zoom/brush/solo, no linked cursor);
  selection controls and their effects are never on screen together; NO coordinated operation (a regime
  change does not visibly ripple through all views); Charts view can be a single bar with a raw exponential.
- R5 QUALITY BAR: under conventions/product-quality-bar.md + interactive-visualization-rubric.md.

## 2. Owner decisions (fixed, applied exactly)

- D1: Tab order STARTS WITH RESULTS, THEN CONTEXT, then the rest. Results is the landing tab of every case.
- D2: Never mirror another app's layout; design for THIS app's content.
- D3: Real computed content only; no fabricated contrasts; judgments must derive from baked numbers.
- D4: No autoplay/compute bombs; animations paused by default, run-once, halt on hidden tab.

## 3. The design

### 3.1 The case is an INVESTIGATION; the RESULTS tab is its face (D1)
Every case's landing view is a designed single-screen RESULTS tab answering, in order, with zero unexplained
jargon:
  1. THE SITUATION: who needs what, with stakes (scenarios.ts, already drafted for all 21).
  2. WHAT THE MODEL CALCULATES: the mechanism in one sentence (conservation/closure -> this equation), what
     the axes/colors of the shown field MEAN physically.
  3. WHAT WE KNOW vs WHAT WE SEEK: the measurable information (constraint chips, re-worded plain) -> the
     target quantity.
  4. HOW THE PINN HELPS + THE DECLARED ASSUMPTIONS (known coefficients, steadiness, noise, budget).
  5. THE ANSWER IN PLAIN LANGUAGE: the estimate re-written as a sentence with units and meaning ("the two
     phase walls end at x = -0.49 and +0.49: they stayed symmetric and pinned"), its markers DRAWN ON the
     key visualization beside it, and THE VERDICT: how good/bad, derived from the baked numbers ("0.4% vs
     the spectral reference: indistinguishable at plot scale; trust wall positions to ~0.01").
  6. THE EVIDENCE INDEX: one line per remaining tab stating what it PROVES ("Compare: the naive PINN loses
     the walls entirely"), each a link.
Content source: a new per-case content module (results.ts) = plain-language answer sentence + verdict +
axis/color explanations + assumption list; verdicts computed from manifest numbers at build time where
possible (never hand-invented).

### 3.2 Layout: an instrument that fits the screen (R1)
- ONE viewport at 1080p: no page scroll on the App for the default tabs; internal panels scroll only
  themselves when content genuinely exceeds (e.g. long Context prose).
- Grid: header (48px) / story stepper (one compact row) / workbench = left rail (fixed, NON-overflowing:
  search + domain + case list, the ONLY scrolling element is the case list itself) | main column:
  investigation header strip (compact: question + verdict, expandable) -> TABS ON THE STAGE (Results |
  Context | Field | Live | Compare | Training | Diagnostics | Charts; conditional ones hidden) -> stage
  filling ALL remaining height.
- Regime chips move OUT of the rail to a single horizontal strip directly above the stage (control next to
  effect); params/sliders live inside the view that uses them (Live), visible without scrolling.
- The old pl-context (equation/pins/meta) dissolves: equation + pins move INTO Results and Context; the meta
  (method/engine/L2/parity) becomes a one-line footer strip of the stage.

### 3.3 Coordinated operation (R4)
A single shared per-case UI state: { variantId, probe (x,y), timeCursor }: every view reads and writes it.
- Change the regime ANYWHERE (chips, chart row, answer-card value chip) -> every view, readout, marker
  updates.
- The probe: hover/pin in any field is mirrored in every other field panel, in profile charts (cursor line)
  and in a persistent readout; double-click releases (existing behavior generalized).
- The time cursor: one transport bar drives Field animation frames, evolution frames, and the reconstructed
  panels in sync.
- Estimate markers: each case's QoI drawn on its fields (interface positions, arrival checkpoint, vortex
  core, sensor dots, front lines) with the value labeled: the number and the picture finally reference each
  other.

### 3.4 Interactive charts to the rubric (R4)
- Line charts (Training L2, Diagnostics profiles/curves, variant charts): crosshair + hover values (exists)
  PLUS x-wheel/drag zoom with reset, per-series click-to-solo / toggle via legend, and the shared time/probe
  cursor where applicable. Implemented on the existing hand-rolled SVG components (no new heavy deps; the
  rubric's Tier-A interactions, not its specific library).
- Heatmaps: wheel zoom + drag pan + reset button, shared across side-by-side panels (Compare panels zoom
  together); pinned value readout; colorbar hover-highlight.
- Charts tab: for multi-variant cases a real bar comparison with judgment lines; for single-variant cases it
  is REPLACED by the error-distribution summary (histogram + percentile readout + where-the-error-lives
  mini-map): never a one-bar chart again.

### 3.5 Jargon purge (R3)
Rule: no term appears in Results/AnswerCard/chips without an inline plain-language gloss. Sweep all 21 cases'
question/why/labels; move method jargon (RAR, hard constraints, PFNN, lanes) into Context where it is
explained, referencing the Methodology ladder.

## 4. Execution (implement all; vertical where content is per-case)

- S1 SHELL: no-scroll grid, stepper row, tabs-on-stage in D1 order, rail = selection only, regime strip,
  meta footer. (Structural edits already in the working tree are absorbed here.)
- S2 RESULTS TAB: the investigation view component + results.ts content for all 21 cases (vertical: one case
  fully written + marked + verdict before the next; flagship + story chapters first).
- S3 COORDINATION: shared case state (variant/probe/time), linked panels, estimate markers on fields.
- S4 CHART INTERACTIVITY: zoom/solo/crosshair upgrades + Charts-tab replacement.
- S5 SWEEP + QA: jargon purge, no-scroll audit at 1500x950 and 1366x768, first-visitor screenshot review of
  every case's Results tab (both themes), pl-validate-all extended to assert rail non-overflow + stage
  height + Results-first order.
- Release as v0.24.000 (with the already-baked flagship case) or v0.25.000 if the owner prefers the
  flagship separated. Issues: one per S-slice, closed on merge.

## 5. What is explicitly kept from the absorbed plans

- Estimation reframe (shipped v0.23.000): questions/estimates stay; their PRESENTATION is rewritten (3.1.5).
- Dynamics work (v0.20-0.22): animations, training replays, evolution frames: kept, wired to the shared
  time cursor.
- Fundamentals content: information budget, identifiability, constraint chips: kept, re-homed (chips into
  Results §3, budget into Context/Introduction).
- The 21st case (hidden velocity, trained, 16.4% swept recovery): ships with this work.

## 6. EXECUTION STATUS (2026-07-14, v0.24.000 LIVE)

- S1 DONE, S2 DONE, S3 DONE (first pass: 34 markers on 12 cases via FieldView/TimeEvolution/VectorField/
  HiddenFlow), S4 partial (single-regime Charts tab replaced). Shipped as v0.24.000 with the #48 flagship
  (PR #50); structural validator asserts Results-first/content/non-overflow/page-fit per case: 21x2 themes
  green ON PROD, 0 console errors; 51 contract tests green.
- Measured on prod: rail overflow 220px -> 0; stage 404/466px -> 247/616px; no page scroll at 1500x950.
- REMAINING (issue #49 open): line-chart zoom + per-series solo; heatmap wheel-zoom/pan; shared probe/time
  cursor across views; markers on SpatioTemporal/UQBand; Context jargon sweep; responsive breakpoints.
