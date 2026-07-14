# THE REAL REVIEW AND PLAN (2026-07-15)

Owner order: "first perform a real review of all; my messages are only suggestions; evaluate the real content,
state and flow; define a real plan to improve and fix all; then implement deeply until completion, deferring
nothing." This document is that review and that plan. Every number below was MEASURED on the live app
(v0.24.001) at 1366x768; every content judgment comes from reading the actual rendered pages.

## PART 1: THE REVIEW

### 1.1 The fit audit (measured: every tab of every case, stage overflow in px at 1366x768)

The no-scroll principle is violated in essentially EVERY view of EVERY case:

- Results: 513-1499px hidden below the fold (all 21 cases; median ~1150). The narrative I placed in the
  stage pushes the answer's graph off-screen.
- Context: 656-2115px hidden (navier 2115). Prose is capped at 760px width inside a ~990px stage (dead
  right half) AND still scrolls for multiple screens.
- Field: 0-933px hidden. The field canvas is HARD-CAPPED at ~458px (a leftover `max-width: 560px` on
  `canvas.field`) regardless of available stage: the graphs are simultaneously TOO SMALL and overflowing.
- Live: up to 596px hidden. For the two replay-only cases (pendulum, darcy) the Live tab renders NO
  visualization at all (nviz=0): a nearly empty tab.
- Compare: 317-740px hidden with panels at 238x238 (thumbnails) or a wrapped second row that falls below
  the fold: the owner's "two rows of tiny graphs" is exact.
- Training: 542px hidden (the second panel row + curve below the fold).
- Diagnostics: 0-806px hidden; charts fixed at 720x339 in a wider stage.
- Charts: renders ZERO visualization elements (bars are plain divs; single-variant shows tiles): the
  weakest tab, "only a chart without any more info", currently not even a chart.

### 1.2 Layout and structure judgments

- L1 The story stepper: an invented narrative device occupying a permanent top band; its blurb row appears
  only for the 9 story cases, so switching to any other case CHANGES the page structure (instability).
  Owner: garbage, never required. Verdict: remove; the material itself belongs in the doc pages.
- L2 The case list: all 21 cards always visible in the rail. Selection is a one-action task; the list
  consumes exactly the space the rail's real jobs need (context + parameter control) and scrolls.
- L3 Controls duplicated: the regime chips exist twice on one screen (strip above tabs + inside the answer
  card). No point. Per the rail's purpose (choose case / give context / change params), the regime control
  belongs in the rail, once.
- L4 The equation: appears in multiple places (Results card, Context top), never in the one place it
  belongs: a full-width, always-visible strip above the tabs: it is the case's identity, invariant
  across views.
- L5 Nothing scales: canvases have fixed pixel caps, charts fixed viewBoxes; on a 1920px screen the same
  tiny graphics float in emptiness. The stage must define the size budget and every viz must fill it.
- L6 The header nav wraps to two lines at 1366px (Benchmark drops); the brand wraps too.

### 1.3 Content judgments

- C1 The five doc pages are thin and disconnected (measured words: Introduction 1367 / Methodology 3081 /
  Implementation 176 / Experiments 503 / Benchmark 427). Implementation is effectively EMPTY: the page that
  should tell how this product is actually built. Experiments lists categories but narrates no experiment.
  Benchmark shows a table with zero interpretation. There is no MIND-FLOW: no page hands off to the next,
  no arc from "why PDEs" to "how built" to "what was run" to "what the numbers mean".
- C2 The per-case investigation content (question, situation, calculates, know/seek, assumptions, answer,
  verdict: results.ts + scenarios.ts, all 21 EN/ES) is genuinely good material: but it is in the WRONG
  PLACE (crammed into the Results stage, displacing the results themselves).
- C3 The Results tab is not results-focused: a visitor must scroll past four narrative sections to reach
  the graph. Owner rule: Results = the result, summary form, clean, motivating, zero scroll.
- C4 Charts tab content is the weakest surface in the product (1.1). Needs to be a real analysis view or die.
- C5 Case Context deep pages (registry) are strong prose but presented as one endless scroll at 60% width.

### 1.4 Flow judgments

- F1 Entry: lands on the first case's Results: acceptable once the stepper noise is gone.
- F2 In-App flow: select domain -> select case -> read the case -> walk tabs: sound IF the rail carries
  selection+context+params and the tabs each fit the screen.
- F3 Cross-flow: no path from a case to the method family that explains it (Methodology) or to its
  benchmark row; doc pages do not link into the App's cases; the pages have no forward links to continue
  reading. The site is five islands plus an App.
- F4 Search works but its transient result list is the only list left: correct pattern.

## PART 2: THE PLAN

### 2.1 The layout contract (the budget at 1366x768; scales up with the viewport)

Header (56, single line: tighten nav gaps, brand no-wrap) / workbench = the rest (~700):
- LEFT RAIL (320px, FIXED structure for every case, nothing scrolls except interior of the narrative box):
  search / Domain dropdown / CASE dropdown (never a list) / the selected case's CARD (name + question) /
  WHAT PINS THE SOLUTION chips (compact context) / REGIME control (the case's parameter presets, ONCE).
- RIGHT AREA: EQUATION STRIP (full width, always visible, ~60px, internal x-scroll for long equations) /
  TABS / STAGE (fills all remaining height; ~985x500 at 1366x768).
- THE STAGE RULE (owner, universal): every tab fits the stage at full view. No vertical scroll in any tab.
  If a view has more to show than fits, it becomes another tab or an in-tab sub-view. Every visualization
  SIZES ITSELF to the stage (remove all fixed pixel caps; kits receive the box, not the box the kit).

### 2.2 Tab set and per-tab layouts (order fixed by owner: Results, Context, then evidence)

1. RESULTS: pure result, motivating: LEFT ~40%: THE ANSWER (plain sentence) + value items + VERDICT +
   evidence links. RIGHT ~60%: the case's KEY GRAPH filling the height with its markers. Nothing else.
2. CONTEXT: sub-navigated, full width, no scroll: sub-sections rendered as an in-tab section switcher:
   [Summary | Problem | Formalization | Method | Scope | How to read]. "Summary" = the four narrative
   blocks (situation / what the model calculates / what we know and seek / how the PINN helps +
   assumptions) as a 2x2 card grid. The rest auto-split from the existing deep Context pages' h3 sections
   (they already have exactly these headings). Full width prose (multi-column where long).
3. FIELD: the interactive field sized to stage height + the two profile cuts stacked in a right column +
   transport on top. Wheel-zoom/pin kept.
4. LIVE: sliders in a compact left strip of the stage, the live field filling the rest. Replay-only cases:
   an honest, designed notice block (why replay, what to use instead): never a near-empty tab.
5. COMPARE: a FITTED GRID computed from the stage box: main lanes large in row 1, error maps as row 2
   sized so the WHOLE grid fits (e.g. allencahn 3@~300px + 2@~190px); summary chips inline. Never
   thumbnails, never below the fold.
6. TRAINING: the two checkpoint fields side by side + the L2-vs-iteration chart in a right column; one screen.
7. DIAGNOSTICS: charts side by side, scaled to the stage; 3+ charts -> in-tab pager (same switcher as Context).
8. CHARTS -> renamed content: THE REGIME ANALYSIS: multi-variant: one full-width interactive chart
   (error per regime + per-regime estimate values below it as a labeled row, click-to-load, judgment
   threshold line) + a one-line reading. Single-variant: the explained metric tiles (kept) + the verdict.
   Real information, fits, interactive.

### 2.3 The content plan (the five pages become one arc; EN/ES; from the persisted research, not memory)

Nav order = reading order, each page ends with a "continue" hand-off:
- INTRODUCTION (rework, ~1200w): why PDEs matter -> what a PINN attempts -> when it is (not) the tool ->
  how this catalogue works (cases as questions with computed answers) -> start with a case / read the
  methods. The "story in 9 chapters" panel REMOVED from here.
- METHODOLOGY (extend, keep the strong ladder): OPEN with the win/lose arc (the story material lands here,
  in prose where it belongs: forward-easy, forward-hard, inverse+data, real data, operators, UQ, chaos:
  each pointing at its demonstrating case) -> the method families (existing) -> estimators-in-the-wild
  (existing) -> references.
- IMPLEMENTATION (REBUILD, 176w -> deep): the two-world architecture (offline pipeline vs static web +
  the artifact contract + lane gate); the pipeline stages; training recipes per family; the numerical
  anchors (analytic/MMS/FD with their stability checks); the test discipline (sandboxed suite: why);
  ENGINEERING LESSONS, honestly told from the wip record: the navier FDM divergence, the allencahn
  metastable trap, the hidden-velocity steady-flow fix, the compute-bomb fix; the architecture diagrams
  (from the existing i-modal SVGs, embedded). Source: docs/ wiki + wip logs. ~1600w.
- EXPERIMENTS (REBUILD, 503w -> deep): what was actually RUN, narrated: fair-budget protocol (and the
  wave1d no-contrast honesty case); the ladder runs per category with their numbers; the identifiability
  sweep; the real-data holdout (soil); the dye holdout (flagship); training-dynamics captures. Each block
  links its case + its Benchmark row. ~1200w.
- BENCHMARK (extend, 427w -> interpretive): the table (kept) + "how to read these numbers" (what rel-L2
  means per anchor class, what parity is, why some cases are honestly at 10-20%) + per-category takeaways
  + the limits table. ~800w.
- CROSS-LINKS: each case's Context links its method family + Benchmark row; Methodology families link
  their cases (exists) + Experiments blocks; the App footer verdict links Benchmark.

### 2.4 QA gates (upgraded so this cannot silently regress)

- Validator v2: for EVERY case x EVERY tab: assert stage overflow <= 2px (the owner rule as a hard gate),
  assert the biggest visualization covers >= 35% of the stage area on viz tabs, tab order, rail
  non-overflow, single-line header; both themes; 1366x768 AND 1920x1080.
- Doc pages: word-count floors (Implementation/Experiments >= 1000, Benchmark >= 700) + heading structure
  asserts + the hand-off link present.
- Screenshot review (human-judgment pass) of every tab type before release; pytest suite green.

### 2.5 Execution order

E1 Shell: remove stepper/story from the App; rail = search + 2 dropdowns + case card + pins + regime;
   equation strip; single-line header; tab set (Results | Context | Field | Live | ...).
E2 The stage sizing engine: kits fill a measured box; all fixed caps removed.
E3 Per-tab layouts to the contract (Results, Context sub-nav, Field, Live, Compare grid, Training,
   Diagnostics, Regime analysis).
E4 The five-page content arc (Introduction rework, Methodology arc, Implementation + Experiments rebuilds,
   Benchmark interpretation).
E5 Validator v2 + full QA (fit audit all-green at both viewports, both themes) + pytest + human screenshot
   pass.
E6 Release v0.25.000; close #49; CAOS_MANAGE close-out (products/plans/heartbeat).

No deferrals: every item above ships in this arc.

## EXECUTION STATUS: COMPLETE (2026-07-15, v0.26.000 LIVE)

All of E1-E6 shipped and verified on prod (pinnlab.fasl-work.com), across releases v0.25.000 -> v0.26.000:
- E1 shell: story stepper removed; rail = search + Domain/Case dropdowns + case card + pins + regime(once);
  equation full-width strip above the tabs; single-line header. DONE (v0.25.000).
- E2 sizing engine: useFitBox contain-fit; all fixed pixel caps removed; every map/chart sizes to the stage box.
  DONE.
- E3 per-tab layouts: Results (answer + rich interactive kit, after correcting an interim single-image
  regression), Context (SectionPager: Summary cards + auto-split deep prose full-width), Field (robust flex),
  Live (compact + absent for replay-only lane), Compare (fitted row, Fields/Errors/Evolution sub-views),
  Training, Diagnostics (pager), Regimes, HeatmapKit chip-row fit. DONE (v0.25.000-002, .26.000).
- E4 content arc: Introduction destaled + real tab structure; Methodology win/lose scope + estimators-in-the-wild;
  Implementation two-worlds + Engineering-lessons; Experiments what-was-actually-run; Benchmark
  how-to-read-these-numbers; hand-offs across the five pages. DONE (v0.25.001, .26.000).
- E5 fit gate: tools/visual-verify/pl-fit-audit.mjs committed; PASSES 508 tab-checks (21 x every tab x 2
  viewports x 2 themes) on prod: no scroll, Results-first, rail non-overflow, viz present. pytest 51 green.
  DONE (v0.26.000).
- E6 release + close-out: v0.26.000 tagged + LIVE + prod-verified; issue #49 closed; product doc + plans +
  heartbeat updated. DONE.
- Bonus per owner review during execution: header/footer font aligned to the ADR-0016 system stack (v0.25.003).
