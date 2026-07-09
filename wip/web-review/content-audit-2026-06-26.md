# Content/docs depth audit (2026-06-26)

Inline audit (R2). The App reorganization (0.15.0) is done + live. Case-level content has NO blanks: all 20 cases
have a deep bilingual Context (124-198 lines each) AND a `docs/cases/<id>.md`. The gaps are DEPTH + FIGURES.

## Gaps, prioritized

1. **SVG figures — CORRECTED.** The ⓘ Architecture modal DOES have hand-authored themed SVGs (external, fetched
   from `public/svg/tech/`, inlined so CSS tokens resolve): 6 files (`system-overview`, `web-architecture`,
   `offline-pipeline`, `train-onnx-web`, `lane-gate`, `method-matrix`) across 7 tabs. ADR-0058 is **met**. The grep
   missed them because they are external assets, not inline `<svg>`.
   - **Real gaps:** (a) those SVGs **predate** the 7 view kits + the new cases (double-pendulum) + the App reorg —
     `method-matrix.svg` and `system-overview.svg` are likely **stale** (no ode-dynamical/vector-flow/UQ/inverse kits,
     20-not-19 cases); refresh them. (b) No **view-kit architecture** diagram (ADR-0063: `system_type→view_kit`) —
     add a modal tab + SVG. (c) The content **pages/Contexts** carry no inline figures, but the modal + the live viz
     cover this, so this is LOW priority (add 1-2 only where a schematic genuinely helps, e.g. the double-pendulum
     leave-time / Lyapunov horizon, or the classical→SOTA→novel ladder on Benchmark).

2. **Benchmark page thin (61 lines).** It's the honesty table; it should also carry the **classical → SOTA →
   candidate-novel** ladder + comparative results per category (from the SOTA research, in progress). 

3. **Experiments page (84 lines) already has rich per-category narratives** (good), but a few are slightly STALE vs
   the new kits (e.g. "time-scrubber" for ocean is now animated; "domain decomposition" framing for soil-barrier).
   Refresh the narratives to match the deployed kits + fold in the SOTA ladder.

4. **Introduction (86 lines)** could gain a "what a PINN is / where it wins vs a classical solver" honest primer +
   a small SVG.

5. Minor: `CATEGORY_INTRO` now exists in BOTH `contract.ts` (App one-liners) and `Experiments.tsx` (paragraphs) —
   intentional (different lengths), but keep them consistent.

## Method-family docs (docs/methods) — check vs the SOTA research
The new methods (ode-dynamical soft-IC IVP, streamlines/vector-flow presentation, UQ band, inverse overlay) may
need method-doc updates. Cross-check against the SOTA research report (wf deep-research, pending) before writing.

## Order
Wait for the SOTA research → then (C1) author the SVG figures (bake-free, big payoff), (C2) enrich Benchmark with the
SOTA ladder + results, (C3) refresh Experiments narratives + Introduction primer, (C4) method-doc top-ups. Then (N)
the new cases with bakes.
