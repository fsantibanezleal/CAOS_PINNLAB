# Stage-fill layout fix (2026-07-14) - v0.26.003 LIVE

Owner sent live screenshots (`Stupid_Bad_PINN_work/`) showing the App layout still broken after the v0.26.000
overhaul: heatmaps collapsing to a few pixels or floating tiny in an empty stage, one map overflowing off the
edge and covering the slider, Context Summary filling only the top half, a caption clipped mid-second-line, and
sub-tab labels truncated mid-word. The v0.26.000 fit engine had NOT actually made the maps fill.

## Root cause (one bug, several symptoms)

The fit hooks (`useFitBox`, `useGridFit`) attached their `ResizeObserver` inside a **deps-gated `useEffect`**.
Several kits render the measured element only AFTER their data loads async (behind `if(!data) return`). For a
**square** field the `aspect` dep never changes across that null->loaded transition, so the effect never re-ran
after the DOM mounted, so the observer never bound, so `fit` stayed `{0,0}` and the map collapsed to its native
canvas size (e.g. 63px). Multi-panel kits (InverseOverlay, HiddenFlow) additionally used a CSS `auto-fit` grid
with `aspect-ratio` panels that collapse to intrinsic size when no definite dimension exists.

## Fix

- **Callback ref** in both hooks: the observer binds the moment the node mounts, independent of deps.
- **`useGridFit`**: new deterministic multi-panel fit that picks the (cols,rows) MAXIMISING per-map area (1xN in
  the wide App stage, 2x2 / 3x2 in the narrow Results column). Applied to InverseOverlayKit + HiddenFlowKit.
- **SpatioTemporalKit**: fit ref now measures a map-only wrapper (colorbar excluded) with the real field aspect;
  no more overflow off the stage edge (was 63px collapse / earlier an overflow, now a contained 568px map).
- **Context Summary**: `grid-auto-rows:1fr` so the narrative cards fill the full stage height.
- **SectionPager**: sub-tab labels derived at the colon/word boundary, never truncated mid-word.
- **HeatmapKit**: space-time caption is one concise line (depth lives in Context), so it never wraps and clips.
- **Doc pages**: Introduction/Methodology(12 family tabs)/Experiments/Benchmark converted to ADR-0016 6 SubTabs
  (Implementation already had them). No content removed.

## The gate blind spot (the reason it recurred)

`pl-fit-audit.mjs` only asserted "no scroll (overflow)". A stage 60% empty with tiny floating maps PASSED. Added
an **UNDERFILL** check: sum painted viz area vs the area it should fill (`.pl-res-viz` for Results, else the
stage), flag map tabs (field/live/results) below 10%. On its first run it caught the **HFM flagship**
(`ind-hidden-velocity`) collapsed to 3-8% - which HiddenFlowKit had missed and no human screenshot pass caught.

## Verification

- 508-check fit gate PASS with the underfill check active: no scroll, no underfill, Results-first, viz present,
  21 cases x every tab x 1366x768 + 1920x1080 x dark+light.
- Live-prod screenshots (pinnlab.fasl-work.com, bundle index-VRj6d2hD.js contains 0.26.003): hidden-velocity
  Field = 5 maps @294px (37% fill), heap-leach Field = 568px (28%), heat2d Results = 2x2 @316px, both themes.

Shipped: v0.26.003 (develop -> PR #59 -> main -> tag -> Deploy Pages success -> live verified).
