# PINN-Lab App UX redesign proposal (2026-07-09)

Designed FOR PINN-Lab (not cloned from another app). Lidar3D only signalled "use full width, do not waste
space". This proposal is driven by the actual content and the owner's direction. Validate before implementing.

## 1. Content + menu inventory (what actually exists)

**Global menus**
- Top nav: App + 5 doc pages (Introduction, Methodology, Implementation, Experiments, Benchmark).
- Header actions: GitHub, personal, portfolio, architecture modal (ⓘ), language, theme.

**App navigation (today)**
- 20 cases across 5 domains; a Highlights list (featured + the REAL-data case); domain chips; a case list;
  per-case regime/variant chips; a 4-way view switch (Field / Live / Charts / Context).

**Per-case content**
- Governing equation (KaTeX, variable width; Navier-Stokes is wide), method, engine, L2, ONNX parity,
  expected band, regime notes.

**The 8 visualization types (the heart of the app)**
| View kit | What it shows | Interactivity today |
|---|---|---|
| Field / Heatmap | space-time heatmap + colorbar + 2 side line-cuts + play | probe (crosshair+readout), play; side graphs are STATIC SVG |
| Vector (navier) | RK4 streamlines + quiver + scalar background | hover readout; background chips |
| Inverse (heat2d) | recovered k + true k* + error + measured T + sensor dots | hover readout |
| UQ (bpinn) | mean +- sigma band | sigma slider |
| Trajectory (pendulum) | pendulum canvas + phase portrait + butterfly + angles-vs-t | play; the 3 plots are STATIC SVG |
| Live | onnx re-eval + param/zoom/resolution sliders | live compute, zoom/pan |
| Charts | regime comparison bars (relative-L2) | click a row to load that regime |
| Context | deep bilingual prose + equations + refs | scroll |

## 2. What is wrong now (owner-reported + my QA)

1. Content was boxed in a narrow centred column while header/footer were full width (fixed in 0.19.000, but...).
2. The 0.19.000 3-column shell put the equation in a 320px right rail -> **the equation is cut off (ugly)**.
3. Highlights + Domain are expanded lists -> **waste vertical space**.
4. The field probe was regressed to **pin-only**; it lost the follow-the-cursor mode.
5. The side/trace graphs are **static plain SVG**: no rich hover, no snapshot, no focus/zoom, no detail popup.

## 3. Proposed design (fluid, viz-first, full-width)

Three stacked full-width zones, no side rails. The visualization is the star; everything else is compact context.

```
+-------------------------------------------------------------------------------+
| HEADER (full width)                                                           |
+-------------------------------------------------------------------------------+
| COMMAND BAR:  [Domain v][Case v (grouped, * featured)]  [Regime: chip chip]   |
|               [View: Field | Live | Charts | Context]      ...  [snapshot][focus][detail] |
+-------------------------------------------------------------------------------+
| CONTEXT STRIP:  f: u_t = d u_xx + 5(u - u^3)     method - L2 - ONNX     [band v]|
+-------------------------------------------------------------------------------+
| STAGE (dominant, fills remaining height, FULL WIDTH):                         |
|    the active view - big, interactive                                        |
+-------------------------------------------------------------------------------+
| FOOTER (full width)                                                          |
+-------------------------------------------------------------------------------+
```

### 3a. Navigation (fluid)
- **Case = one compact dropdown, grouped by domain (optgroups)**, featured/REAL cases starred. Replaces the
  Highlights list + Domain chips + Case list (all the space-hungry parts) with 1 (or 2) dropdowns.
  Option: keep a small separate **Domain dropdown** to filter, then a **Case dropdown** of that domain. (Owner asked
  for Domain-as-dropdown; both work, pick in validation.)
- **Regime** = inline chips (few per case).
- **View** = a segmented control (Field | Live | Charts | Context).
- Switching a case/regime/view updates only the context strip + stage; the command bar stays put (no reflow).
- Deep-linkable hash (`#/case/regime/view`) so a state is shareable (nice-to-have).

### 3b. Context strip (equation NEVER cut)
- The governing equation renders in a **full-width** strip, so even Navier-Stokes fits; if a future equation is
  huge, its own box scrolls horizontally rather than clipping.
- method - engine - L2 - ONNX parity inline; the expected-band sentence is a click-to-expand disclosure.

### 3c. Interactive visualizations (the big one: "no static plain graphs")
A shared, reusable interactive-plot component used by every kit, so all graphs get the same power:
- **Rich hover tooltip**: value + coordinates + relevant derived info per plot type, e.g.
  - Field side-cut: `u = ...`, `(x,t) = ...`, `|u - u0| = ...` (deviation from the initial state).
  - Phase portrait: `(θ1, θ2)` at the hovered t, plus the time.
  - Charts: variant label + L2 + rank + delta vs best.
- **Snapshot button** (per viz): export the current canvas/SVG to PNG (for slides/papers).
- **Change focus**: click a sub-plot to promote it to the focus (enlarge), or zoom/pan on the plot; the field map
  already zooms in Live, extend the pattern to the replay plots.
- **Detail popup**: click a graph -> a modal with the enlarged plot, full axes/ticks, extra series, and export.
- **Both-mode probe (field)**: released = the two side graphs FOLLOW the cursor live; single click = pin/fix;
  double click = release back to follow. (Restores the mode I regressed.)
- Everything stays paused-by-default / no compute bombs (existing rule).

### 3d. Per-kit application
- Field/Heatmap: both-mode probe + interactive side-cuts (hover/snapshot/detail) + play.
- Trajectory: the phase portrait + butterfly + angles-vs-t become interactive (hover the curves, snapshot, focus).
- Vector: hover already rich; add snapshot + a detail popup with a larger streamline plot.
- Inverse: hover shows k, k*, |k-k*| at the point across all four panels at once; snapshot; detail popup.
- UQ: hover shows mean and +-sigma and the sample; snapshot.
- Charts: interactive bars (hover = full metrics; click = load regime); snapshot.

### 3e. Fluidity + responsiveness
- Command bar wraps on narrow screens; the stage stacks graphs vertically < 900px.
- Smooth view/regime transitions (no full remount flfor the shell; only the stage swaps).
- Keyboard: left/right to step cases, up/down to step regimes (nice-to-have).

## 4. Suggested implementation phases (each a patch, screenshot-QA'd, issue-tracked)
1. Layout: 3 stacked zones (command bar + context strip + full-width stage); equation uncut; dropdown nav.
2. Field probe both-modes.
3. Shared InteractivePlot (rich hover + snapshot) applied to the field side-cuts + trajectory plots.
4. Detail popup + focus/zoom on all plots.
5. Charts + Inverse + UQ + Vector enrichments.

## 5. Open choices for validation
- **A.** Case nav: one grouped dropdown, OR Domain dropdown + Case dropdown? (leaning: Domain + Case dropdowns.)
- **B.** Snapshot scope: per-graph PNG only, or also a "copy the whole view" button?
- **C.** Detail popup: a modal, or an expand-in-place (the plot grows and pushes the others)?
- **D.** Phase 1 (layout) first and deploy, then enrich graphs incrementally? (leaning: yes, one phase per patch.)
