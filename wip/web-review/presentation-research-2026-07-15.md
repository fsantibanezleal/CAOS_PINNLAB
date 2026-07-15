# Deep research: how the best interactive science explainers PRESENT results (2026-07-15)

Real deep-research run (deep-research workflow wf_4e9ff5c3-b70; 105 agents, 22 sources, 25 claims verified 3-vote
adversarial, 23 confirmed / 2 refuted). NOT from memory. This is the SOTA-of-presentation basis for the PINN-Lab
content+visualization rebuild (the owner's mandatory sequence: research -> plan -> implement).

## The verified findings (each with its primary source)

1. TIGHT COUPLING is the core mechanism. Keep the computed number + its governing equation + the live
   visualization + the prose PHYSICALLY ADJACENT, and communicate one idea through MULTIPLE LINKED
   REPRESENTATIONS that inform each other. (Tufte CSS "data, graphs and figures are kept with the text that
   discusses them"; Distill "single idea, multiple representations". https://edwardtufte.github.io/tufte-css/ ,
   https://distill.pub/2020/communicating-with-interactive-articles/ )
2. EQUATION <-> EXPLANATION via mouseover: hovering a term in the equation highlights the corresponding word in
   the prose (maps notation to meaning). Generalizes to equation<->plot. (Distill, same URL.)
3. PROSE -> IMMEDIATELY-FOLLOWING LIVE FIGURE rhythm: a sentence states a relationship, the very next element is
   an interactive control that updates the visual in real time; complex phenomena use LINKED SIDE-BY-SIDE panels
   (input left, output right). (Ciechanowski, https://ciechanow.ski/cameras-and-lenses/ )
4. MARTINI-GLASS narrative: lead the viewer through a guided linear story (text + annotations), THEN open
   interactive controls for free exploration; animating each component teaches the interactions.
   (Heer & Shneiderman, CACM 55(4) 2012, DOI 10.1145/2133806.2133821.)
5. HERO NUMBER: size-based hierarchy (primary value largest; target/gap/trend smaller). IMPORTANT SCOPE (2-1,
   medium): the "a number is meaningless without a benchmark" rule is BI-DASHBOARD doctrine and does NOT transfer
   to SCIENCE hero-numbers, where meaning comes from UNITS + PHYSICAL INTERPRETATION + the PAIRED EQUATION.
   (Tabular Editor KPI blog, corroborated; scoped by the science exemplars.)
6. DEEP MULTI-SECTION without scroll: TABS for a FEW LONG non-comparative sections (chunk into scannable panels,
   one at a time); ACCORDIONS for MANY SHORT sections. TABS ARE WRONG when the user must COMPARE/cross-reference
   sections simultaneously (taxes short-term memory). (NN/g, https://www.nngroup.com/articles/tabs-used-right/ ,
   https://www.nngroup.com/videos/tabs-vs-accordions/ )
7. PROGRESSIVE DISCLOSURE: Shneiderman "overview first, zoom and filter, details-on-demand". (Distill / Shneiderman 1996.)
8. THREE FIGURE TIERS: main-column (default), margin (ancillary), full-width (large/dense). (Tufte CSS.)
9. COLORMAPS: perceptually-uniform, monotonic-luminance, colour-blind-safe (viridis/plasma/inferno; Crameri
   scientific maps). RAINBOW/JET is a poor default "emphatically reviled". (Moreland
   https://www.kennethmoreland.com/color-advice/ ; Crameri Nat.Commun. 2020 DOI 10.1038/s41467-020-19160-7.)
10. INTERACTIVE READOUTS: on-hover pointer+tip tooltip (nearest datum) AND a CROSSHAIR whose x/y labels are
    anchored at the FRAME EDGES (atop the axes) so they never obscure the marks. (Observable Plot,
    https://observablehq.com/plot/features/interactions )
11. BRUSHING-AND-LINKING across COORDINATED VIEWS: select/hover in one plot highlights the same data in the
    others; better than one cluttered multi-dimensional plot. Grounded in Heer & Shneiderman's 12-task taxonomy.

REFUTED (do NOT do): tabs for sections that must be COMPARED (0-3); a global "pause all animations" control (0-3).

## What this VALIDATES in PINN-Lab today (keep)

- viridis colormap everywhere (finding 9): compliant.
- hover value read-out + crosshair on the field (finding 10): present; refine per below.
- Context as a TAB/section-switcher for the long, non-comparative deep sections (finding 6): correct pattern.
- the per-case linked cursor: field hover drives the profile cuts (finding 11): present within FieldView.
- the equation shown per case, the computed answer as a hero (findings 1, 5): present; tighten coupling.

## THE PLAN (grounded in the findings + the repo's real docs: 10 method-family docs + 22 case docs)

P1. COUPLE the case as ONE unit (finding 1,3): the Results tab already sits answer+viz side by side. Strengthen:
    the ANSWER hero tile shows the value with its UNIT and a one-line PHYSICAL INTERPRETATION (finding 5, science-
    scoped), and the governing equation stays adjacent as the masthead. Add the estimate MARKER on the viz that
    corresponds to the hero number (already have markers; ensure the hero and the marked point match).
P2. EQUATION <-> ANSWER coupling (finding 2, generalized): a subtle visual tie between the equation masthead and
    the answer (shared accent, "read off at the marker"), and where feasible highlight the answered quantity.
P3. CROSSHAIR at frame edges (finding 10): the field read-out already exists; keep the crosshair labels from
    obscuring marks (edge-anchored). Verify it holds after the fit rewrite.
P4. DOC PAGES to TABS (finding 6 + ADR-0016 §6): the five pages have long, non-comparative sections -> tabs are
    the RIGHT pattern (NN/g) and the ADR mandate. Implementation already uses SubTabs; bring Methodology,
    Experiments, Benchmark, Introduction to the same top-level Tabs/SubTabs structure. Content stays graduate-
    level and drawn from docs/methods + docs/cases (coherent with the repo, finding: real sources).
P5. REFERENCES per-section (ADR-0016 §7.5, done for Methodology): extend the no-bottom-bibliography +
    per-section Refs (citations.ts real DOIs) to any page that cites.
P6. HERO/type hierarchy (finding 5,8): the luminous answer tiles (shipped v0.26.002) match the size-hierarchy
    finding; keep. Give dense figures the full-width tier where they earn it.

Open questions the research flagged (not blockers): concrete type-scale/elevation tokens of premium science
microsites; animation-control UX specifics; colorbar/legend tick strategy; small-multiples for case comparison.
These are refinements, addressable after P1-P5.
