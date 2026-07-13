# PINN-Lab max-depth product review: dynamics + organization (2026-07-10, issue #36)

Owner questions: Is this the best way and organization to show the USE of PINNs, their advantages and limitations?
Are these the best visualizations and dynamics? **Why are no dynamics observed?** Redesign and implement everything.

## 1. Diagnosis: why the app read as STATIC even after the deep pass

1. **The space-time carpet encodes motion instead of showing it.** For every 1-D evolution case the Field view's
   dominant element was the (x,t) heatmap: a picture OF the whole evolution, not the evolution itself. The play
   button existed, but all it visibly did was move a hairline cursor and update a SMALL side profile. The eye reads
   the big static image; the motion was demoted to the margin. (The one truly animated case, the double pendulum,
   is exactly the one the owner originally liked.)
2. **The comparisons were static panels.** The method ladder (standard | naive | adapted) shipped as side-by-side
   images + error maps. For (x,t) cases the same baked data contains the full evolution of every lane, and the most
   convincing form is watching the lanes evolve TOGETHER - the naive profile visibly peeling away from the standard.
   That animation was never rendered.
3. **The 2-D transport cases never moved.** ocean-transport and heap-leach-rt have time as a network input; the Field
   view showed ONE static snapshot, and only the Live slider could change t (one frame at a time, by hand).
4. **The deepest dynamics of all - TRAINING dynamics - were absent.** Spectral bias and the metastable collapse are
   TRAINING pathologies: the naive Helmholtz PINN never leaves its low-frequency blur across 12k iterations while the
   Fourier lane snaps onto the pattern in the first few hundred. Showing only the final fields hides the mechanism;
   showing the training trajectory IS the insight.
5. **Organization taught browsing, not judgment.** Domain groups (benchmarks / mining / pollution / industrial) are
   fine for finding a case, but they do not teach the thing the catalogue exists to teach: WHEN to reach for a PINN
   and when not to. That is a use-case narrative (easy forward -> hard forward -> inverse/data -> real data ->
   operator -> UQ -> chaotic limit), and it existed only implicitly.

Root cause in one line: **we computed the right content (deep pass, issue #25) but presented its dynamics as
stills.** Compute was fixed; presentation of MOTION was not.

## 2. What the redesign implements (all real, all replay of baked artifacts, no compute bombs)

| # | Gap | Implementation | Where |
|---|-----|----------------|-------|
| 1 | evolution not visible | **Animated evolution HERO**: for every time case the big element is now the MOVING profile u(space) at time t (locked y-scale, ghost = initial state), driven by the transport bar; the space-time map is demoted to the seek/probe carpet below. Both-mode probe (follow / pin / release) retained. | `FieldView.tsx` (Field + Live tabs) |
| 2 | static comparisons | **CompareEvolution**: for any (space,t) comparison, PLAY the lanes together - standard (grey), naive (red), adapted (green) profiles evolving on one locked scale. Zero new compute: it replays the baked comparison fields. Lights up on allencahn, soil-barrier, burgers, heat1d, wave, thickener, tailings, comminution, flotation. | `CompareEvolution.tsx` in `CompareKit` |
| 3 | 2-D transport frozen | **Evolution frames**: offline ONNX evals bake a smooth t-sequence (ocean 24 frames; heap-leach 16 x 2 species); the Field view becomes the ANIMATED 2-D field (SpatioTemporalKit, fixed colour scale across frames). | `build_evolution_frames.py`; manifests -> `evolution` + `view_kit` |
| 4 | training dynamics absent | **TrainingKit - "watch it learn"**: real checkpoint fields (0..12k iters) for naive vs adapted, side by side on one colour scale, with live per-lane L2 and the L2-vs-iteration (log) curve with a moving cursor. Baked for helmholtz (spectral bias visible: naive L2 ~1.0-1.6 at EVERY checkpoint) + allencahn (the metastable slide). | `build_training_dynamics.py`; `TrainingKit.tsx`; manifest `training` |
| 5 | no use-case narrative | **The story: when PINNs win / lose** - an 8-chapter selector in the App rail (each chapter lands on the case whose computed Compare/Training/Diagnostics view DEMONSTRATES it) + the same storyline written into the Introduction. | `AppPage.tsx` STORY, `Introduction.tsx` |

All animations obey the standing rules: paused by default, play runs once then stops, loop is opt-in, animation
halts on hidden tab (`useAnimator`).

## 3. Answer to the owner's questions (my evaluation)

- **Is domain grouping the best organization?** For browsing, yes - keep it. For TEACHING advantage/limitation, no:
  the missing axis is the use-case storyline, now added as the primary entry point (the story selector) without
  destroying the domain browse. Both axes coexist; neither is invented.
- **Are these the best visualizations?** The static forms (comparison panels + error maps, probe heatmaps,
  benchmark-scatter vs model-curve) match what the strongest PINN resources use. What was missing - and is now the
  centrepiece - is MOTION where the content is motion: solution evolution, ladder evolution, and training evolution.
- **Advantages/limitations honestly?** The ladder already carries it (chapter 1 explicitly says the classical solver
  wins the easy forward case; chapter 8 shows chaos defeating the surrogate). The storyline makes that narrative
  navigable instead of implicit.

## 4. Research grounding (deep-research dossier: `dynamics-research-2026-07-10.md`, 13 verified findings)

The fan-out + adversarially-verified research CONFIRMS the redesign direction and sharpens it:

- **Organization**: no single pattern dominates the best 2024-2026 resources; three coexist by purpose. The
  most-cited PINN library (DeepXDE) organizes **use-case-first** (function approximation / forward / inverse /
  operator learning); the strongest research showcases (jaxpi + the Expert's Guide arXiv:2308.08468, PINNacle)
  organize by benchmark in service of an explicit **failure-mode-and-remedy narrative**; framework vendors
  (PhysicsNeMo) go domain-first with PINNs demoted to one method tag (5 of 37 cards). The research's explicit
  conclusion for an educational catalogue: **a use-case-narrative primary axis with an honesty layer of openly
  published weak results and failure-mode framing, delivered as animated field evolution plus error quantification**
  - exactly the STORY selector + honest ladder + animated heroes implemented here. Keeping the domain browse as the
  secondary axis matches the coexistence pattern.
- **Visualization**: the verified state of practice for time-dependent solutions is **animated field-evolution
  paired with quantitative relative-L2 tables** - implemented (animated heroes + the L2 headlines/tables). Notably,
  **no surviving evidence establishes training-dynamics animation as standard practice**: our TrainingKit is
  BEYOND the state of practice, i.e. the catalogue's genuine novel presentation element (grounded in Krishnapriyan
  NeurIPS 2021: the pathologies are training-time, ill-conditioned loss landscapes - so showing training IS showing
  the mechanism). Claimed as novel presentation, not as a research result.
- **Honesty**: the strongest resources co-publish weak results next to strong ones in the same table (jaxpi's
  16-35% errors on chaotic/transitional flows), name the pathology and its root cause, and pair every limitation
  with a named mitigation - the pattern the method ladder + story chapters follow (chapter 1 says the classical
  solver wins; chapter 8 shows chaos defeating the surrogate).
