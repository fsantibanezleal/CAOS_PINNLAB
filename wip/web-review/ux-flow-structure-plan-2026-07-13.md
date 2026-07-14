# App flow & structure improvement plan (PROPOSAL, 2026-07-13; pending owner validation)

Owner question: what is the plan to improve the FLOW and STRUCTURE of the web app. This is the persisted
proposal. Grounded in the app as it stands at v0.23.000/v0.24.000 (estimation-centric workbench, 21 cases,
9-chapter story, deep links); every weakness below is observed in the current code, not hypothesized.

## Diagnosis: where the flow breaks today

1. ENTRY IS ARBITRARY. A first visit (no deep link) lands on bench-poisson2d (first case of the first
   category), not on the story or the flagship. The narrative spine exists but the visitor must FIND it.
2. TWO COMPETING NAV AXES, NO HIERARCHY. The story dropdown and the domain dropdown sit stacked in the rail
   with equal visual weight. The story is the pedagogical axis, but as a <select> it reads as a filter, not
   as the spine. Users do not know which control drives the app.
3. THE STAGE STARTS TOO LOW. AnswerCard + equation row + constraint chips + meta + honesty band stack above
   the stage; on a 768-900px-tall viewport the visualization is at or below the fold. The context is all
   valuable, but it is ALWAYS fully expanded.
4. VIEW SWITCHING IS FAR FROM WHAT IT SWITCHES. The view buttons live at the BOTTOM of the left rail; the
   stage they control is center-right. Standard workbench pattern is tabs across the stage top.
5. WEAK CROSS-CASE FLOW. From a case there is no inline "vs its siblings" summary; comparing means leaving
   to the Benchmark page and finding the row. (The Benchmark case link back IS a deep link: the reverse
   flow exists, the forward one does not.)
6. LONG READING PAGES WITHOUT IN-PAGE NAV. Introduction/Methodology/Implementation/Experiments are long
   scrolls with no sticky TOC; concept-to-case links exist only where hand-placed (estimators section).
7. NO CASE SEARCH. 21 cases and growing; finding a case requires knowing its domain bucket. The questions
   are now on the cards (good) but not searchable.
8. NO FIRST-VISIT PRIMER IN THE APP. The Introduction teaches the frame, but the App itself never says
   "each case = question -> estimate -> the views that earn it" to a visitor who lands cold.
9. RESPONSIVE COVERAGE UNAUDITED BELOW ~1100px. The workbench grid has one media tweak (band alignment);
   the 300px rail + stage layout has no drawer/stacked mode for narrow screens.

## Proposed plan (4 phases, each a validated screenshot-QA'd release, one issue per phase)

### Phase 1 - the narrative spine and the entry (highest value)
- First visit with no deep link lands on STORY CHAPTER 1 (poisson, "classical wins"), not on an arbitrary case.
- Replace the story <select> with a compact horizontal STEPPER (1-9 numbered dots + label) pinned above the
  workbench; prev/next chapter buttons; the current chapter's question+evidence line under it. The domain
  dropdown + case list stay in the rail as "browse all cases" (demoted, clearly secondary).
- A dismissible one-time primer card (localStorage): three sentences: the question, the computed estimate,
  the views that prove it; "walk the story" CTA.
- "Next chapter" affordance at the stage bottom when the active case is a story chapter.

### Phase 2 - workbench ergonomics
- Move the view switch to TABS across the stage top (Compare | Field | Live | Charts | Training |
  Diagnostics | Context), directly attached to what they change; rail keeps selection only.
- Collapsible context strip: after the first case visited, equation + chips collapse to one summary line
  (toggle persists in localStorage); AnswerCard always stays visible (it IS the product).
- Keyboard: left/right arrows switch regime chips; [ ] switch views (power-user flow, documented in a
  tooltip).

### Phase 3 - findability and cross-case flow
- A search box above the case list filtering by case name + question text (both languages).
- An inline "vs siblings" strip on each case (its own Benchmark numbers + the category's min/median, one
  row, link to the full Benchmark table).
- Sticky in-page TOC for the four reading pages; systematic concept->case chips (each method family names
  its demonstrating case as a deep-link chip, everywhere, not only in the estimators section).

### Phase 4 - responsive audit
- Real breakpoints: under ~900px the rail becomes a top drawer (case picker as a modal sheet), stage first;
  under 640px the AnswerCard items stack. Screenshot QA at 1500/1024/390 widths, both themes, per case class.

## Effort and order

Phases are independent releases in the order above (1 = most value per effort). Each: issue -> implement ->
pl-validate-all + a new flow-QA harness (entry path, stepper, tabs, search, drawer) -> release. No content
or pipeline changes required; this is purely presentation/flow: zero risk to the baked artifacts.

## Status

- PROPOSAL persisted; awaiting owner validation before any implementation (the v0.24.000 flagship release
  is finishing in parallel and is unaffected).
