# Guides

How to *do* things with PINN-Lab — task-oriented, numbered from "I just cloned it" to "I added my own case".

| # | Guide | You want to… |
|---|-------|--------------|
| 01 | [Getting started](01_getting-started.md) | Clone, build both environments, bake a case, and run the web app locally. |
| 02 | [Running the pipeline](02_running-the-pipeline.md) | Drive the offline pipeline — single case, all cases, seeds, determinism, lanes, `--quick`. |
| 03 | [Using the web app](03_using-the-web-app.md) | Read a case workbench — the variant bar, the Field/Live/Charts/Context sub-tabs, the cursor read-out. |
| 04 | [Adding a new case](04_adding-a-new-case.md) | Author a new PDE case end-to-end: spec → method → variants → bake → Context → verify. |
| 05 | [Interpreting results](05_interpreting-results.md) | Make sense of relative-L2, ONNX parity, lanes, and the honesty labels. |

All commands are shown in **both** PowerShell (`.ps1`, Felipe's Windows default) and bash (`.sh`) form. The repo ships
script parity under `scripts/`, so you rarely type the raw Python.
