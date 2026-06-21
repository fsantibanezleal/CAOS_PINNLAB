# Deploy

PINN-Lab is a **static site**: the offline pipeline bakes artifacts into the repo, and the frontend is a plain
Vite/React build served by **GitHub Pages via Actions**. No server, no runtime backend (the FastAPI lane is dormant —
ADR-0057's backend-optional clause).

## The build

`frontend/` is a Vite + React 19 + TypeScript SPA:

- **`base: "./"`** in `vite.config.ts` — all asset URLs are relative, so the bundle works under a project subpath
  (`/CAOS_PINNLAB/`) and would also work under a custom domain root without rebuilding.
- **`copy-data.mjs`** — a prebuild step that copies `data/derived/` (manifests + traces) and `models/*.onnx` into the
  frontend `public/` so they ship as static assets the SPA `fetch`es at runtime. The pipeline output is the single
  source; the web never re-derives it.
- **Type-check** — `tsc --noEmit` runs as a separate gate (no `noEmit`/project-reference conflict), so a contract
  drift in `lib/contract.ts` fails the build.

## Routing

The SPA uses **HashRouter**. GitHub Pages has no SPA rewrite, so a `BrowserRouter` deep-link (`/methodology`) would
404 on refresh; `#/methodology` is served by the single `index.html` and routed client-side. Six pages:
App · Introduction · Methodology · Implementation · Experiments · Benchmark.

## GitHub Pages (Actions)

`.github/workflows/deploy-pages.yml` builds on every push to `main` and publishes to Pages:

1. `npm ci` + `npm run build` in `frontend/` (runs `copy-data` then `vite build`);
2. upload `frontend/dist` as the Pages artifact;
3. deploy to the `github-pages` environment.

Two gotchas already resolved, recorded here so they are not re-hit:

- **Environment branch policy.** The `github-pages` environment defaulted to "only `main`-protected" and rejected the
  deploy with *"Branch 'main' is not allowed to deploy"*. Fixed by clearing the environment's
  `deployment_branch_policy` (set to `null`) via `gh api`.
- **CNAME does not set the Actions domain.** A `CNAME` file in the artifact is ignored by Actions deploys; the custom
  domain must be set via `gh api PUT .../pages -f cname=…` (cname-only, no `https_enforced`) *then* redeploy, or the
  domain 404s. (Custom domain `pinnlab.fasl-work.com` is staged pending DNS.)

Live: **https://fsantibanezleal.github.io/CAOS_PINNLAB/**. Each push to `main` redeploys.

## CI guard (`ci.yml`)

Separate from deploy: a smoke job runs `pipeline.py bench-poisson2d --quick` (full chain incl. ONNX export + parity +
gate) and re-derives the lane from each committed manifest, failing on any mislabel. So `main` is never deployed with
a broken pipeline or a dishonest lane.
