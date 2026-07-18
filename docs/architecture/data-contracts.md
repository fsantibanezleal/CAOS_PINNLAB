# The two data contracts

The pipeline is bracketed by two explicit, versioned schemas. One governs *what may come in* (observations, for
inverse cases); the other governs *what goes out to the web* (the artifact the SPA replays). Nothing crosses either
boundary unvalidated.

## 1 · Ingestion contract — raw observations → pipeline

**Owner:** `io/schema.py` (`ObservationRow`) + `io/contract.py` (`validate_observations`).

Only **inverse** cases consume external data; forward cases are fully specified by their PDE + domain and need none.
An observation table is a flat, columnar record:

```
case_id, x0, x1, …, x{d-1}, value[, weight]
```

- `x0..x{d-1}` — the `d` input coordinates (must match the case's `inputs` arity).
- `value` — the observed field value at that point (e.g. a measured concentration / head).
- `weight` — optional per-row confidence; defaults to 1.

`validate_observations` enforces:

- **schema** — required columns present, numeric, correct arity for the case;
- **domain** — coordinates inside the case domain (out-of-domain rows rejected with a reason);
- **outlier policy** — an explicit, declared rule (not a silent clip): rows beyond the configured bound are *flagged*
  with the reason recorded, never dropped invisibly.

Bad rows fail loudly with a per-row reason. This is the "bring-your-own-data" gate: an inverse case can be re-run on
a user's own measurements, and the contract is what makes that safe. The real-data inverse cases (USGS / OpenAQ, see
[`../../wip` real-datasets notes](../frameworks/)) flow through exactly this path.

## 2 · Artifact contract — pipeline → web

**Owner:** `core/manifest.py` + `core/trace.py`. Three record types, each versioned:

| Record | Version tag | Holds |
|--------|-------------|-------|
| **field trace** | `pinnlab.field/v1` | the baked field: axes (≤ `MAX_AXIS=81` per axis), output channel(s), values; the replay payload |
| **manifest** | `pinnlab.manifest/v1` | governing equation, SOTA method, engine, seed, `real_or_synthetic`, ONNX pointer + parity, lane + measured gate numbers, evaluation metrics |
| **index** | `pinnlab.index/v1` | the case catalog the SPA loads first (id, category, title, manifest path) |

The web loads **only** these artifacts — never the Python engine. `frontend/src/lib/contract.ts` mirrors the schema
in TypeScript, so any drift between what the pipeline writes and what the web expects **fails the frontend build**
rather than shipping a broken page.

### Why a trace *and* an ONNX

A live case ships both: the ONNX is the interactive engine (evaluate anywhere, any resolution), the trace is the
deterministic baseline the page renders instantly on load and the reference the live output is checked against. A
precompute case ships only the trace. The [gate](the-gate.md) decides which.

## The invariant

Both contracts are **append-only versioned** (`/v1`). A breaking change bumps the version and updates both the
Python writer and the TS reader in lockstep; CI fails if they disagree. That is what lets the offline engine and the
static web app evolve independently without silent corruption.
