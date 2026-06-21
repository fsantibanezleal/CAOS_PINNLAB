# Cases

Each case is one documented PINN problem: a governing equation, a SOTA method, a validation anchor, and a baked
artifact the web app replays. Cases carry a **category** (the registry groups by it) and an honesty flag
(`synthetic` · `synthetic-illustrative` · `validated-real`). Every number below is the committed manifest's measured
value; lanes are derived from measurements (see [the gate](../architecture/the-gate.md)).

## The catalogue (17 cases)

| Category | Cases |
|----------|-------|
| **canonical-benchmark** | poisson2d · heat1d · wave1d · burgers1d · allencahn · navier-cavity |
| **industrial-fluids-heat** | helmholtz · heat2d-inverse |
| **mining / mineral-processing** | heap-leach-rt · thickener-settling · flotation-kinetics · comminution-pbe |
| **pollution / environmental** | ocean-transport · soil-barrier · tailings-seepage · **soil-heat-real (REAL DATA)** |
| **control** | zero-source |

## Honesty: what "real" means here

Most cases are **synthetic** (closed-form/MMS truth) or **synthetic-illustrative** (a faithful reduced model of a
real process, but the field is illustrative, not fit to a measured dataset). Exactly one case is trained and
validated against a **real measured dataset**:

- [**env-soil-heat-real**](env-soil-heat-real.md) — recovers soil thermal diffusivity from NOAA USCRN soil
  temperatures, validated out-of-sample against held-out depths. **This is the flagship real-data case.**

The mining/pollution differentiators are honest about their status: the reduced models (flotation first-order
kinetics, comminution size-transport, Bürger–Concha thickening, Richards/Gardner seepage) are the standard
engineering closures, but no open plant/field dataset exists for them (documented in the management dossier's
`real-datasets.md`), so they are labeled `synthetic-illustrative`, never dressed up as fit-to-data.

## Per-case write-ups

- [env-soil-heat-real](env-soil-heat-real.md) — the real-data inverse (USCRN).

(Remaining per-case pages are added as the catalogue is documented; every case's full spec — equation, method,
anchor, metrics — is always available live in its manifest and on the App/Benchmark pages.)
