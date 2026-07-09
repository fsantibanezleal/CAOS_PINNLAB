"""Universal, TRAINING-FREE standard-vs-PINN comparison for every case with a closed-form/reference solution.

The adapted PINN field is ALREADY baked in each variant's trace; the standard (analytic / MMS) is numpy-only. So for
every analytic case we can bake, with NO retraining, a comparison trace {standard, adapted, |adapted-standard|} and
patch its manifest with a `comparison` block. This gives the whole catalogue the "standard PDE solution vs the PINN"
view immediately. Cases that already carry a richer comparison (a naive lane, e.g. Helmholtz) are skipped.

Nothing invented: `standard` is the case's own closed-form reference; `adapted` is the baked PINN field.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinnlab import registry
from pinnlab.model.analytic import l2_relative, param_grid
from pinnlab.registry import case_module

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MANIFESTS = DERIVED / "manifests"


def rnd(a):
    return np.round(np.asarray(a, dtype=float), 5).tolist()


def main():
    has_full_ladder = {"ind-helmholtz"}  # cases getting a dedicated naive/adapted/diagnostics ladder run
    done, skipped = [], []
    for case in registry.list_cases():
        cid = case.id
        if cid in has_full_ladder:
            skipped.append((cid, "dedicated full ladder")); continue
        man_path = MANIFESTS / f"{cid}.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if man.get("comparison"):
            skipped.append((cid, "already has comparison")); continue
        mod = case_module(cid)
        analytic = getattr(mod, "analytic", None)
        if man.get("validation_anchor") != "analytic" or analytic is None:
            skipped.append((cid, f"no analytic anchor ({man.get('validation_anchor')})")); continue
        primary = man["outputs"][0]
        var = man["variants"][0]
        trace_path = DERIVED / var["trace"]["path"]
        if not trace_path.exists():
            skipped.append((cid, "no baked trace")); continue
        tr = json.loads(trace_path.read_text(encoding="utf-8"))
        if primary not in tr.get("fields", {}):
            skipped.append((cid, f"primary {primary} not in trace")); continue
        adapted = np.asarray(tr["fields"][primary], dtype=float)

        # standard = the case's closed-form reference on the SAME (decimated) grid the trace was baked on, +
        # this variant's parameters. Build the query from the TRACE's own axes so shapes match the baked field.
        params = var.get("params") or {}
        dims = list(tr["dims"])
        coord_vecs = {a: np.asarray(tr["axes"][a], dtype=float) for a in dims}
        mesh = np.meshgrid(*[coord_vecs[a] for a in dims], indexing="ij")
        field_cols = {a: m.ravel() for a, m in zip(dims, mesh)}
        shape = tuple(len(coord_vecs[a]) for a in dims)
        n = int(next(iter(field_cols.values())).size)
        XY = np.empty((n, len(case.inputs)), dtype=np.float64)
        for j, ax in enumerate(case.inputs):
            XY[:, j] = field_cols[ax] if ax in field_cols else float(params.get(ax, 0.0))
        truth = analytic(XY)
        if truth is None:
            skipped.append((cid, "analytic returned None")); continue
        truth = np.asarray(truth, dtype=float).reshape(shape)
        if truth.shape != adapted.shape:
            skipped.append((cid, f"shape mismatch {truth.shape} vs {adapted.shape}")); continue
        err = np.abs(adapted - truth)
        rel = round(l2_relative(adapted, truth), 5)

        cmp_trace = {
            "schema": "pinnlab.compare/v1",
            "case_id": cid,
            "dims": list(tr["dims"]),
            "axes": tr["axes"],
            "fields": {"standard": rnd(truth), "adapted": rnd(adapted), "err_adapted": rnd(err)},
            "summary": {"adapted_vs_std": rel},
        }
        (DERIVED / cid).mkdir(parents=True, exist_ok=True)
        (DERIVED / cid / "comparison.json").write_text(json.dumps(cmp_trace, ensure_ascii=False), encoding="utf-8")

        man["comparison"] = {
            "trace": f"{cid}/comparison.json",
            "lanes": [
                {"key": "standard", "label_en": "standard (analytic)", "label_es": "estándar (analítica)", "role": "reference"},
                {"key": "adapted", "label_en": f"{case.method} PINN", "label_es": f"PINN {case.method}", "role": "fix", "err": "err_adapted"},
            ],
            "note_en": f"The PINN ({case.method}) field vs the closed-form standard solution, and their pointwise error. Relative-L2 = {rel*100:.1f}%.",
            "note_es": f"El campo de la PINN ({case.method}) frente a la solución estándar de forma cerrada, y su error puntual. L2 relativa = {rel*100:.1f}%.",
        }
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        done.append((cid, rel))

    print("=== standard comparisons baked (no training) ===")
    for cid, rel in done:
        print(f"  {cid:24s} adapted_vs_std L2 = {rel*100:.1f}%")
    print(f"baked: {len(done)}   skipped: {len(skipped)}")
    for cid, why in skipped:
        print(f"  skip {cid:24s} {why}")


if __name__ == "__main__":
    main()
