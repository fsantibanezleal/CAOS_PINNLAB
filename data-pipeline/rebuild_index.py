"""Regenerate `manifests/index.json` from the case registry WITHOUT retraining anything.

`pipeline.run_all()` is the only writer of the index, and it retrains every case to get there. When a case is
added (or its metadata changes) the index has to list it, but nothing about the index depends on training:
every field comes from the CaseSpec. This rebuilds it from the registry alone, so adding a case does not cost
a full catalogue rebuild.

It refuses to list a case that has no manifest on disk, so the index can never advertise a case the web app
would then fail to load.

Run:  python data-pipeline/rebuild_index.py
Then: python data-pipeline/build_estimates.py   (it patches the per-case questions back into the index)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnlab import registry  # noqa: E402
from pinnlab.core.manifest import build_index  # noqa: E402
from pinnlab.io.formats import write_json  # noqa: E402

MANIFESTS = Path(__file__).resolve().parents[1] / "data" / "derived" / "manifests"


def main() -> None:
    entries, missing = [], []
    for c in registry.list_cases():
        if not (MANIFESTS / f"{c.id}.json").exists():
            missing.append(c.id)
            continue
        entries.append({
            "case_id": c.id, "category": c.category, "title": c.title,
            "manifest_path": f"manifests/{c.id}.json",
            "system_type": c.system_type, "view_kit": c.kit,
            "method": c.method, "real_or_synthetic": c.real_or_synthetic,
        })
    if missing:
        raise SystemExit(
            "refusing to write the index: these registered cases have no manifest (build them first):\n  "
            + "\n  ".join(missing)
        )
    write_json(MANIFESTS / "index.json", build_index(entries))
    print(f"index.json rebuilt: {len(entries)} cases")
    for e in entries:
        print(f"  {e['case_id']:26} {e['method']}")


if __name__ == "__main__":
    main()
