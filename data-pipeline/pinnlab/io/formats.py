"""Standard-format readers/writers.

In: CSV (observation tables for inverse cases) + the case registry (PDE configs). Out: compact JSON (the committed
web-replay artifact) + ONNX (the trained PINN, written by stages/export). Heavy intermediate fields go to npz under
data/raw/ (git-ignored). Never invent a bespoke ad-hoc format — keep everything standard so data is portable.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, obj: Any) -> int:
    """Write compact JSON; return the byte size (used by the gate + manifest). UTF-8, no BOM."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    p.write_bytes(encoded)
    return len(encoded)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_npz(path: str | Path, **arrays: np.ndarray) -> int:
    """Write a heavy intermediate field bundle (data/raw, git-ignored). Returns byte size."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
    return p.stat().st_size


def strip_onnx_metadata(path: str | Path) -> None:
    """Clear doc_strings + metadata_props from an exported ONNX. The dynamo exporter embeds the absolute local build
    path in the graph metadata; this keeps the committed public artifact free of local-machine paths (CI guard) without
    touching the weights or graph (inference + parity are unaffected)."""
    import onnx

    p = Path(path)
    m = onnx.load(str(p))
    m.doc_string = ""
    del m.metadata_props[:]
    g = m.graph
    g.doc_string = ""
    for node in g.node:
        node.doc_string = ""
        del node.metadata_props[:]
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        vi.doc_string = ""
    onnx.save(m, str(p))
