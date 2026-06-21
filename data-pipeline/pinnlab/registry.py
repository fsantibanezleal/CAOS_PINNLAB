"""The case registry — cases grouped by CATEGORY. The App shows ONE selected case; Experiments/Benchmark show
cross-case summaries by category."""
from __future__ import annotations

from .cases import CASES, get_module
from .cases.base import CaseSpec, Variant

_BY_ID: dict[str, CaseSpec] = {c.id: c for c in CASES}


def case_variants(case_id: str) -> list[Variant]:
    """The case's parameter-regime family (SimLab-style). A case module may expose `variants() -> list[Variant]`;
    otherwise the case ships a single 'default' variant (non-parametric: empty params)."""
    mod = get_module(case_id)
    fn = getattr(mod, "variants", None)
    if fn is not None:
        vs = list(fn())
        if vs:
            return vs
    return [Variant("default", "Default", "Por defecto", {})]


def list_cases() -> list[CaseSpec]:
    return list(CASES)


def get_case(case_id: str) -> CaseSpec:
    if case_id not in _BY_ID:
        raise KeyError(f"unknown case: {case_id!r}. known: {sorted(_BY_ID)}")
    return _BY_ID[case_id]


def case_module(case_id: str):
    """The case's heavy module (exposes build()/analytic()). Import-light: pulls deepxde only when build() runs."""
    return get_module(case_id)


def list_categories() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in CASES:
        out.setdefault(c.category, []).append(c.id)
    return out
