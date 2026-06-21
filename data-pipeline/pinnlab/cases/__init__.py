"""Documented cases, each carrying a CATEGORY (the domain problem-type taxonomy). The registry groups them by
category; the App shows ONE selected case, while Experiments/Benchmark show cross-case summaries by category.

Group A (canonical-benchmark) is implemented first to harden the engine + the train->ONNX->web contract; the
mining/pollution differentiators (Groups B+C) and industrial (Group D) follow per the agreed build order.
"""
from __future__ import annotations

from types import ModuleType

from . import (
    bench_allencahn,
    bench_burgers1d,
    bench_darcy_operator,
    bench_heat1d,
    bench_navier_cavity,
    bench_poisson2d,
    bench_wave1d,
    ctrl_zero_source,
    env_soil_heat_real,
    ind_heat2d_inverse,
    ind_helmholtz,
    mine_comminution_pbe,
    mine_flotation_kinetics,
    mine_heap_leach_rt,
    mine_thickener_settling,
    poll_ocean_transport,
    poll_soil_barrier,
    poll_source_uq_bpinn,
    poll_tailings_seepage,
)
from .base import CaseSpec

# Order here = display order. One module per case (dossier §7).
_MODULES: list[ModuleType] = [
    bench_poisson2d,
    bench_heat1d,
    bench_wave1d,
    bench_burgers1d,
    bench_allencahn,
    bench_navier_cavity,
    ind_helmholtz,
    poll_ocean_transport,
    mine_heap_leach_rt,
    poll_soil_barrier,
    ctrl_zero_source,
    ind_heat2d_inverse,
    mine_thickener_settling,
    poll_tailings_seepage,
    mine_flotation_kinetics,
    mine_comminution_pbe,
    env_soil_heat_real,
    poll_source_uq_bpinn,
    bench_darcy_operator,
]

CASES: list[CaseSpec] = [m.CASE for m in _MODULES]
_BY_ID: dict[str, ModuleType] = {m.CASE.id: m for m in _MODULES}


def get_module(case_id: str) -> ModuleType:
    if case_id not in _BY_ID:
        raise KeyError(f"unknown case: {case_id!r}; known: {sorted(_BY_ID)}")
    return _BY_ID[case_id]
