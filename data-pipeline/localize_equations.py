"""Bake `governing_equations_es`: the Spanish reading of each case's governing equation.

The equation is one LaTeX string per case. The MATH is language-neutral, but the prose inside `\\text{...}`
was English only ("rate constant", "recover k from sparse T obs", "validate vs held-out 10/20/50 cm"), so the
Spanish app rendered English inside its equations. 18 of the 21 cases carry such annotations.

This step rewrites ONLY the `\\text{...}` payloads through PHRASES and writes `governing_equations_es` next to
the untouched original. It is training-free and deterministic: it reads the committed manifests and writes
back one field.

It FAILS LOUDLY on an annotation it does not know, so a new case cannot silently ship English into the
Spanish app: add the phrase here and re-run.

Run:  python data-pipeline/localize_equations.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "derived" / "manifests"

# Every distinct \text{...} payload across the catalogue. Symbols, units and standard abbreviations
# (Re, rtol, atol, cm, FNO, RK45) stay as they are: they read the same in Spanish.
PHRASES: dict[str, str] = {
    "on": "en",
    "recover": "recuperar",
    "learn the operator": "aprender el operador",
    "(FNO)": "(FNO)",
    "Re": "Re",
    "anchor: RK45": "ancla: RK45",
    "rtol": "rtol",
    "atol": "atol",
    "cm": "cm",
    "obs": "obs",
    "5 \\& 100 cm = real Dirichlet boundaries": "5 \\& 100 cm = bordes de Dirichlet reales",
    "validate vs held-out 10/20/50 cm": "validar contra 10/20/50 cm retenidos",
    "from sparse": "desde datos dispersos",
    "from sparse noisy": "desde datos dispersos y ruidosos",
    "samples": "muestras",
    "size-transport reduction of the PBE": "reducción tamaño-transporte del PBE",
    "rate constant": "constante cinética",
    "degenerate above": "degenera sobre",
    "outside": "fuera",
    "inside": "dentro",
    "deep ensemble": "ensamble profundo",
    "mean": "media",
    "std": "desv. est.",
    "(epistemic UQ from sparse noisy sensors)": "(UQ epistémica desde sensores dispersos y ruidosos)",
}

TEXT = re.compile(r"(\\(?:text|mathrm)\{)([^}]*)(\})")


def localize(eq: str, case_id: str, unknown: list[str]) -> str:
    def sub(m: re.Match) -> str:
        head, body, tail = m.group(1), m.group(2), m.group(3)
        key = body.strip()
        if key not in PHRASES:
            unknown.append(f"{case_id}: {key!r}")
            return m.group(0)
        # keep the original padding so spacing inside the equation is unchanged
        lead = body[: len(body) - len(body.lstrip())]
        trail = body[len(body.rstrip()):]
        return f"{head}{lead}{PHRASES[key]}{trail}{tail}"

    return TEXT.sub(sub, eq)


def main() -> None:
    unknown: list[str] = []
    touched = skipped = 0
    for f in sorted(MANIFESTS.glob("*.json")):
        if f.name == "index.json":
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        eq = m.get("governing_equations")
        if not eq:
            continue
        es = localize(eq, m["case_id"], unknown)
        if es == eq:
            # no annotations: the equation reads identically in both languages, no ES twin needed
            m.pop("governing_equations_es", None)
            skipped += 1
        else:
            m["governing_equations_es"] = es
            touched += 1
        f.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    if unknown:
        raise SystemExit(
            "UNKNOWN equation annotations (add them to PHRASES so no English reaches the Spanish app):\n  "
            + "\n  ".join(unknown)
        )
    print(f"governing_equations_es baked for {touched} cases; {skipped} need none (pure math).")


if __name__ == "__main__":
    main()
