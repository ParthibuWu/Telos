from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

CANONICAL = set("ACGT")

@dataclass(frozen=True)
class BaseComposition:
    counts: Dict[str, int]
    denom: int
    fractions: Dict[str, Optional[float]]

def clean_atgc(seq: str, treat_u_as_t: bool = True) -> str:
    s = (seq or "").strip().upper()
    cleaned = []
    for ch in s:
        if treat_u_as_t and ch == "U":
            ch = "T"
        if ch in CANONICAL:
            cleaned.append(ch)
    return "".join(cleaned)

def atgc_content(seq: str, treat_u_as_t: bool = True) -> BaseComposition:
    cleaned = clean_atgc(seq, treat_u_as_t=treat_u_as_t)
    counts = {"A": 0, "T": 0, "G": 0, "C": 0}
    for ch in cleaned:
        counts[ch] += 1
    denom = len(cleaned)
    fractions = {base: (counts[base] / denom if denom > 0 else None) for base in "ATGC"}
    return BaseComposition(counts=counts, denom=denom, fractions=fractions)

def gc_fraction(seq: str, treat_u_as_t: bool = True) -> Optional[float]:
    comp = atgc_content(seq, treat_u_as_t=treat_u_as_t)
    if comp.denom == 0:
        return None
    return (comp.counts["G"] + comp.counts["C"]) / comp.denom