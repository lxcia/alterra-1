"""Parse a free-text FINDINGS section into: is an endometrioma reported, and on which side(s)?

Clause-level rules (deliberately simple so they can be hand-audited; PATCH.md, P07):
  * sentences are split into CLAUSES at , ; : and the words "and", "but", "however", "with"
  * a clause is NEGATED if it contains a negation or exclusion cue ("no", "not", "without",
    "absence of", "negative for", "unremarkable", "normal", "rule out", "cannot be excluded", ...)
  * a sentence is POSITIVE for a lesion if a non-negated clause mentions an ovarian/adnexal LESION
    (endometrioma, chocolate cyst, cyst, lesion, mass); em_pos: some non-negated clause mentions
    an endometrioma (or "chocolate cyst")
  * lesion_sides: the sides ("left", "right"; "bilateral"/"both" -> both) named in the
    non-negated clauses of positive sentences
  * em_sides: the same, restricted to sentences with a non-negated endometrioma clause
Sides are returned as 'none', 'left', 'right' or 'both'.
"""
from __future__ import annotations

import re

NEG = re.compile(r"\b(no|not|without|absence of|absent|negative for|unremarkable|normal|rule out|ruled out|free of|"
                 r"cannot be excluded|cannot exclude|can't exclude|can't be excluded)\b", re.I)
CLAUSE = re.compile(r"[,;:]|\b(?:and|but|however|with)\b", re.I)
EM = re.compile(r"endometrioma|chocolate cyst|endometriotic cyst", re.I)
LESION = re.compile(r"endometrioma|chocolate cyst|endometriotic cyst|\bcysts?\b|\blesions?\b|\bmass(es)?\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if s.strip()]


def _sides(sentences: list[str]) -> str:
    left = right = False
    for s in sentences:
        low = s.lower()
        if re.search(r"\bbilateral|\bboth\b", low):
            left = right = True
        left |= bool(re.search(r"\bleft\b", low))
        right |= bool(re.search(r"\bright\b", low))
    return {(False, False): "none", (True, False): "left", (False, True): "right", (True, True): "both"}[(left, right)]


def parse_report(text: str) -> dict:
    pos_lesion, pos_em = [], []  # non-negated clauses of positive sentences
    for s in _sentences(text):
        clauses = [c for c in CLAUSE.split(s) if c and c.strip()]
        live = [c for c in clauses if not NEG.search(c)]
        if any(LESION.search(c) for c in live):
            pos_lesion.append(" ".join(live))
            if any(EM.search(c) for c in live):
                pos_em.append(" ".join(live))
    return dict(em_pos=bool(pos_em), n_lesion_sents=len(pos_lesion),
                lesion_sides=_sides(pos_lesion), em_sides=_sides(pos_em))
