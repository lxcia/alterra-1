"""Answer parsing. Unparseable outputs are returned as 'unparsed' (never dropped).

All prompts ask the model to conclude with 'Final Answer: <x>'. We take the LAST such
line; if absent, we fall back to a conservative whole-text rule and tag the method so
fallback parses can be audited separately.

Rules for the text after the last 'Final Answer:' (PATCH.md, P07; tests in review/parser_tests.py):
  * markdown, parenthesised asides and a repeated 'Final Answer:' header are removed, and a
    leading 'the answer is' is skipped;
  * yes/no: an answer naming both yes and no (e.g. 'yes/no') or 'N/A' is unparsed;
  * side: a side word right after 'not'/'no' is skipped; both 'left' and 'right' -> bilateral;
  * lesion type: exactly one type must be named ('teratoma' = dermoid); a negated or ambiguous
    answer is unparsed.
"""
from __future__ import annotations

import re

FINAL_RE = re.compile(r"final\s*answer\s*[:\-]?\s*\**\s*\(?([a-z][a-z \-]*)", re.I)
FINAL_LINE_RE = re.compile(r"final\s*answer\s*[:\-]?([^\n]*(?:\n\s*\n?[^\n]*)?)", re.I)

YES = {"yes", "y", "present", "positive", "true"}
NO = {"no", "n", "absent", "negative", "false", "none"}
SIDES = {"left": "left", "right": "right", "bilateral": "bilateral", "both": "bilateral",
         "none": "none", "neither": "none", "no": "none"}
LESION = {"endometrioma": "endometrioma", "simple cyst": "simple cyst", "simple": "simple cyst",
          "cyst": "simple cyst", "dermoid": "dermoid", "hemorrhagic cyst": "hemorrhagic cyst",
          "haemorrhagic cyst": "hemorrhagic cyst", "none": "none", "no lesion": "none"}


def _final(text: str) -> str | None:
    hits = FINAL_RE.findall(text or "")
    return hits[-1].strip().lower() if hits else None


def _final_line(text: str) -> str | None:
    """The answer text after the LAST 'Final Answer:', cleaned (see module docstring)."""
    t = text or ""
    starts = [m.end() for m in re.finditer(r"final\s*answer\s*[:\-]?", t, re.I)]
    if not starts:
        return None
    # a doubled header ('Final Answer: Final Answer: X') leaves an empty tail: use the last
    # header that is followed by some answer text
    for st in reversed(starts):
        rest = t[st:]
        line = rest.lstrip(" *\t:-")
        if not line.strip():
            continue
        if line.startswith("\n"):  # answer on the next non-empty line
            line = line.lstrip()
        line = line.split("\n", 1)[0]
        line = re.sub(r"[*_`]", "", line).lower()
        bare = re.sub(r"\([^)]*\)", " ", line)  # parenthesised asides ...
        line = bare if bare.strip() else re.sub(r"[()]", " ", line)  # ... unless that is all there is
        line = re.sub(r"^\s*(the\s+)?answer\s+is\s+", "", line.strip())
        if line.strip():
            return line.strip()
    return ""


def parse_yesno(text: str) -> tuple[str, str]:
    """-> (answer in {'yes','no','unparsed'}, method)."""
    f = _final_line(text)
    if f is not None:
        if re.match(r"^n\s*/\s*a\b", f):
            return "unparsed", "none"
        words = re.findall(r"[a-z]+", f)
        if words:
            if "yes" in words[:3] and "no" in words[:3]:
                return "unparsed", "none"
            w = words[0]
            if w in YES:
                return "yes", "final"
            if w in NO:
                return "no", "final"
            if re.match(r"^there\s+is\s+no\b", f):
                return "no", "final"
    t = (text or "").strip().lower()
    first = re.split(r"[\s.,:;!]+", t, maxsplit=1)[0] if t else ""
    if first in ("yes", "no"):
        return first, "first_word"
    return "unparsed", "none"


def parse_side(text: str) -> tuple[str, str]:
    """-> (answer in {'left','right','bilateral','none','unparsed'}, method)."""
    f = _final_line(text)
    if f is not None:
        words = re.findall(r"[a-z]+", f)
        lat = {"left", "right", "bilateral", "both"}
        if words and words[0] in ("none", "neither", "no") and not (len(words) > 1 and words[1] in lat):
            return "none", "final"
        found = []
        for i, w in enumerate(words):
            if w in lat and not (i > 0 and words[i - 1] in ("not", "no")):
                found.append(SIDES[w])
        if found:
            if "bilateral" in found or ("left" in found and "right" in found):
                return "bilateral", "final"
            return found[0], "final"
    t = (text or "").strip().lower()
    first = re.split(r"[\s.,:;!]+", t, maxsplit=1)[0] if t else ""
    if first in SIDES:
        return SIDES[first], "first_word"
    return "unparsed", "none"


TYPE_RES = {"endometrioma": r"\bendometriomas?\b", "dermoid": r"\bdermoid\b|\bteratoma\b",
            "hemorrhagic cyst": r"\bha?emorrhagic\b", "simple cyst": r"\bsimple\b|^cysts?\.?$"}


def parse_lesion_type(text: str) -> tuple[str, str]:
    f = _final_line(text)
    if f is not None:
        if re.match(r"^(none|no lesion|no cyst)\b", f):
            return "none", "final"
        if re.match(r"^not\b", f):
            return "unparsed", "none"
        found = {k for k, rx in TYPE_RES.items() if re.search(rx, f)}
        if len(found) == 1:
            return found.pop(), "final"
    return "unparsed", "none"


def parse_choice(text: str, letters: str = "ABCDE") -> tuple[str, str]:
    """Multiple-choice letter after 'Final Answer:'."""
    m = re.findall(r"final\s*answer\s*[:\-]?\s*\**\s*\(?([%s])\b" % letters, text or "", re.I)
    if m:
        return m[-1].upper(), "final"
    return "unparsed", "none"
