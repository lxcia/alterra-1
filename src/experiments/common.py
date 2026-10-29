"""Shared helpers for experiment scripts: case lists, prompts, and a result-table runner.

Every experiment script imports this first. It
  * puts src/ on the import path and fixes the Hugging Face cache location;
  * defines the question texts shared across experiments (Q_ENDO, Q_SIDE, Q_TYPE);
  * turns derived/inventory.csv into case lists (`d2_cases`, `cohort_cases`), which is where the
    excluded subjects are dropped and each case gets its ground-truth `label`;
  * decides where results go (`out_dir`) and writes them with a provenance record (`save`).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))  # so experiments can `import io_orient, pipeline, models, ...`
# The login shell sets HF_HOME to a different (empty, token-less) cache; always use the cache that
# holds the downloaded weights and the token, unless HF_HOME_OVERRIDE is given explicitly.
os.environ["HF_HOME"] = os.environ.get("HF_HOME_OVERRIDE", str(Path.home() / ".cache" / "huggingface"))

ROOT = SRC.parent
RESULTS = ROOT / "results"

# ------------------------------------------------------------------ prompts
# Question texts appended after the image slices. They start with "\n\n" like the CT notebook's
# query. Exact wording matters (the answers are wording-sensitive), so they are defined once here.
FINAL_YN = "Please provide your reasoning and conclude with 'Final Answer: yes' or 'Final Answer: no'."

Q_ENDO = ("\n\nBased on the visual evidence in the slices provided above, is there an ovarian "
          "endometrioma? " + FINAL_YN)

Q_SIDE = ("\n\nBased on the visual evidence in the slices provided above, on which side of the "
          "patient is the largest ovarian cystic lesion? Please provide your reasoning and conclude "
          "with 'Final Answer: left', 'Final Answer: right', 'Final Answer: bilateral' or "
          "'Final Answer: none'.")

Q_TYPE = ("\n\nBased on the visual evidence in the slices provided above, what is the most likely "
          "nature of the dominant ovarian cystic lesion? Please provide your reasoning and conclude "
          "with exactly one of 'Final Answer: endometrioma', 'Final Answer: simple cyst', "
          "'Final Answer: dermoid', 'Final Answer: hemorrhagic cyst' or 'Final Answer: none'.")


# ------------------------------------------------------------------ cases
def inventory() -> pd.DataFrame:
    """Inventory without the subjects in io_orient.EXCLUDED_SUBJECTS (PATCH.md, P01).

    derived/inventory.csv keeps every subject; rows with a non-empty `excluded` reason are removed
    here, so every case list below (and therefore every experiment) skips them."""
    inv = pd.read_csv(ROOT / "derived" / "inventory.csv")
    if "excluded" in inv:
        inv = inv[inv.excluded.isna() | (inv.excluded == "")]
    return inv


def d2_cases(require: tuple[str, ...] = ()) -> pd.DataFrame:
    """D2 subjects that have all sequences in `require`, with a simple label column.

    `require=("T2FS",)` keeps subjects whose `sequences` column (e.g. "T1;T1FS;T2;T2FS") lists T2FS;
    subjects without images or without that series drop out here.
    `label` is one of "endometrioma", "cyst", "negative". The order of the two assignments below
    matters: a subject with both an endometrioma and a cyst (D2-041) is labelled "endometrioma"."""
    inv = inventory()
    d = inv[inv.cohort == "D2"].copy()
    d = d[d.sequences.fillna("").apply(lambda s: all(q in s.split(";") for q in require))]
    d["label"] = "negative"
    d.loc[d.cy_present == True, "label"] = "cyst"
    d.loc[d.em_present == True, "label"] = "endometrioma"
    return d.reset_index(drop=True)


def cohort_cases(cohort: str, require: tuple[str, ...] = ()) -> pd.DataFrame:
    """Like d2_cases but for either cohort. D1 has no cyst labels; D1 endometrioma presence is
    the union over raters, and D1 subjects without any em label file are treated as negative
    (low confidence, see NOTEBOOK.md).

    The ground-truth side used by the side experiments comes from the inventory columns
    `em_side` / `cy_side` of these rows (side of the largest lesion, see inventory.lesion_summary)."""
    if cohort == "D2":
        return d2_cases(require)
    inv = inventory()
    d = inv[inv.cohort == cohort].copy()
    d = d[d.sequences.fillna("").apply(lambda s: all(q in s.split(";") for q in require))]
    d["label"] = np.where(d.em_present == True, "endometrioma", "negative")
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ results
def out_dir(name: str) -> Path:
    """Folder for one experiment's result CSVs (created if needed): normally results/<name>/."""
    # hardware-reproducibility reruns (VLM_RECHECK=<tag>) write to results/recheck/<tag>/<exp>/
    # RESULTS_SUBDIR=<dir>: write to results/<dir>/<exp>/ instead (e.g. the v2 prompt-format ablation, P05)
    tag = os.environ.get("VLM_RECHECK", "")
    sub = os.environ.get("RESULTS_SUBDIR", "")
    p = RESULTS / "recheck" / tag / name if tag else RESULTS / sub / name if sub else RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def src_hash() -> str:
    """sha1 over every .py file under src/ (path + contents): which code produced a result."""
    h = hashlib.sha1()
    for f in sorted(SRC.rglob("*.py")):
        h.update(str(f.relative_to(SRC)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def save(df: pd.DataFrame, path: Path):
    """Write a result CSV plus <name>.provenance.json (code hash, command, prompt format, time; P21)."""
    path = Path(path)
    df.to_csv(path, index=False)
    prov = {"csv": path.name, "rows": len(df), "src_sha1": src_hash(), "argv": sys.argv,
            "prompt_format": os.environ.get("VLM_PROMPT_FORMAT", "v1 (default)"),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    path.with_name(path.stem + ".provenance.json").write_text(json.dumps(prov, indent=2))
