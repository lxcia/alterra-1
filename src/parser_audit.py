"""Parser audit: dump a random sample of cached generations with their parsed answers into
results/parser_audit.csv for hand checking (column `hand_label` is filled in by a human).

Run: python src/parser_audit.py [--n 60] [--seed 0]
"""
import argparse
import glob
import json
import random
from pathlib import Path

import pandas as pd

from parse import parse_lesion_type, parse_side, parse_yesno

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

recs = []
for f in glob.glob(str(ROOT / "results" / "cache" / "*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["kind"] != "generate":
            continue
        q = r["msgs"][-1]["content"][-1]["text"]
        if "'Final Answer: left'" in q:
            kind, p = "side", parse_side(r["output"])
        elif "'Final Answer: endometrioma'" in q:
            kind, p = "type", parse_lesion_type(r["output"])
        else:
            kind, p = "yesno", parse_yesno(r["output"])
        recs.append(dict(key=r["key"][:12], model=r["model"], kind=kind, parsed=p[0], method=p[1],
                         tail=r["output"][-250:].replace("\n", " "), hand_label=""))
random.Random(args.seed).shuffle(recs)
df = pd.DataFrame(recs[: args.n])
df.to_csv(ROOT / "results" / "parser_audit.csv", index=False)
print(f"{len(recs)} generations in cache; wrote {len(df)} to results/parser_audit.csv")
print(pd.DataFrame(recs).groupby(["kind", "method"]).size())
