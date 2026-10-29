"""Candidate D confirmation: does an incoherent (shuffled) slice stack create endometrioma calls in
the GENERATED (reason-first) answer, and is the 'lesion slices only' effect just a short-stack effect?

Per D2 T2FS volume (all lesion cases + the first --neg lesion-free cases):
  ref, shuffled (seeds 0, 1, 2)                       -> generated yes/no
  short_block: a random contiguous block of k slices,  -> generated yes/no
     k = 6 (typical lesion_only length), lesion-free volumes only
     (slice-count-matched control for expD's 'lesion_only')

Run: python src/experiments/expD2_shuffle_confirm.py [--model medgemma] [--neg 15]
"""
import argparse

import numpy as np
import pandas as pd

from common import Q_ENDO, d2_cases, out_dir, save
from models import VLM
from parse import parse_yesno
from perturb import keep_z, shuffle_order
from pipeline import build_messages, prepare

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--neg", type=int, default=15)
ap.add_argument("--block", type=int, default=6)
args = ap.parse_args()

cases = d2_cases(require=(args.seq,))
cases = pd.concat([cases[cases.label != "negative"], cases[cases.label == "negative"].head(args.neg)]).reset_index(drop=True)
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    ref = prepare(c.subject, args.seq)
    conds = {"ref": ref, **{f"shuffled{s}": shuffle_order(ref, s) for s in range(3)}}
    if c.label == "negative":
        rng = np.random.default_rng(i)
        start = int(rng.integers(0, len(ref.z_indices) - args.block))
        conds["short_block"] = keep_z(ref, set(ref.z_indices[start:start + args.block]))
    for name, vin in conds.items():
        msgs, _ = build_messages(vin, Q_ENDO)
        out = m.generate(msgs, meta={"exp": "expD2", "subject": c.subject, "cond": name})
        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta={"exp": "expD2", "cond": name})
        rows.append(dict(subject=c.subject, label=c.label, cond=name, n_shown=len(vin.slices),
                         endo=parse_yesno(out)[0], logit_yes_no=sc[" yes"] - sc[" no"], raw=out))
        print(c.subject, c.label, name, rows[-1]["endo"], round(rows[-1]["logit_yes_no"], 2), flush=True)
save(pd.DataFrame(rows), out_dir("expD2_shuffle_confirm") / f"{args.model}_{args.seq}.csv")
