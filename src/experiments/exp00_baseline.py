"""Exp 00: baseline MedGemma performance on D2 (context only, NOT a failure claim).

For every D2 subject and each available sequence in {T1FS, T2FS}: reference pipeline,
ask (a) endometrioma yes/no, (b) side of the largest ovarian cystic lesion. Also records
the continuous score log P(' yes') - log P(' no') after 'Final Answer:'.

Run: python src/experiments/exp00_baseline.py [--model medgemma] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import Q_ENDO, Q_SIDE, d2_cases, out_dir, save
from models import VLM
from parse import parse_side, parse_yesno
from pipeline import build_messages, prepare

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
args = ap.parse_args()

m = VLM(args.model)
cases = d2_cases()
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    for seq in ["T1FS", "T2FS"]:
        if seq not in str(c.sequences).split(";"):
            continue
        vin = prepare(c.subject, seq)
        msgs, _ = build_messages(vin, Q_ENDO)
        out = m.generate(msgs, meta={"exp": "exp00", "subject": c.subject, "seq": seq, "q": "endo"})
        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:")
        msgs_s, _ = build_messages(vin, Q_SIDE)
        out_s = m.generate(msgs_s, meta={"exp": "exp00", "subject": c.subject, "seq": seq, "q": "side"})
        rows.append(dict(subject=c.subject, label=c.label, em_side=c.em_side, cy_side=c.cy_side, seq=seq,
                         n_slices=len(vin.slices), endo=parse_yesno(out)[0], endo_method=parse_yesno(out)[1],
                         logit_yes_no=sc[" yes"] - sc[" no"], side=parse_side(out_s)[0],
                         endo_raw=out, side_raw=out_s))
        print(c.subject, c.label, seq, rows[-1]["endo"], round(rows[-1]["logit_yes_no"], 2), rows[-1]["side"],
              flush=True)
save(pd.DataFrame(rows), out_dir("exp00_baseline") / f"{args.model}_shard{args.shard}of{args.nshards}.csv")
