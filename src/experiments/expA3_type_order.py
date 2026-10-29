"""Lead A / answer-format mechanism: is the lesion-type answer driven by OPTION POSITION?

Q_TYPE lists 5 options (endometrioma, simple cyst, dermoid, hemorrhagic cyst, none). We rotate
the list so each option appears first exactly once (5 prompts per volume) and record the
generated answer. If the chosen option follows its position rather than its content, the
answer is a format artefact, not a reading of the image.
Cases: every D2 endometrioma or cyst, on T1FS and T2FS where available.

Run: python src/experiments/expA3_type_order.py [--model medgemma] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import d2_cases, out_dir, save
from models import VLM
from parse import parse_lesion_type
from pipeline import build_messages, prepare

OPTIONS = ["endometrioma", "simple cyst", "dermoid", "hemorrhagic cyst", "none"]


def q_type(opts):
    listed = ", ".join(f"'Final Answer: {o}'" for o in opts[:-1]) + f" or 'Final Answer: {opts[-1]}'"
    return ("\n\nBased on the visual evidence in the slices provided above, what is the most likely nature of "
            "the dominant ovarian cystic lesion? Please provide your reasoning and conclude with exactly one of "
            + listed + ".")


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--seqs", default="T2FS,T1FS")
args = ap.parse_args()

cases = d2_cases()
cases = cases[cases.label != "negative"].reset_index(drop=True)
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    for seq in args.seqs.split(","):
        if seq not in str(c.sequences).split(";"):
            continue
        vin = prepare(c.subject, seq)
        for r in range(len(OPTIONS)):
            opts = OPTIONS[r:] + OPTIONS[:r]
            msgs, _ = build_messages(vin, q_type(opts))
            out = m.generate(msgs, meta={"exp": "expA3", "subject": c.subject, "seq": seq, "rot": r})
            ans = parse_lesion_type(out)[0]
            rows.append(dict(subject=c.subject, label=c.label, seq=seq, rot=r, first_option=opts[0], answer=ans,
                             answer_position=opts.index(ans) if ans in opts else -1, raw=out))
            print(c.subject, c.label, seq, "first=" + opts[0], "->", ans, flush=True)
save(pd.DataFrame(rows), out_dir("expA3_type_order") / f"{args.model}_shard{args.shard}of{args.nshards}.csv")
