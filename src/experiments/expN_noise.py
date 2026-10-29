"""Noise floors that every candidate failure must clearly exceed.

1. Paraphrase variance: 5 paraphrases of the endometrioma question and 5 of the side question
   (scored; the reference wording is paraphrase 0).
2. Sampling variance: temperature 1.0, 5 seeds, generated answers on a subset of cases
   (default first 20 cases).

Run: python src/experiments/expN_noise.py [--model medgemma] [--seq T2FS] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import FINAL_YN, Q_ENDO, Q_SIDE, d2_cases, out_dir, save
from models import VLM
from parse import parse_side, parse_yesno
from pipeline import build_messages, prepare

ENDO_PARA = [
    Q_ENDO,
    "\n\nLooking at the slices above, do you see an endometrioma in either ovary? " + FINAL_YN,
    "\n\nIs an ovarian endometrioma present in this pelvic MRI? Base your answer on the slices above. " + FINAL_YN,
    "\n\nReview the images above. Does this patient have an endometrioma of the ovary? " + FINAL_YN,
    "\n\nFrom the provided slices, determine whether there is an endometriotic cyst (endometrioma) of the ovary. " + FINAL_YN,
]
SIDE_TAIL = ("Please provide your reasoning and conclude with 'Final Answer: left', 'Final Answer: right', "
             "'Final Answer: bilateral' or 'Final Answer: none'.")
SIDE_PARA = [
    Q_SIDE,
    "\n\nLooking at the slices above, which side of the patient's pelvis contains the largest ovarian cyst? " + SIDE_TAIL,
    "\n\nIn the images above, is the dominant adnexal cystic lesion on the patient's left or right? " + SIDE_TAIL,
    "\n\nReview the slices above and state the laterality of the largest ovarian cystic lesion. " + SIDE_TAIL,
    "\n\nFrom the provided slices, on which side (patient's left or right) is the biggest cystic ovarian lesion? " + SIDE_TAIL,
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n-sample-cases", type=int, default=20)
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

cases = d2_cases(require=(args.seq,))
# sampling subset: all lesion cases first, then negatives
order = pd.concat([cases[cases.label != "negative"], cases[cases.label == "negative"]])
sample_subjects = set(order.subject.head(args.n_sample_cases))
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    vin = prepare(c.subject, args.seq)
    for k, q in enumerate(ENDO_PARA):
        msgs, _ = build_messages(vin, q)
        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta={"exp": "expN", "para": k})
        rows.append(dict(subject=c.subject, label=c.label, kind="para_endo", idx=k, logit_yes_no=sc[" yes"] - sc[" no"]))
    for k, q in enumerate(SIDE_PARA):
        msgs, _ = build_messages(vin, q)
        sc = m.score(msgs, [" left", " right", " bilateral", " none"], assistant_prefix="Final Answer:",
                     meta={"exp": "expN", "para_side": k})
        rows.append(dict(subject=c.subject, label=c.label, kind="para_side", idx=k,
                         side_argmax=max(sc, key=sc.get).strip(), **{f"lp_{a.strip()}": b for a, b in sc.items()}))
    if c.subject in sample_subjects:
        for q, kind, parser in [(Q_ENDO, "sample_endo", parse_yesno), (Q_SIDE, "sample_side", parse_side)]:
            msgs, _ = build_messages(vin, q)
            for s in range(args.seeds):
                out = m.generate(msgs, do_sample=True, temperature=1.0, seed=s, meta={"exp": "expN", "seed": s})
                rows.append(dict(subject=c.subject, label=c.label, kind=kind, idx=s, answer=parser(out)[0], raw=out))
    print(c.subject, c.label, flush=True)
save(pd.DataFrame(rows), out_dir("expN_noise") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")
