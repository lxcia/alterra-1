"""Failure 2 vs the noise floor: does the history effect hold under EVERY question paraphrase?

The endometrioma question is itself wording-sensitive (expN: direct-answer 'yes' ranges 3%-70%
across 5 paraphrases). To show the history effect is not just more of that fragility, cross
5 question paraphrases (expN.ENDO_PARA) x {no history, 'known ovarian endometrioma, follow-up',
'endometrioma resected 6 months ago'} on every D2 T2FS volume.
Outcome: direct-answer logit (all cells) + generated answers for paraphrases 1-4 under
none / known_em (paraphrase 0 = reference, already generated in expB2).

Run: python src/experiments/expB3_history_x_paraphrase.py [--model medgemma] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import FINAL_YN, d2_cases, out_dir, save
from models import VLM
from parse import parse_yesno
from pipeline import INSTRUCTION, SEQ_DESC, build_messages, prepare

# identical to expN.ENDO_PARA (kept literal here so this script does not import argparse-running modules)
Q_ENDO = ("\n\nBased on the visual evidence in the slices provided above, is there an ovarian "
          "endometrioma? " + FINAL_YN)
ENDO_PARA = [
    Q_ENDO,
    "\n\nLooking at the slices above, do you see an endometrioma in either ovary? " + FINAL_YN,
    "\n\nIs an ovarian endometrioma present in this pelvic MRI? Base your answer on the slices above. " + FINAL_YN,
    "\n\nReview the images above. Does this patient have an endometrioma of the ovary? " + FINAL_YN,
    "\n\nFrom the provided slices, determine whether there is an endometriotic cyst (endometrioma) of the ovary. " + FINAL_YN,
]
HIST = {"none": None,
        "known_em": "Clinical history: known ovarian endometrioma, follow-up study.",
        "resected": "Clinical history: ovarian endometrioma surgically resected 6 months ago, follow-up study."}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--no-generate", action="store_true")
args = ap.parse_args()

cases = d2_cases(require=(args.seq,))
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    vin = prepare(c.subject, args.seq)
    for k, q in enumerate(ENDO_PARA):
        for h, text in HIST.items():
            instr = INSTRUCTION.format(seq_desc=SEQ_DESC[args.seq]) + (f" {text}" if text else "")
            msgs, _ = build_messages(vin, q, instruction=instr)
            meta = {"exp": "expB3", "subject": c.subject, "para": k, "hist": h}
            sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
            row = dict(subject=c.subject, label=c.label, para=k, hist=h, logit_yes_no=sc[" yes"] - sc[" no"])
            if k > 0 and h in ("none", "known_em") and not args.no_generate:
                out = m.generate(msgs, meta=meta)
                row.update(endo=parse_yesno(out)[0], endo_raw=out)
            rows.append(row)
    print(c.subject, c.label, flush=True)
save(pd.DataFrame(rows), out_dir("expB3_history_x_paraphrase") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")
