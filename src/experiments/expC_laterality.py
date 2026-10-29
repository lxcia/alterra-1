"""Lead C: laterality equivariance.

For each D2 case: ask which side the largest ovarian cystic lesion is on, for
  geometry  : original | LR-mirrored (patient left<->right) | AP-mirrored (control: side should NOT change)
  convention: none | correct statement of radiological convention | WRONG statement (neurological)
Label-free metric: does the stated side flip with an LR mirror (equivariance)?
Accuracy metric (lesion cases): agreement with the mask-derived side (world coordinates).

Outcome recorded both as generated free text (parsed) and as next-token log-probs of
' left' / ' right' / ' bilateral' / ' none' after 'Final Answer:'.

Run: python src/experiments/expC_laterality.py [--model medgemma] [--cohort D2] [--seq T2FS] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import Q_SIDE, cohort_cases, out_dir, save
from models import VLM
from parse import parse_side
from perturb import ap_flip, lr_flip
from pipeline import INSTRUCTION, SEQ_DESC, build_messages, load_ras, prepare_from_array

CONVENTIONS = {
    "none": "",
    "radiological": (" The images are displayed in standard radiological convention: the patient's "
                     "right side appears on the left side of each image."),
    "wrong": (" The images are displayed in neurological convention: the patient's right side "
              "appears on the right side of each image."),
}
GEOMS = {"orig": lambda v: v, "lr": lr_flip, "ap": ap_flip}
CANDS = [" left", " right", " bilateral", " none"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--conventions", default="none,radiological,wrong")
ap.add_argument("--geoms", default="orig,lr,ap")
ap.add_argument("--generate", action="store_true")
ap.add_argument("--lesion-only", action="store_true")
ap.add_argument("--cohort", default="D2")
args = ap.parse_args()

cases = cohort_cases(args.cohort, require=(args.seq,))
if args.lesion_only:
    cases = cases[cases.label != "negative"].reset_index(drop=True)
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    _, vol = load_ras(c.subject, args.seq)
    gt = c.em_side if c.label == "endometrioma" else (c.cy_side if c.label == "cyst" else "none")
    for g in args.geoms.split(","):
        vin = prepare_from_array(GEOMS[g](vol), c.subject, args.seq, tags={"geom": g})
        for conv in args.conventions.split(","):
            instr = INSTRUCTION.format(seq_desc=SEQ_DESC[args.seq]) + CONVENTIONS[conv]
            msgs, _ = build_messages(vin, Q_SIDE, instruction=instr)
            meta = {"exp": "expC", "subject": c.subject, "seq": args.seq, "geom": g, "conv": conv}
            sc = m.score(msgs, CANDS, assistant_prefix="Final Answer:", meta=meta)
            row = dict(cohort=args.cohort, subject=c.subject, label=c.label, gt_side=gt, seq=args.seq, geom=g, conv=conv,
                       **{f"lp_{k.strip()}": v for k, v in sc.items()},
                       side_argmax=max(sc, key=sc.get).strip())
            if args.generate:
                out = m.generate(msgs, meta=meta)
                row.update(side=parse_side(out)[0], side_raw=out, truncated=m.truncated())
            rows.append(row)
            print(c.subject, c.label, gt, g, conv, row["side_argmax"], row.get("side"), flush=True)
save(pd.DataFrame(rows), out_dir("expC_laterality") / f"{args.model}_{'' if args.cohort == 'D2' else args.cohort + '_'}{args.seq}_shard{args.shard}of{args.nshards}.csv")
