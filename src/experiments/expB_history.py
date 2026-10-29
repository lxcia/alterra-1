"""Lead B: clinical-history anchoring.

Same volume, only the clinical history sentence (appended to the instruction) varies.
Primary outcome: false-positive 'endometrioma: yes' on endometrioma-NEGATIVE D2 cases
(lesion-free and simple-cyst cases), plus the continuous logit log P(yes) - log P(no).

Image ablations (how much of the answer comes from the prior alone):
  real = the patient's own volume; blank = zeros; noise = uniform noise; text = no image at all.
Endometrioma-positive cases are included so that sensitivity under each history is also known.

Run: python src/experiments/expB_history.py [--model medgemma] [--seq T2FS] [--shard i --nshards n]
"""
import argparse

import numpy as np
import pandas as pd

from common import Q_ENDO, d2_cases, out_dir, save
from models import VLM
from parse import parse_yesno
from perturb import blank_like, noise_like
from pipeline import INSTRUCTION, SEQ_DESC, build_messages, load_ras, prepare_from_array

HISTORIES = {
    "none": None,
    "knee": "Clinical history: knee pain.",
    "pain": "Clinical history: chronic pelvic pain, no prior imaging.",
    "suspected": "Clinical history: suspected endometriosis.",
    "known_em": "Clinical history: known ovarian endometrioma, follow-up study.",
    "known_em_left": "Clinical history: known left ovarian endometrioma, follow-up study.",
    "known_em_right": "Clinical history: known right ovarian endometrioma, follow-up study.",
    "no_em": "Clinical history: prior imaging showed no endometrioma.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--ablations", default="real,blank,noise,text")
ap.add_argument("--histories", default=",".join(HISTORIES))
ap.add_argument("--generate", action="store_true", help="also generate free-text answers (slow)")
args = ap.parse_args()

cases = d2_cases(require=(args.seq,))
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    _, vol = load_ras(c.subject, args.seq)
    vins = {"real": prepare_from_array(vol, c.subject, args.seq),
            "blank": prepare_from_array(blank_like(vol), c.subject, args.seq, tags={"abl": "blank"}),
            "noise": prepare_from_array(noise_like(vol, seed=i), c.subject, args.seq, tags={"abl": "noise"}),
            "text": None}
    for abl in args.ablations.split(","):
        for h in args.histories.split(","):
            hist = HISTORIES[h]
            instr = INSTRUCTION.format(seq_desc=SEQ_DESC[args.seq]) + (f" {hist}" if hist else "")
            if abl == "text":
                instr = instr.replace("Please review the slices provided below carefully.",
                                      "The images are not available.")
            msgs, _ = build_messages(vins[abl], Q_ENDO, instruction=instr)
            meta = {"exp": "expB", "subject": c.subject, "seq": args.seq, "abl": abl, "hist": h}
            sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
            row = dict(subject=c.subject, label=c.label, em_side=c.em_side, seq=args.seq, abl=abl, hist=h,
                       logit_yes_no=sc[" yes"] - sc[" no"])
            if args.generate:
                out = m.generate(msgs, meta=meta)
                row.update(endo=parse_yesno(out)[0], endo_raw=out)
            rows.append(row)
        print(c.subject, c.label, abl, {r["hist"]: round(r["logit_yes_no"], 1) for r in rows[-len(args.histories.split(",")):]},
              flush=True)
save(pd.DataFrame(rows), out_dir("expB_history") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")
