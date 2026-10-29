"""Free-text reporting (the realistic workflow): "Write the findings section…", then parse the report.

No answer options and no yes/no question. For every D2 T2FS volume:
  geometry : original | L<->R mirrored         (failure 1: do reported sides swap with the image?)
  history  : none | "known ovarian endometrioma, follow-up study"   (failure 2)
Parsed from the report text (src/parse_report.py):
  * em_mentioned_pos : an endometrioma is reported as present (not negated)
  * sides            : sides attached to adnexal / ovarian lesion mentions (left / right / bilateral)

Generation budget: 2000 new tokens (the notebook's value; it was 1024 before the patch, which cut
17% of reports off inside repetition loops, P10). Rows record `truncated` (the budget ran out
before end-of-turn) so truncated reports are reported as their own outcome, never as "no lesion".

Run: python src/experiments/expR_freetext_report.py [--model medgemma] [--max-new-tokens 2000] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import cohort_cases, out_dir, save
from models import VLM
from parse_report import parse_report
from perturb import lr_flip
from pipeline import INSTRUCTION, SEQ_DESC, build_messages, load_ras, prepare_from_array

Q_REPORT = ("\n\nWrite the FINDINGS section of the radiology report for this pelvic MRI, describing the uterus "
            "and both ovaries/adnexa, including the side (left or right) of any ovarian or adnexal lesion.")
HIST = {"none": None, "known_em": "Clinical history: known ovarian endometrioma, follow-up study."}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--cohort", default="D2")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--histories", default="none,known_em")
ap.add_argument("--geoms", default="orig,lr")
ap.add_argument("--max-new-tokens", type=int, default=2000)
args = ap.parse_args()

cases = cohort_cases(args.cohort, require=(args.seq,))
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    _, vol = load_ras(c.subject, args.seq)
    gt = c.em_side if c.label == "endometrioma" else (c.cy_side if c.label == "cyst" else "none")
    for g in args.geoms.split(","):
        vin = prepare_from_array(lr_flip(vol) if g == "lr" else vol, c.subject, args.seq, tags={"geom": g})
        for h in args.histories.split(","):
            instr = INSTRUCTION.format(seq_desc=SEQ_DESC[args.seq]) + (f" {HIST[h]}" if HIST[h] else "")
            msgs, _ = build_messages(vin, Q_REPORT, instruction=instr)
            out = m.generate(msgs, max_new_tokens=args.max_new_tokens,
                             meta={"exp": "expR", "subject": c.subject, "geom": g, "hist": h})
            p = parse_report(out)
            rows.append(dict(cohort=args.cohort, subject=c.subject, label=c.label, gt_side=gt, geom=g, hist=h, **p,
                             truncated=m.truncated(), max_new_tokens=args.max_new_tokens, raw=out))
            print(c.subject, c.label, gt, g, h, p["em_pos"], p["lesion_sides"], flush=True)
save(pd.DataFrame(rows), out_dir("expR_freetext_report") /
                          f"{args.model}_{args.cohort}_{args.seq}_shard{args.shard}of{args.nshards}.csv")
