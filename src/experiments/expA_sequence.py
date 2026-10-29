"""Lead A: does the endometrioma call track the image contrast or the sequence NAME in the prompt?

2x3 design per D2 case (paired):
  pixels shown  : T1FS | T2FS          (D2 sequences are registered in world space)
  name in prompt: true | swapped | none ("MRI")
Questions: Q_ENDO (yes/no + logit) and Q_TYPE (endometrioma / simple cyst / dermoid / ...).
Swapped = T1FS pixels described as "T2-weighted fat-suppressed" and vice versa.

Physics: endometrioma = T1FS hyperintense; simple cyst = T1FS hypointense, T2 bright.
If decisions follow the NAME rather than the PIXELS, the model is reading metadata, not contrast.

Run: python src/experiments/expA_sequence.py [--model medgemma] [--shard i --nshards n] [--neg 10]
"""
import argparse

import pandas as pd

from common import Q_ENDO, Q_TYPE, d2_cases, out_dir, save
from models import VLM
from parse import parse_lesion_type, parse_yesno
from pipeline import SEQ_DESC, build_messages, prepare

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--neg", type=int, default=10, help="number of lesion-free negatives to include")
ap.add_argument("--no-generate", action="store_true", help="logit scoring only (fast)")
args = ap.parse_args()

SWAP = {"T1FS": "T2FS", "T2FS": "T1FS"}
cases = d2_cases(require=("T1FS", "T2FS"))
cases = pd.concat([cases[cases.label != "negative"], cases[cases.label == "negative"].head(args.neg)])
cases = cases.reset_index(drop=True)

m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    for pix in ["T1FS", "T2FS"]:
        vin = prepare(c.subject, pix)
        for name_cond, desc in [("true", SEQ_DESC[pix]), ("swapped", SEQ_DESC[SWAP[pix]]), ("none", "")]:
            meta = {"exp": "expA", "subject": c.subject, "pix": pix, "name": name_cond}
            msgs, _ = build_messages(vin, Q_ENDO, seq_desc=desc)
            sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
            row = dict(subject=c.subject, label=c.label, pix=pix, name=name_cond, described_as=desc or "(none)",
                       logit_yes_no=sc[" yes"] - sc[" no"])
            if not args.no_generate:
                out = m.generate(msgs, meta={**meta, "q": "endo"})
                msgs_t, _ = build_messages(vin, Q_TYPE, seq_desc=desc)
                out_t = m.generate(msgs_t, meta={**meta, "q": "type"})
                row.update(endo=parse_yesno(out)[0], lesion_type=parse_lesion_type(out_t)[0],
                           endo_raw=out, type_raw=out_t)
            rows.append(row)
            print(c.subject, c.label, pix, name_cond, round(row["logit_yes_no"], 2), row.get("endo"),
                  row.get("lesion_type"), flush=True)
save(pd.DataFrame(rows), out_dir("expA_sequence") / f"{args.model}_shard{args.shard}of{args.nshards}.csv")
