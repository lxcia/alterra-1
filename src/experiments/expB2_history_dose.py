"""Lead B follow-up: dose-response, paraphrase robustness, and generated-answer confirmation.

Histories (appended to the reference instruction), ordered by strength of the prior:
  none < possible < suspected < history_endo < known_em < known_large_bilateral
Plus 3 paraphrases of 'known endometrioma' and a realistic post-surgical scenario
('endometrioma resected 6 months ago'), where the correct read of a lesion-free pelvis is 'no'.

Outcomes:
  * logit yes-no (direct answer) for every history x {real, blank} image;
  * generated reason-then-answer (the reference output format) for a subset of histories on
    real images, so the failure is shown in the notebook-style output, not just in logits.

Run: python src/experiments/expB2_history_dose.py [--model medgemma] [--seq T2FS] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import Q_ENDO, cohort_cases, out_dir, save
from models import VLM
from parse import parse_yesno
from perturb import blank_like
from pipeline import INSTRUCTION, SEQ_DESC, build_messages, load_ras, prepare_from_array

HIST = {
    "none": None,
    "possible": "Clinical history: possible endometriosis.",
    "suspected": "Clinical history: suspected endometriosis.",
    "history_endo": "Clinical history: history of endometriosis.",
    "known_em": "Clinical history: known ovarian endometrioma, follow-up study.",
    "known_large_bilat": "Clinical history: known large bilateral ovarian endometriomas, follow-up study.",
    "known_em_p1": "Follow-up of known ovarian endometrioma.",
    "known_em_p2": "Patient with a previously diagnosed endometrioma of the ovary, here for follow-up imaging.",
    "known_em_p3": "Indication: endometrioma, follow-up.",
    "resected": "Clinical history: ovarian endometrioma surgically resected 6 months ago, follow-up study.",
}
GENERATE_FOR = ["none", "suspected", "known_em", "known_em_p2", "resected"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--no-generate", action="store_true")
ap.add_argument("--cohort", default="D2")
ap.add_argument("--reverse", action="store_true", help="traverse cases in reverse (extra parallel worker)")
args = ap.parse_args()

cases = cohort_cases(args.cohort, require=(args.seq,))
m = VLM(args.model)
rows = []
order = list(cases.iterrows())[::-1] if args.reverse else list(cases.iterrows())
for i, c in order:
    if i % args.nshards != args.shard:
        continue
    m.refresh_cache()
    _, vol = load_ras(c.subject, args.seq)
    vins = {"real": prepare_from_array(vol, c.subject, args.seq),
            "blank": prepare_from_array(blank_like(vol), c.subject, args.seq, tags={"abl": "blank"})}
    for abl, vin in vins.items():
        for h, text in HIST.items():
            instr = INSTRUCTION.format(seq_desc=SEQ_DESC[args.seq]) + (f" {text}" if text else "")
            msgs, _ = build_messages(vin, Q_ENDO, instruction=instr)
            meta = {"exp": "expB2", "subject": c.subject, "seq": args.seq, "abl": abl, "hist": h}
            sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
            row = dict(cohort=args.cohort, subject=c.subject, label=c.label, em_volume_ml=c.em_volume_ml, seq=args.seq, abl=abl,
                       hist=h, logit_yes_no=sc[" yes"] - sc[" no"])
            if abl == "real" and h in GENERATE_FOR and not args.no_generate:
                out = m.generate(msgs, meta=meta)
                row.update(endo=parse_yesno(out)[0], endo_raw=out, truncated=m.truncated())
            rows.append(row)
    print(c.subject, c.label, {r["hist"]: r.get("endo") for r in rows if r["subject"] == c.subject and "endo" in r},
          flush=True)
save(pd.DataFrame(rows), out_dir("expB2_history_dose") / f"{args.model}_{'' if args.cohort == 'D2' else args.cohort + '_'}{args.seq}_shard{args.shard}of{args.nshards}{'_rev' if args.reverse else ''}.csv")
