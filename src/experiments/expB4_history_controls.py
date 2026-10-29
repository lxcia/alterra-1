"""Failure 2 controls requested in the rigor review (NEXT_STEPS items 3 and 4).

Parts (D2 T2FS endometrioma-NEGATIVE volumes; generated answers unless stated):
  irrelevant : histories "knee pain" and "chronic pelvic pain, no prior imaging", on every volume.
               Lets the known-history flip be tested against an IRRELEVANT-history flip among the
               same baseline-"no" volumes (controls for "any prompt change flips answers").
  doses      : "possible endometriosis", "history of endometriosis", "known large bilateral
               endometriomas", so the dose-response exists in generated answers, not only logits.
  blanks     : real per-patient variation in the 'no image content' input, x {none, known_em}:
               zeros_matched (all-zero, patient's own slice count/shape), gray_matched (uniform at
               the patient's median intensity), pelvis_blanked (patient's own volume with an
               ellipse covering the central 70% of the in-plane FOV filled with the median body
               intensity: body outline kept, pelvic contents erased).
  indeterminate : endometrioma question with a third option 'cannot assess', options in two
               orders, x {real, zeros_matched} x {none, known_em}.

Run: python src/experiments/expB4_history_controls.py [--parts irrelevant,doses,blanks,indeterminate] [--shard i --nshards n]
"""
import argparse

import numpy as np
import pandas as pd

from common import Q_ENDO, d2_cases, out_dir, save
from models import VLM
from parse import parse_yesno
from perturb import pelvis_blanked
from pipeline import INSTRUCTION, SEQ_DESC, build_messages, load_ras, prepare_from_array

HIST = {
    "none": None,
    "knee": "Clinical history: knee pain.",
    "pain": "Clinical history: chronic pelvic pain, no prior imaging.",
    "possible": "Clinical history: possible endometriosis.",
    "history_endo": "Clinical history: history of endometriosis.",
    "known_em": "Clinical history: known ovarian endometrioma, follow-up study.",
    "known_large_bilat": "Clinical history: known large bilateral ovarian endometriomas, follow-up study.",
}
Q_IND = {
    "yn_first": ("\n\nBased on the visual evidence in the slices provided above, is there an ovarian endometrioma? "
                 "Please provide your reasoning and conclude with 'Final Answer: yes', 'Final Answer: no' or "
                 "'Final Answer: cannot assess'."),
    "ca_first": ("\n\nBased on the visual evidence in the slices provided above, is there an ovarian endometrioma? "
                 "Please provide your reasoning and conclude with 'Final Answer: cannot assess', 'Final Answer: no' "
                 "or 'Final Answer: yes'."),
}


def parse_ind(out: str) -> str:
    import re
    hits = re.findall(r"final\s*answer\s*[:\-]?\s*\**\s*([a-z ]+)", out or "", re.I)
    if not hits:
        return "unparsed"
    t = hits[-1].strip().lower()
    if t.startswith("cannot") or t.startswith("indeterminate"):
        return "cannot assess"
    return parse_yesno(out)[0]


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--parts", default="irrelevant,doses,blanks,indeterminate")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n-blanks", type=int, default=30, help="em-negative volumes used for the blank variants")
ap.add_argument("--n-ind", type=int, default=20, help="em-negative volumes used for the indeterminate option")
args = ap.parse_args()
parts = args.parts.split(",")

# Every test here concerns endometrioma-NEGATIVE volumes (false positives), so only those are run.
cases = d2_cases(require=(args.seq,))
cases = cases[cases.label != "endometrioma"].reset_index(drop=True)
m = VLM(args.model)
rows = []


def instr(h):
    return INSTRUCTION.format(seq_desc=SEQ_DESC[args.seq]) + (f" {HIST[h]}" if HIST[h] else "")


for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    _, vol = load_ras(c.subject, args.seq)
    real = prepare_from_array(vol, c.subject, args.seq)
    base = dict(subject=c.subject, label=c.label)

    def ask(part, image, h, vin, q=Q_ENDO, parser=lambda o: parse_yesno(o)[0], qname="ref"):
        msgs, _ = build_messages(vin, q, instruction=instr(h))
        meta = {"exp": "expB4", "subject": c.subject, "part": part, "image": image, "hist": h, "q": qname}
        out = m.generate(msgs, meta=meta)
        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
        rows.append(dict(**base, part=part, image=image, hist=h, q=qname, answer=parser(out),
                         logit_yes_no=sc[" yes"] - sc[" no"], input_sha=vin.digest(), raw=out))

    if "irrelevant" in parts:
        for h in ["none", "knee", "pain"]:
            ask("irrelevant", "real", h, real)
    if "known" in parts:  # the 'known' arm (for MedGemma it came from expB2; other models need it here)
        ask("known", "real", "known_em", real)
    if "doses" in parts:
        for h in ["possible", "history_endo", "known_large_bilat"]:
            ask("doses", "real", h, real)
    if "blanks" in parts and i < args.n_blanks:
        med = float(np.median(vol))
        variants = {"zeros_matched": np.zeros_like(vol), "gray_matched": np.full_like(vol, med),
                    "pelvis_blanked": pelvis_blanked(vol)}
        for name, v in variants.items():
            # gray/zeros are constant volumes: give the window a range so they render as one grey level
            vin = prepare_from_array(v, c.subject, args.seq, tags={"blank": name},
                                     p_lo=0, p_hi=100) if name != "pelvis_blanked" else \
                prepare_from_array(v, c.subject, args.seq, tags={"blank": name})
            if name == "gray_matched":
                vin.slices = [np.full_like(s, 128) for s in vin.slices]
            for h in ["none", "known_em"]:
                ask("blanks", name, h, vin)
    if "indeterminate" in parts and i < args.n_ind:
        zeros = prepare_from_array(np.zeros_like(vol), c.subject, args.seq, tags={"blank": "zeros"})
        for image, vin in [("real", real), ("zeros_matched", zeros)]:
            for h in ["none", "known_em"]:
                for qn, q in Q_IND.items():
                    ask("indeterminate", image, h, vin, q=q, parser=parse_ind, qname=qn)
    print(c.subject, c.label, {(r["part"], r["image"], r["hist"], r["q"]): r["answer"] for r in rows if r["subject"] == c.subject},
          flush=True)
parts_tag = "" if args.parts == "irrelevant,doses,blanks,indeterminate" else "_" + "-".join(parts)
save(pd.DataFrame(rows), out_dir("expB4_history_controls") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}{parts_tag}.csv")
