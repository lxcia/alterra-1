"""Lead M: Google's own MRI evaluation prompt ("MRI-US1").

The MedGemma 1.5 Technical Report (arXiv:2604.05081, Table 13) evaluates 3D MRI with:

    {{ images }} + After looking at the patient history "history_text", Is there {{label}} in the
    MRI volume? You may write out your argument before stating your final answer
    "Final Answer: yes" or "Final Answer: no"

i.e. only the slices, then one text item; no instruction, no "SLICE n" labels. (The report's table
has stray quote characters around this text, probably a typesetting artefact; we use plain
double quotes, as in its CT-US1 prompt.) Training prompts also contained the report's indication
(§2.3.1), so this format is the closest we can get to what the model was tuned on.

Questions:
  1. Does the official format change the endometrioma call compared with our notebook-style prompt
     (instruction + "SLICE n" + Q_ENDO)? -> layout "us1" vs "notebook" (same MRI-US1 question text in
     both, so only the layout differs) and our original Q_ENDO prompt as reference ("ours").
  2. Does Failure 2 (history drives the answer) hold in the format the model was evaluated with?
     -> history conditions below, paired on the same volume.
  3. Does Failure 1 (default "left") hold? The template is yes/no only, so side is asked as two
     labels, "left ovarian endometrioma" and "right ovarian endometrioma"; a side-blind model gives
     the same logit to both.
Image ablation "blank" (all-zero volume) measures how much of the answer is the prior alone.

Outcome per read: logit = log P(" yes") - log P(" no") after "Final Answer:" (models.VLM.score);
--generate also stores the free-text answer and its parse (slow).

Before running, copy into this repo (as for the other experiments): src/models.py, src/parse.py,
src/perturb.py. Uses pipeline.py's min-max normalisation (changed 2026-10-26), so results are not
directly comparable with expB results made with the old 0.5-99.5 window; rerun expB with the same
code before comparing.

Size (defaults): 68 D2 T2FS cases x 2 ablations x 35 prompts (us1 and notebook: 3 labels x 5
histories each; ours: 5 histories) = 4,760 score calls (one forward pass each). Use --shard/--nshards to split across GPUs.

Run: python src/experiments/expM_mri_us1.py [--model medgemma] [--seq T2FS] [--shard i --nshards n]
     [--layouts us1,notebook,ours] [--histories ...] [--labels ...] [--ablations real,blank] [--generate]
Output: results/expM_mri_us1/<model>_<seq>_shard<i>of<n>.csv
"""
import argparse

import pandas as pd

from common import Q_ENDO, d2_cases, out_dir, save
from models import VLM
from parse import parse_yesno
from perturb import blank_like
from pipeline import INSTRUCTION, SEQ_DESC, VolumeInput, build_messages, load_ras, prepare_from_array

# Table 13 template, "{history}" and "{label}" filled below. No leading "\n\n": in the official
# layout the text follows the last image directly.
US1 = ('After looking at the patient history "{history}", Is there {label} in the MRI volume? '
       'You may write out your argument before stating your final answer "Final Answer: yes" or '
       '"Final Answer: no"')

# Wording of the history follows expB (same clinical content) without the "Clinical history:"
# prefix, since the template supplies "patient history". "empty" keeps the template's quotes empty.
HISTORIES = {
    "empty": "",
    "knee": "knee pain",
    "pain": "chronic pelvic pain, no prior imaging",
    "suspected": "suspected endometriosis",
    "known_em": "known ovarian endometrioma, follow-up study",
}

# label -> text inserted after "Is there"
LABELS = {
    "em": "an ovarian endometrioma",
    "em_left": "a left ovarian endometrioma",
    "em_right": "a right ovarian endometrioma",
}


def messages_us1(vin: VolumeInput, query: str) -> list:
    """Official MRI-US1 layout: [image, image, ..., query]; no instruction, no slice labels."""
    content = [{"type": "image", "image": im} for im in vin.images()]
    content.append({"type": "text", "text": query})
    return [{"role": "user", "content": content}]


def build(layout: str, vin: VolumeInput, seq: str, hist: str, label: str) -> list:
    """Messages for one read.

    us1      = official layout + MRI-US1 text
    notebook = our notebook layout (instruction + "SLICE n") + MRI-US1 text (isolates the layout)
    ours     = our original prompt: notebook layout, history appended to the instruction as in expB,
               Q_ENDO question; only defined for label "em" (Q_ENDO does not name a side)."""
    us1 = US1.format(history=HISTORIES[hist], label=LABELS[label])
    if layout == "us1":
        return messages_us1(vin, us1)
    if layout == "notebook":
        return build_messages(vin, "\n\n" + us1)[0]
    if layout == "ours":
        h = HISTORIES[hist]
        instr = INSTRUCTION.format(seq_desc=SEQ_DESC[seq]) + (f" Clinical history: {h}." if h else "")
        return build_messages(vin, Q_ENDO, instruction=instr)[0]
    raise ValueError(layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--seq", default="T2FS")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--layouts", default="us1,notebook,ours")
    ap.add_argument("--histories", default=",".join(HISTORIES))
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--ablations", default="real,blank")
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
                "blank": prepare_from_array(blank_like(vol), c.subject, args.seq, tags={"abl": "blank"})}
        for abl in args.ablations.split(","):
            for layout in args.layouts.split(","):
                for label in args.labels.split(","):
                    if layout == "ours" and label != "em":
                        continue
                    for h in args.histories.split(","):
                        msgs = build(layout, vins[abl], args.seq, h, label)
                        meta = {"exp": "expM", "subject": c.subject, "seq": args.seq, "abl": abl,
                                "layout": layout, "label": label, "hist": h}
                        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
                        row = dict(subject=c.subject, gt=c.label, em_side=c.em_side, seq=args.seq, abl=abl,
                                   layout=layout, label=label, hist=h, logit_yes_no=sc[" yes"] - sc[" no"])
                        if args.generate:
                            out = m.generate(msgs, meta=meta)
                            row.update(endo=parse_yesno(out)[0], endo_raw=out, truncated=m.truncated())
                        rows.append(row)
        print(c.subject, c.label, "done", flush=True)
    save(pd.DataFrame(rows), out_dir("expM_mri_us1") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")


if __name__ == "__main__":
    main()
