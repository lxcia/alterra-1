"""Positive control 1: reproduce Google's MedGemma 1.5 CT notebook on Google's own example volume.

Replicates google-health/medgemma notebooks/high_dimensional_ct_hugging_face.ipynb @ ea92d0f
step by step (same series, same 85-slice sampling formula, same HU windows, JPEG data URIs,
same prompt, greedy decoding with max_new_tokens=2000), then compares the generated text with
the output saved in the notebook. It runs two input paths:
  (a) "notebook": processor.apply_chat_template(..., tokenize=True) exactly as in the notebook;
  (b) "wrapper":  this repo's models.VLM.generate (the path every experiment uses).
If (a) == (b) == the published output, our chat template, image-token interleaving and context
handling are validated. Also reports prompt length and the number of image tokens.

Data: TCIA Colorectal-Liver-Metastases series 1.3.6.1.4.1.14519.5.2.1.9203.8273.275179554444442893192427753220
(downloaded from the Imaging Data Commons public bucket into derived/ct_repro/dicom/ with idc-index).

Run: python src/ct_repro.py   -> results/ct_repro/ct_repro.json
"""
import base64
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "experiments"))
import common  # noqa: F401,E402  (sets HF_HOME, sys.path)
from models import VLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derived" / "ct_repro" / "ct_notebook_ea92d0f.ipynb"
OUT = ROOT / "results" / "ct_repro"

# ---- verbatim from the notebook -------------------------------------------------------------
MAX_SLICE = 85
instruction = ("You are an instructor teaching medical students. You are "
               "analyzing a contiguous block of CT slices from the center of "
               "the abdomen. Please review the slices provided below "
               "carefully.")
query_text = ("\n\nBased on the visual evidence in the slices provided above, "
              "is this image a good teaching example of liver pathology? "
              "Comment on hypodense lesions or other hepatic irregularities. "
              "Do not comment on findings outside the liver. Please provide "
              "your reasoning and conclude with a 'Final Answer: yes' or "
              "'Final Answer: no'.")


def norm(ct_vol, min, max):
    ct_vol = np.clip(ct_vol, min, max)
    ct_vol = ct_vol.astype(np.float32)
    ct_vol -= min
    ct_vol /= (max - min)
    ct_vol *= 255.0
    return ct_vol


def window(ct_vol):
    window_clips = [(-1024, 1024), (-135, 215), (0, 80)]
    return np.stack([norm(ct_vol, clip[0], clip[1]) for clip in window_clips], axis=-1)


def _encode(data):
    format = "jpeg"
    with io.BytesIO() as img_bytes:
        with Image.fromarray(data) as img:
            img.save(img_bytes, format=format)
        img_bytes.seek(0)
        encoded_string = base64.b64encode(img_bytes.getbuffer()).decode("utf-8")
    return f"data:image/{format};base64,{encoded_string}"
# ----------------------------------------------------------------------------------------------


def published_output() -> str:
    nb = json.loads(NB.read_text())
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            md = "".join(o.get("data", {}).get("text/markdown", []))
            if "[ MedGemma ]" in md:
                return md.split("**[ MedGemma ]**")[1].strip().strip("-").strip()
    raise RuntimeError("published output not found")


def main():
    files = sorted(glob.glob(str(ROOT / "derived" / "ct_repro" / "dicom" / "**" / "*.dcm"), recursive=True))
    dicom_instances = sorted((pydicom.dcmread(f) for f in files), key=lambda i: int(i.InstanceNumber))
    n_total = len(dicom_instances)
    if len(dicom_instances) > MAX_SLICE:
        dicom_instances = [dicom_instances[int(round(i / (MAX_SLICE) * (len(dicom_instances) - 1)))]
                           for i in range(1, MAX_SLICE + 1)]
    slices = [np.round(window(pydicom.pixels.apply_rescale(d.pixel_array, d)), 0).astype(np.uint8)
              for d in dicom_instances]
    uris = [_encode(s) for s in slices]

    def messages(image_items):
        content = [{"type": "text", "text": instruction}]
        for n, im in enumerate(image_items, 1):
            content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": f"SLICE {n}"})
        content.append({"type": "text", "text": query_text})
        return [{"role": "user", "content": content}]

    m = VLM("medgemma", load=True)
    proc, model = m.processor, m.model

    # (a) notebook path: data-URI images, tokenize=True inside apply_chat_template
    inputs = proc.apply_chat_template(messages(uris), add_generation_prompt=True, continue_final_message=False,
                                      return_tensors="pt", tokenize=True, return_dict=True)
    n_prompt = int(inputs["input_ids"].shape[-1])
    image_token_id = model.config.image_token_index if hasattr(model.config, "image_token_index") else None
    n_image_tokens = int((inputs["input_ids"] == image_token_id).sum()) if image_token_id is not None else -1
    with torch.inference_mode():
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=2000)
    raw_a = proc.decode(gen[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    out_a = raw_a.strip()

    # (b) this repo's wrapper path: decoded PIL images through models.VLM.generate
    pil = [Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB") for u in uris]
    raw_b = m.generate(messages(pil), max_new_tokens=2000, meta={"exp": "ct_repro"})
    out_b = raw_b.strip()

    pub = published_output()
    res = dict(series_instances=n_total, slices_used=len(slices), prompt_tokens=n_prompt,
               image_tokens=n_image_tokens, max_position_embeddings=getattr(model.config.text_config, "max_position_embeddings", None),
               wrapper_format=m.fmt, wrapper_prompt_tokens=m.last.get("n_prompt_tokens"),
               prompt_tokens_equal=m.last.get("n_prompt_tokens") == n_prompt,
               published=pub, notebook_path=out_a, wrapper_path=out_b,
               notebook_path_raw=raw_a, wrapper_path_raw=raw_b,  # unstripped (P20)
               notebook_equals_published=out_a == pub, wrapper_equals_notebook=out_b == out_a,
               notebook_equals_published_unstripped=raw_a == pub, wrapper_equals_notebook_unstripped=raw_b == raw_a,
               final_answer_published=pub.strip().split()[-1], final_answer_notebook=out_a.split()[-1],
               final_answer_wrapper=out_b.split()[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ct_repro.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
