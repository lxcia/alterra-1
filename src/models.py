"""MedGemma 1.5 / Gemma 3 / Qwen wrappers with an append-only JSONL cache.

Raw outputs are stored in results/cache/<model>__<host>_<pid>.jsonl so reruns are free and every
number traces back to raw model output.

Prompt formats (PATCH.md, P05). Select with VLM(fmt=...) or the VLM_PROMPT_FORMAT env variable
(default DEFAULT_FORMAT):
  * "v2": the prompt is tokenised with ONE <bos>, as in Google's notebook. The cache key
    holds (format, model id, pinned revision, decoding config incl. repetition_penalty, prompt
    text, image hashes incl. size/mode).
  * "v1": the pre-patch behaviour, kept so that results produced before the patch reproduce from
    the cache: the chat template is rendered to text (which already starts with <bos>) and the
    processor adds a second <bos>; the key holds (model id, decoding config, prompt text, image
    hashes) only; the model's generation_config (e.g. Qwen's repetition_penalty=1.05) applies.
VLM_CACHE_ONLY=1 raises on a cache miss instead of loading the model (CPU re-derivation).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results" / "cache"

MODELS = {
    "medgemma": "google/medgemma-1.5-4b-it",
    "gemma3": "google/gemma-3-4b-it",          # same-architecture general baseline (no medical tuning)
    "medgemma27": "google/medgemma-27b-it",    # MedGemma 27B multimodal (scale comparison)
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}
# Pinned snapshots (the only revisions present in results/cache/). Loading another revision fails.
REVISIONS = {
    "medgemma": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "gemma3": "093f9f388b31de276ce2de164bdc2081324b9767",
    "medgemma27": "2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f",
    "qwen": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}
# Non-default HF cache locations for models that were downloaded elsewhere on this machine
# (VLM_HF_CACHE_<NAME> overrides, e.g. VLM_HF_CACHE_QWEN=/path/to/hub).
CACHE_DIRS = {"qwen": os.environ.get("VLM_HF_CACHE_QWEN", "/lotterlab/users/lucia/EndoSeg/.cache/huggingface/hub")}
FORMATS = ("v1", "v2")
DEFAULT_FORMAT = "v2"


def _img_hash(im: Image.Image, fmt: str = "v1") -> str:
    a = np.asarray(im)
    h = hashlib.sha1(a.tobytes())
    if fmt != "v1":  # include mode and size so two images with the same bytes cannot collide
        h.update(f"|{im.mode}|{im.size}".encode())
    return h.hexdigest()[:16]


def _msg_key(messages: list, fmt: str = "v1") -> list:
    """Messages with PIL images replaced by content hashes (for cache keys and logs)."""
    out = []
    for m in messages:
        content = []
        for c in m["content"]:
            if c["type"] == "image":
                content.append({"type": "image", "sha": _img_hash(c["image"], fmt)})
            else:
                content.append(c)
        out.append({"role": m["role"], "content": content})
    return out


class VLM:
    def __init__(self, name: str = "medgemma", device: str = "cuda", load: bool = False,
                 fmt: str | None = None):
        self.name = name
        self.model_id = MODELS[name]
        self.device = device
        # default "v2" (one <bos>, as in Google's notebook): every result in results/ uses it
        self.fmt = fmt or os.environ.get("VLM_PROMPT_FORMAT", DEFAULT_FORMAT)
        assert self.fmt in FORMATS, self.fmt
        self.cache_only = os.environ.get("VLM_CACHE_ONLY", "") == "1"
        self.last: dict = {}  # cache record of the most recent call (n_new_tokens, truncated, ...)
        self.cache: dict[str, dict] = {}
        # VLM_RECHECK=<tag>: hardware-reproducibility rerun. Ignore the cache (recompute every call)
        # and write to separately tagged cache files so the original raw outputs stay untouched.
        self.recheck = os.environ.get("VLM_RECHECK", "")
        for f in ([] if self.recheck else glob.glob(str(CACHE_DIR / f"{name}__*.jsonl"))):
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        self.cache[r["key"]] = r
                    except json.JSONDecodeError:
                        pass  # tolerate a truncated last line from a killed run
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prefix = f"{name}-recheck-{self.recheck}" if self.recheck else name
        self._cache_file = CACHE_DIR / f"{prefix}__{socket.gethostname()}_{os.getpid()}.jsonl"
        self.model = self.processor = None
        self.revision = REVISIONS[name]
        if load:
            self._load()

    def refresh_cache(self):
        """Pick up calls cached by sibling processes since start-up (so parallel workers that
        traverse cases in different orders skip each other's finished work)."""
        if self.recheck:
            return
        for f in glob.glob(str(CACHE_DIR / f"{self.name}__*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.cache.setdefault(r["key"], r)

    def _load(self):
        if self.model is not None:
            return
        if self.cache_only:
            raise RuntimeError(f"VLM_CACHE_ONLY=1 but a {self.name} call is not in the cache")
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor
        kw = dict(cache_dir=CACHE_DIRS.get(self.name), revision=REVISIONS[self.name])
        try:  # prefer the already-downloaded snapshot (no network / token needed)
            path = snapshot_download(self.model_id, local_files_only=True, **kw)
        except Exception:
            path = snapshot_download(self.model_id, **kw)
        assert Path(path).name == REVISIONS[self.name], (path, REVISIONS[self.name])
        self.processor = AutoProcessor.from_pretrained(path, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map={"": self.device}).eval()

    # ------------------------------------------------------------------ helpers
    def _key(self, kind: str, messages: list, cfg: dict) -> tuple[str, str]:
        k = {"model": self.model_id, "kind": kind, "cfg": cfg, "msgs": _msg_key(messages, self.fmt)}
        if self.fmt != "v1":
            k.update(fmt=self.fmt, revision=self.revision)
        blob = json.dumps(k, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest(), blob

    def _store(self, rec: dict):
        self.cache[rec["key"]] = rec
        self.last = rec
        with open(self._cache_file, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _inputs(self, messages: list, assistant_prefix: str | None = None):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if assistant_prefix:
            text = text + assistant_prefix
        images = [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]
        kw = {"images_kwargs": {"device": self.model.device}} if images else {}
        if self.fmt != "v1":
            # the rendered template already contains <bos>; don't let the tokenizer add another
            kw["add_special_tokens"] = False
        inputs = self.processor(text=text, images=images or None, return_tensors="pt", **kw)
        if self.fmt != "v1":
            ids = inputs["input_ids"][0]
            bos = self.processor.tokenizer.bos_token_id
            assert bos is None or not (len(ids) > 1 and ids[0] == bos and ids[1] == bos), "double <bos>"
        return inputs.to(self.model.device, dtype=torch.bfloat16)

    def truncated(self) -> bool | None:
        """Did the most recent generate() run out of max_new_tokens before end-of-turn?
        Records written before the patch have no token count; for those the output is re-tokenised
        (CPU, tokenizer only) and counted as truncated if it is within 8 tokens of the budget."""
        rec = self.last
        if rec.get("kind") != "generate":
            return None
        if "truncated" in rec:
            return bool(rec["truncated"])
        if not hasattr(self, "_tok"):
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
            path = snapshot_download(self.model_id, cache_dir=CACHE_DIRS.get(self.name),
                                     revision=REVISIONS[self.name], local_files_only=True)
            self._tok = AutoTokenizer.from_pretrained(path)
        n = len(self._tok.encode(rec["output"], add_special_tokens=False))
        return n >= rec["cfg"]["max_new_tokens"] - 8

    # ------------------------------------------------------------------ API
    def generate(self, messages: list, max_new_tokens: int | None = None, do_sample: bool = False,
                 temperature: float = 1.0, seed: int = 0, meta: dict | None = None) -> str:
        if max_new_tokens is None:  # v2 uses the notebook's 2000; v1 kept the pre-patch 1024
            max_new_tokens = 1024 if self.fmt == "v1" else 2000
        cfg = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            cfg.update(temperature=temperature, seed=seed)
        gen_kw = {}
        if self.fmt != "v1":  # pure greedy: override any generation_config extras (Qwen: 1.05)
            gen_kw = {"repetition_penalty": 1.0}
            cfg.update(gen_kw)
        key, _ = self._key("generate", messages, cfg)
        if key in self.cache:
            self.last = self.cache[key]
            return self.last["output"]
        self._load()
        inputs = self._inputs(messages)
        if do_sample:
            torch.manual_seed(seed)
        t0 = time.time()
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                      temperature=temperature if do_sample else None,
                                      top_p=None, top_k=None, **gen_kw)
        new = gen[0, inputs["input_ids"].shape[-1]:]
        out = self.processor.decode(new, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        n_new = int(new.shape[-1])
        self._store({"key": key, "kind": "generate", "fmt": self.fmt, "model": self.model_id,
                     "revision": self.revision, "cfg": cfg, "msgs": _msg_key(messages, self.fmt), "output": out,
                     "n_prompt_tokens": int(inputs["input_ids"].shape[-1]), "n_new_tokens": n_new,
                     "truncated": bool(n_new >= max_new_tokens and int(new[-1]) not in eos),
                     "secs": round(time.time() - t0, 2), "meta": meta or {}})
        return out

    def score(self, messages: list, candidates: list[str], assistant_prefix: str = "",
              meta: dict | None = None) -> dict[str, float]:
        """Log-probability of each candidate's FIRST token at the position after the prefix.

        Returns {candidate: logprob}. Used as a continuous outcome (e.g. P(yes) vs P(no)).
        Candidates are case-sensitive tokens: pass every surface form the model uses (e.g. both
        " left" and " Left") and combine them with `logsumexp_forms`."""
        cfg = {"candidates": candidates, "prefix": assistant_prefix}
        key, _ = self._key("score", messages, cfg)
        if key in self.cache:
            self.last = self.cache[key]
            return self.last["output"]
        self._load()
        inputs = self._inputs(messages, assistant_prefix)
        with torch.inference_mode():
            # logits_to_keep=1: only the last position's logits (full-vocab logits at every
            # position of a 20k-token prompt would cost ~12 GB and most of the runtime)
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        out = {}
        for c in candidates:
            ids = self.processor.tokenizer.encode(c, add_special_tokens=False)
            if self.fmt != "v1":
                assert len(ids) == 1, f"score candidate {c!r} is {len(ids)} tokens"
            out[c] = float(lp[ids[0]])
        self._store({"key": key, "kind": "score", "fmt": self.fmt, "model": self.model_id,
                     "revision": self.revision, "cfg": cfg, "msgs": _msg_key(messages, self.fmt), "output": out,
                     "n_prompt_tokens": int(inputs["input_ids"].shape[-1]), "meta": meta or {}})
        return out


def logsumexp_forms(scores: dict[str, float], forms: list[str]) -> float:
    """log(sum of probabilities) over several surface forms of one answer (e.g. " left", " Left")."""
    v = np.array([scores[f] for f in forms])
    return float(v.max() + np.log(np.exp(v - v.max()).sum()))
