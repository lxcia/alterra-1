#!/bin/bash
# Reproduce every result in results/ (the 2026-10-26/27 run: D2 T2FS, prompt format v2,
# min-max preprocessing). The same commands, split across GPUs, are in scripts/overnight/.
#
# Model calls are cached in results/cache/*.jsonl (keyed by model revision, prompt format, decoding
# settings, prompt text and image hashes). A command whose calls are all cached loads no model, so
#     CACHE_ONLY=1 bash run_all.sh     # re-derive every table from the cache on CPU; fails on any cache miss
# From an empty cache this is roughly 2-3 GPU-days on one H100 (prompts are 10k-22k tokens);
# split it across GPUs with --shard i --nshards n, or with scripts/overnight/run_lane.sh.
#
# Usage: bash run_all.sh            (uses GPU $CUDA_VISIBLE_DEVICES, default 0)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export HF_HOME=${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface} PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLM_PROMPT_FORMAT=v2
[ "${CACHE_ONLY:-0}" = "1" ] && export VLM_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES=""

# 0. data inventory, QC images, cohort overview (CPU)
# Data (see README "Data"): UT-EndoMRI under $UTENDO_DATA_ROOT; positive-control data under derived/.
$PY src/inventory.py
$PY src/brain_inventory.py                # BraTS subset for expP P2
$PY src/qc_overlay.py
$PY src/qc_synthetic.py D2-077 D2-002 D2-054
$PY src/qc_synthetic_cyst.py
$PY src/lesion_regime.py                  # real-lesion brightness/size on the expC4 axes
$PY dataset_overview.py

# 1. experiments (each writes results/<exp>/<model>_*.csv + .provenance.json)
E=src/experiments
# pipeline check and positive controls
[ -n "${VLM_CACHE_ONLY:-}" ] || $PY src/ct_repro.py   # also runs the notebook path directly: needs a GPU
$PY $E/expP_positive_controls.py --parts P1,P3,P4
$PY $E/expP_positive_controls.py --parts P2
$PY $E/expX_cxr_laterality.py
$PY $E/exp00_baseline.py                  # context: baseline presence / side answers

# Failure 1: default side
$PY $E/expC3_side_prompt.py
$PY $E/expC_laterality.py --geoms orig,lr --conventions none --generate
$PY $E/expC2_synthetic_side.py --generate
for i in 0 1; do $PY $E/expC4_psychometric.py --shard $i --nshards 2; done   # file names match results/
$PY $E/expA3_type_order.py

# Failure 2: clinical history
$PY $E/expB4_history_controls.py --parts irrelevant,known
$PY $E/expB_history.py
$PY $E/expB2_history_dose.py
$PY $E/expB3_history_x_paraphrase.py
for i in 0 1; do $PY $E/expR_freetext_report.py --shard $i --nshards 2; done  # 2000 new tokens; truncation flagged
$PY $E/expM_mri_us1.py                    # Google's MRI-US1 prompt, scores only
for i in 0 1; do  # generated answers for the official layout, in results/expM_generate/
  RESULTS_SUBDIR=expM_generate $PY $E/expM_mri_us1.py --layouts us1 --labels em \
      --histories empty,known_em --ablations real,blank --generate --shard $i --nshards 2
done
$PY $E/expF_pushback.py

# appendix experiments and noise floors
$PY $E/expA_sequence.py
$PY $E/expA2_contrast.py --generate
$PY $E/expD_slices.py
$PY $E/expD2_shuffle_confirm.py
$PY $E/expE_invariance.py
$PY $E/expN_noise.py

# other models (core experiments only)
for M in gemma3 medgemma27 qwen; do
  $PY $E/expC3_side_prompt.py --model $M --variants ref,order_rl,open
  $PY $E/expB4_history_controls.py --model $M --parts irrelevant,known
done
$PY $E/expR_freetext_report.py --model medgemma27 --geoms orig

# 2. tables
$PY src/analyze.py exp00 expP expX expC expC2 expC3 expC4 expA3 expB4 expB expB2 expB3 expR expF \
    expA expA2 expD expD2 expE expN | tee results/summary_medgemma.txt
for M in gemma3 medgemma27 qwen; do
  $PY src/analyze.py --model $M expC3 expB4 | tee results/summary_$M.txt
done
$PY src/analyze_expM.py | tee results/summary_expM.txt
