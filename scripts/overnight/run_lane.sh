#!/bin/bash
# Overnight queue runner (2026-10-26). Usage: run_lane.sh <lane-name> <gpu> <queue-file> [extras-file]
# Runs every line of <queue-file> in order (each line = experiment script + arguments, relative to
# src/), then, if given, claims lines from the shared <extras-file> until none are left. A claim is an
# atomic mkdir in results/logs/claims/, so two workers never run the same extras line. A failed job is
# retried once after 2 minutes (model calls are cached, so a retry resumes where it stopped).
set -u
LANE=$1; GPU=$2; Q=$3; X=${4:-}
R=/lotterlab/users/lucia/endo-release
cd $R
export CUDA_VISIBLE_DEVICES=$GPU VLM_PROMPT_FORMAT=v2 PYTHONUNBUFFERED=1 HF_HOME=$HOME/.cache/huggingface
L=$R/results/logs; mkdir -p $L/claims
log() { echo "$(date '+%F %T') [$LANE gpu$GPU] $*" | tee -a $L/lanes.log; }
run() {  # $1 = job id, rest = command line
  local id=$1; shift
  for attempt in 1 2; do
    log "START $id (attempt $attempt): $*"
    .venv/bin/python src/$* > $L/${LANE}_${id}_a${attempt}.log 2>&1
    local rc=$?
    log "END $id rc=$rc"
    [ $rc -eq 0 ] && return 0
    sleep 120
  done
  log "FAILED $id (two attempts)"
}
n=0
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  n=$((n+1)); run "q$(printf %02d $n)" $line
done < $Q
if [ -n "$X" ]; then
  i=0
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    i=$((i+1)); id="x$(printf %02d $i)"
    mkdir $L/claims/$id 2>/dev/null || continue
    echo "$LANE $line" > $L/claims/$id/by
    run $id $line
  done < $X
fi
log "LANE DONE"
