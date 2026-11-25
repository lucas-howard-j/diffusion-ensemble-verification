#!/usr/bin/env bash
# SSH DDPM ablation for the supplement training figure (panel b).
# Each variant changes one setting from the final configuration in src/config.py;
# every variant is trained with 3 seeds on the same data build and splits.
#
# Run from the repo root:  bash scripts/run_ablation.sh [N_PARALLEL]
# Logs:        logs/ablation/{variant}_s{seed}.log
# Checkpoints: checkpoints/ablation/{variant}_s{seed}.pt   (gitignored)
set -euo pipefail
N_PARALLEL=${1:-8}
PY=${PY:-python}
# DEVICE=mps|cuda|cpu to force a device; default lets train.py pick (cuda > mps > cpu)
DEV_FLAG=${DEVICE:+--device $DEVICE}
SEEDS="0 1 2"

# name | extra train.py flags
VARIANTS=(
  "final|"
  "dropout05|--dropout 0.05"
  "dropout15|--dropout 0.15"
  "plain_silu|--no-beta-silu"
  "nao_t_only|--nao-lags 0"
  "ssh_t_only|--ssh-lags 0"
  "larger_net|--hidden 1024 1024 512 --lr 1e-4"
  "cosine|--beta-schedule cosine"
)

mkdir -p logs/ablation
jobs=()
for v in "${VARIANTS[@]}"; do
  name=${v%%|*}; flags=${v#*|}
  for s in $SEEDS; do
    jobs+=("PYTHONPATH=. $PY -m src.model.train $DEV_FLAG --seed $s --out ablation/${name}_s${s}.pt --log-dir logs/ablation $flags > logs/ablation/${name}_s${s}.log 2>&1")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$N_PARALLEL" -I{} bash -c '{}'
grep -H "Best val loss" logs/ablation/*_s*.log
