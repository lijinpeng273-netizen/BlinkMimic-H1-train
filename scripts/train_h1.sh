#!/usr/bin/env bash
# Usage: bash scripts/train_h1.sh logs/h1_run [--profile debug|local|4090|fast] [train flags]
set -euo pipefail
cd "$(dirname "$0")/.."
log_dir=${1:?usage: train_h1.sh LOG_DIR [--profile debug|local|4090|fast] [train flags]}
shift
profile=local
if [[ ${1:-} == --profile ]]; then
  profile=${2:?missing profile}
  shift 2
fi
common=(--robot h1 --log_dir "$log_dir" --buffer_steps 256 --wm_hidden 1024,1024,1024
        --wm_window 8 --horizon 32 --hist 10 --goal_frames 10 --lr 2e-4 --pi_lr_mult 1
        --pi_ref_skip --wm_structured --pi_ema 0.995 --bin_fix --ckpt_minutes 0)
small=(--num_envs 32 --wm_batch 64 --pi_batch 32)
case "$profile" in
  debug)
    recipe=("${small[@]}" --no_compile --iterations 300 --wm_only_iters 100 --ckpt_every 100)
    ;;
  local)
    recipe=("${small[@]}" --iterations 5000 --wm_only_iters 500 --ckpt_every 500
            --wm_mirror 0.5 --contact_reward 0.25 --ref_contact_fix --fail_reset 0.3)
    ;;
  4090)
    recipe=(--num_envs 4096 --wm_batch 2048 --pi_batch 2048 --iterations 10000 --wm_only_iters 500
            --ckpt_every 500 --wm_mirror 0.5 --contact_reward 0.25 --ref_contact_fix --fail_reset 0.3
            --pi_opt muon --wm_opt muon --no_sync --fused_adam --fused_rot --compile_more)
    ;;
  fast)
    source scripts/presets.sh
    read -r -a recipe <<< "${PRESET[main]}"
    recipe+=(--buffer_steps 256 --lr 2e-4 --pi_lr_mult 1 --wm_only_iters 200 --iterations 10000
             --eprobe_async 0)
    ;;
  *) printf 'Unknown H1 profile: %s\n' "$profile" >&2; exit 2 ;;
esac
if [[ ! -f data/h1_manifest.json ]]; then
  printf 'Prepare H1 assets and motions with tools/prepare_h1.py first.\n' >&2
  exit 2
fi
mkdir -p "$log_dir"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-2}
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
# PYTHON may name an alternate tested interpreter; default is this repository's locked environment.
PYTHONPATH=src ${PYTHON:-.venv/bin/python} -u -m blinkmimic.train "${common[@]}" "${recipe[@]}" "$@" 2>&1 | tee "$log_dir/stdout.log"
