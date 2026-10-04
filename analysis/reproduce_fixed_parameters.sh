#!/usr/bin/env bash
# Run using the original experiment's Python environment.
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: bash reproduce_fixed_parameters.sh NEW_REPO_DIR {plan|oof|final|tables|all} [GPU_IDS...|cpu]" >&2
  exit 1
fi
REPRO_ROOT=$1
REPRO_STAGE=$2
shift 2
REPRO_GPUS=("$@")
if (( ${#REPRO_GPUS[@]} == 0 )); then REPRO_GPUS=(0 1); fi
REPRO_MODE=gpu
if [[ "${REPRO_GPUS[*]}" == cpu ]]; then
  REPRO_MODE=cpu
  export REPRO_ALLOW_CPU=1
  export CUDA_VISIBLE_DEVICES=""
else
  unset REPRO_ALLOW_CPU
fi
REPRO_PYTHON=${REPRO_PYTHON:-python3}
REPRO_SOURCE=/home/gvadzabd/cnn
REPRO_TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

case "$REPRO_STAGE" in plan|oof|final|tables|all) ;; *) echo "Unknown stage: $REPRO_STAGE" >&2; exit 1 ;; esac
if (( ${#REPRO_GPUS[@]} > 4 )); then
  echo "Use one to four GPU IDs." >&2
  exit 1
fi
if [[ "$REPRO_MODE" == gpu ]]; then
  for REPRO_GPU in "${REPRO_GPUS[@]}"; do
    if [[ "$REPRO_GPU" == cpu || -z "$REPRO_GPU" || "$REPRO_GPU" == *,* ]]; then
      echo "Pass separate GPU IDs, or use cpu alone." >&2
      exit 1
    fi
  done
fi

# A separate source/data copy gives the hard-coded OOF results paths isolation.
"$REPRO_PYTHON" - "$REPRO_SOURCE" "$REPRO_ROOT" "$REPRO_TOOLS" "$REPRO_MODE" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
source, target, tools = [Path(p).resolve() for p in sys.argv[1:4]]
mode = sys.argv[4]
if target == source or target.is_relative_to(source) or source.is_relative_to(target):
    raise SystemExit('Choose a separate sibling directory outside the original repository.')
marker=target/'fixed_reproduction_origin.json'
if target.exists():
    if not marker.is_file() or json.loads(marker.read_text())['source_repository'] != str(source):
        raise SystemExit('Existing target is not a workspace created by this launcher.')
    metadata = json.loads(marker.read_text())
    if metadata.get('runtime_device', 'gpu') != mode:
        raise SystemExit('Choose a separate directory when switching between CPU and GPU modes.')
else:
    target.mkdir(parents=True)
    shutil.copytree(source/'src',target/'src',ignore=shutil.ignore_patterns('__pycache__'))
    (target/'data').mkdir()
    for name in ['KDDTrain+.txt','KDDTest+.txt']:
        shutil.copy2(source/'data'/name,target/'data'/name)
    (target/'results').mkdir()
    (target/'analysis').mkdir()
    shutil.copy2(source/'results/variant_specific_scaling_selection_ea63ca7e718d.json',target/'fixed_selection.json')
    metadata = {'source_repository':str(source),'parameters_fixed':True}
metadata['runtime_device'] = mode
marker.write_text(json.dumps(metadata,indent=2)+'\n')
for name in ['export_fixed_parameter_tables.py','reverify_paper_results.py','adapt_reproduction_gpu_workers.py']:
    shutil.copy2(tools/name,target/'analysis'/name)
PY

if [[ "$REPRO_MODE" == cpu ]]; then
  "$REPRO_PYTHON" "$REPRO_ROOT/analysis/adapt_reproduction_gpu_workers.py" --repo "$REPRO_ROOT" --allow-cpu
else
  "$REPRO_PYTHON" "$REPRO_ROOT/analysis/adapt_reproduction_gpu_workers.py" --repo "$REPRO_ROOT"
fi

cd -- "$REPRO_ROOT"
repro_run() {
  if [[ "$REPRO_STAGE" == plan ]]; then "$@" --dry-run; else "$@"; fi
}

if [[ "$REPRO_STAGE" == oof || "$REPRO_STAGE" == all || "$REPRO_STAGE" == plan ]]; then
  for REPRO_ARCH in conv2d conv1d transformer mlp; do
    case "$REPRO_ARCH" in conv2d) REPRO_GAMMA=0.50 ;; transformer) REPRO_GAMMA=0.75 ;; *) REPRO_GAMMA=0.25 ;; esac
    repro_run "$REPRO_PYTHON" "src/tune_${REPRO_ARCH}_focal_cv_4gpu.py" \
      --betas 0.99 --focal-gammas "$REPRO_GAMMA" \
      --seeds 0 1 2 --fold-seed 0 --epochs 25 --batch-size 256 \
      --gpus "${REPRO_GPUS[@]}"
    repro_run "$REPRO_PYTHON" "src/run_${REPRO_ARCH}_baseline_cv_4gpu.py" \
      --seeds 0 1 2 --fold-seed 0 --epochs 25 --batch-size 256 \
      --coefficient-values 1.0 --gpus "${REPRO_GPUS[@]}"
    repro_run "$REPRO_PYTHON" "src/run_${REPRO_ARCH}_batch_baseline_cv_4gpu.py" \
      --seeds 0 1 2 --fold-seed 0 --epochs 25 --batch-size 256 \
      --minority-per-batch 1 --coefficient-values 1.0 \
      --gpus "${REPRO_GPUS[@]}"
    repro_run "$REPRO_PYTHON" "src/run_${REPRO_ARCH}_focal_batch_cv_4gpu.py" \
      --cb-beta 0.99 --focal-gamma "$REPRO_GAMMA" \
      --seeds 0 1 2 --fold-seed 0 --epochs 25 --batch-size 256 \
      --minority-per-batch 1 --coefficient-values 1.0 \
      --gpus "${REPRO_GPUS[@]}"
  done
fi

if [[ "$REPRO_STAGE" == final || "$REPRO_STAGE" == all || "$REPRO_STAGE" == plan ]]; then
  repro_final() {
    if [[ "$REPRO_MODE" == cpu ]]; then
      repro_run "$@" --allow-cpu --gpus ""
    else
      repro_run "$@" --gpus "${REPRO_GPUS[@]}"
    fi
  }
  repro_final "$REPRO_PYTHON" src/run_final_baseline_vs_full_kddtest_4gpu.py \
    --architectures conv2d conv1d transformer mlp \
    --variants baseline focal_only batch_only full \
    --seeds 0 1 2 --epochs 25 --batch-size 256 --minority-per-batch 1 \
    --results-dir "$REPRO_ROOT/results"
fi

if [[ "$REPRO_STAGE" == tables || "$REPRO_STAGE" == all ]]; then
  "$REPRO_PYTHON" analysis/export_fixed_parameter_tables.py \
    --repo "$REPRO_ROOT" --selection "$REPRO_ROOT/fixed_selection.json" \
    --output-dir "$REPRO_ROOT/paper_tables"
fi
