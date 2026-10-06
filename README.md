# Rare-class intrusion detection on NSL-KDD

Experiments with focal loss, minority-guaranteed batching, and score scaling across Conv2D, Conv1D, Transformer, and MLP. Rare Macro-F1 is the average of R2L and U2R F1.

## Setup

Use Linux, Python 3.12, and compatible NVIDIA GPUs. On a cluster, allocate GPUs through the scheduler first. Run these commands from the repository root in the same Bash session.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

set -euo pipefail
experiment_results="results/factorial"
device_args=(--gpus 0 1)
```

This uses two GPUs. For one, set `device_args=(--gpus 0)`. IDs refer to the GPUs visible in your allocation. The `_4gpu.py` filenames do not require four GPUs. Training stops if no GPU is available.

Both datasets are included in `data/`. Defaults are three seeds (0, 1, 2), four training folds, 25 epochs, and batch size 256. Run the steps below in order. The full workflow trains 1,056 models.

## 1. Search focal-loss parameters

Each backbone tests 18 pairs: β ∈ {0.99, 0.999, 0.9999} and γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}. Settings are selected using held-out predictions from KDDTrain+ only, called out-of-fold (OOF) predictions.

```bash
for backbone in conv2d conv1d transformer mlp; do
    python "src/tune_${backbone}_focal_cv_4gpu.py" \
        --results-dir "$experiment_results" "${device_args[@]}"
done
```

## 2. Get predictions for the other training settings

Load each backbone's chosen focal parameters, then train baseline, batching only, and focal + batching (focal only is already trained from part 1). The shared runner supports all four backbones.

```bash
for backbone in conv2d conv1d transformer mlp; do
    read -r beta gamma < <(
        python - "$experiment_results/${backbone}_focal_stage1_latest.json" <<'PY'
import json
from pathlib import Path
import sys

pointer = Path(sys.argv[1])
record = json.loads(pointer.read_text())
best = json.loads((pointer.parent / Path(record["best_config"]).name).read_text())
print(best["beta"], best["focal_gamma"])
PY
    )
    for mode in baseline_ce baseline_batch focal_balanced; do
        python src/tune_conv2d_score_scaling_cv_4gpu.py \
            --architecture "$backbone" --training-mode "$mode" \
            --cb-beta "$beta" --focal-gamma "$gamma" \
            --coefficient-values 1.0 \
            --results-dir "$experiment_results" "${device_args[@]}"
    done
done
```

## 3. Search score-scaling parameters

Search 576 R2L/U2R coefficient pairs for each backbone and training setting using the saved OOF predictions. This saves the selected parameters before KDDTest+ evaluation.

```bash
python src/tune_variant_specific_score_scaling.py \
    --results-dir "$experiment_results" select
```

## 4. Train final models and evaluate KDDTest+

Train on all KDDTrain+ using the saved parameters, then evaluate raw and scaled predictions. This produces all eight combinations of the three controls.

```bash
python src/run_final_baseline_vs_full_kddtest_4gpu.py \
    --results-dir "$experiment_results" "${device_args[@]}"

python src/tune_variant_specific_score_scaling.py \
    --results-dir "$experiment_results" evaluate
```

## 5. Generate tables and statistics

Create all nine paper tables from the experiment outputs:

```bash
python src/export_paper_tables.py \
    --results-dir "$experiment_results" --output-dir tables/factorial
```

Outputs:

- `results/factorial/`: predictions, training logs, parameter rankings, selected parameters, and per-seed metrics.
- `tables/factorial/`: nine tables in CSV and LaTeX, unrounded metric summaries, and marginal contrasts and interactions.

Rerun the same command to resume completed fits. Use a new output directory when changing code or settings. Run tests with `python -m unittest discover -s tests`.


## Reproducibility note

Training randomness and differences in hardware or software can affect scores and parameter rankings. Some candidates score closely: the top two settings in the fresh Conv2D focal search differed by 0.10 percentage points in OOF Rare Macro-F1. Differences between candidates are not always this small.

Comparing the original experiment with one reproduced completed fresh run, **3 of 4 focal (β, γ) pairs** and **7 of 16 score-scaling (R2L, U2R) coefficient pairs** matched exactly. The other differences were also razor close. 

However, regardelss of small hyper-param search differences, both runs supported the same broad finding: batching had a positive KDDTest+ marginal contrast in every backbone, and every backbone's test winner included it. The best addition to batching varied by backbone and evaluation setting. The highest test architecture mean came from focal + batching originally and batching + scaling in the fresh run. In the end, both winners include batching, consistent with the paper's practical conclusion: establish reliable rare-class exposure first, then assess additional controls for the particular pipeline.
