# Stage 4: Sinkhorn Ablation Tests
set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PLANT_NAME="Plant_023-1"
DATASET_SPECS=(
    "50000 10000 25000"
)

# If set to 1, Stage 4 always starts from scratch per dataset.
STAGE04_FORCE_RERUN=0

ABLATION_REPLICATES=2
ABLATION_EPOCHS=20
ABLATION_PATIENCE=20
ABLATION_FRACTION=1.0

SINKHORN_LR="0.0005"
SINKHORN_BS="8"

OPT_RESTARTS=10
OPT_STEPS=1000

ABLATION_ROOT="Ablation Tests/${PLANT_NAME}"
LOG_FILE="Research Pipeline/04_ablation.log"
mkdir -m 777 -p "${ABLATION_ROOT}" "$(dirname "$LOG_FILE")"

log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"; }

optimizer_result_exists() {
    local optimizer_root="$1"
    local model_name="$2"
    local replicate_id="$3"

    find "$optimizer_root" -type f -path "*/Run_*/${model_name}/*/Rep_${replicate_id}/opt_result.csv" -print -quit 2>/dev/null | grep -q .
}

stage4_complete() {
    local optimizer_root="$1"
    local expected=$(( ${#ABLATION_MODELS[@]} * ABLATION_REPLICATES ))
    local found=0

    for model_name in "${ABLATION_MODELS[@]}"; do
        for rep in $(seq 1 "$ABLATION_REPLICATES"); do
            if optimizer_result_exists "$optimizer_root" "$model_name" "$rep"; then
                found=$((found + 1))
            fi
        done
    done

    [[ "$found" -ge "$expected" ]]
}

log_tuning_metrics() {
    local tuning_summary="$1"
    local model_name="$2"

    if [[ ! -f "$tuning_summary" ]]; then
        log "Metric summary not found for $model_name at $tuning_summary"
        return 0
    fi

    python3 - "$tuning_summary" "$model_name" <<'PY' | while IFS= read -r line; do log "$line"; done
import csv
import math
import sys

summary_path = sys.argv[1]
model_name = sys.argv[2]

with open(summary_path, "r", newline="") as f:
    rows = [row for row in csv.DictReader(f) if row.get("model", "").startswith(model_name)]

if not rows:
    print(f"Metric summary not found for {model_name} in {summary_path}")
    raise SystemExit(0)

r2_values = []
lpfg_values = []
surrogate_values = []
for row in rows:
    try:
        value = float(row.get("best_val_r2", "nan"))
    except Exception:
        value = float("nan")
    if math.isfinite(value):
        r2_values.append(value)

    try:
        cost = float(row.get("best_vlab_cost", row.get("best_lpfg_cost", "nan")))
    except Exception:
        cost = float("nan")
    if math.isfinite(cost):
        lpfg_values.append(cost)

    try:
        surrogate = float(row.get("best_lpfg_surrogate_cost", "nan"))
    except Exception:
        surrogate = float("nan")
    if math.isfinite(surrogate):
        surrogate_values.append(surrogate)

if not lpfg_values and not r2_values:
    print(f"Metric summary for {model_name}: no finite LPFG/R2 values in {summary_path}")
    raise SystemExit(0)

parts = [f"Metric summary for {model_name}:"]
if lpfg_values:
    parts.append(f"best_vlab_cost_min={min(lpfg_values):.3f}")
    parts.append(f"best_vlab_cost_mean={sum(lpfg_values)/len(lpfg_values):.3f}")
if surrogate_values:
    parts.append(f"best_lpfg_surrogate_cost_min={min(surrogate_values):.3f}")
if r2_values:
    parts.append(f"best_val_r2_max={max(r2_values):.6f}")
    parts.append(f"best_val_r2_mean={sum(r2_values) / len(r2_values):.6f}")
parts.append(f"n={len(rows)}")
print(" | ".join(parts))
PY
}

ABLATION_MODELS=("sinkhorn" "sinkhorn_no_encoder" "sinkhorn_no_scaler" "sinkhorn_no_aggregator" "sinkhorn_hollow")

log "=== STAGE 4: Sinkhorn Ablation Tests ==="

for spec in "${DATASET_SPECS[@]}"; do
    read -r train_size val_size test_size <<< "$spec"
    dataset_name="${PLANT_NAME}-${train_size}_${val_size}_${test_size}"
    run_dir_base="${ABLATION_ROOT}/${dataset_name}/sinkhorn_ablation"

    if [[ "$STAGE04_FORCE_RERUN" == "1" ]]; then
        run_dir="${run_dir_base}_rerun_$(date +%Y%m%d_%H%M%S)"
        log "Force rerun enabled; keeping prior ablation artifacts and using new run dir: ${run_dir}"
    else
        run_dir="$run_dir_base"
    fi

    optimizer_stage_dir="${run_dir}/optimizer"

    if [[ "$STAGE04_FORCE_RERUN" != "1" ]] && stage4_complete "$optimizer_stage_dir"; then
        log "Stage 4 already complete for ${dataset_name} (all expected optimizer outputs found). Skipping rerun."
        python3 optimizer_script.py \
            --summary_only \
            --run_dir "$run_dir" \
            --output_dir "$optimizer_stage_dir" | tee -a "$LOG_FILE" || {
                log "Summary-only aggregation failed for ablation optimizer outputs on $dataset_name"
        }
        continue
    fi

    for model_name in "${ABLATION_MODELS[@]}"; do
        stage_dir="${run_dir}/${model_name}"
        mkdir -m 777 -p "$stage_dir"
        log "Running ablation model: $model_name on dataset: $dataset_name -> $stage_dir"

        python3 tune_models.py \
            --dataset "$dataset_name" \
            --plant "$PLANT_NAME" \
            --models "$model_name" \
            --learning-rates "$SINKHORN_LR" \
            --batch-sizes "$SINKHORN_BS" \
            --replicates "$ABLATION_REPLICATES" \
            --dataset-fraction "$ABLATION_FRACTION" \
            --epochs "$ABLATION_EPOCHS" \
            --patience "$ABLATION_PATIENCE" \
            --opt-restarts "$OPT_RESTARTS" \
            --opt-steps "$OPT_STEPS" \
            --tuning-dir "$stage_dir" \
            --train-only \
            --resume | tee -a "$LOG_FILE" || {
                log "Ablation run failed: $model_name on $dataset_name"
                continue
        }

        log_tuning_metrics "$stage_dir/tuning_summary.csv" "$model_name"
    done

    mkdir -m 777 -p "$optimizer_stage_dir"

    missing_model_paths=()
    for model_name in "${ABLATION_MODELS[@]}"; do
        for rep in $(seq 1 "$ABLATION_REPLICATES"); do
            if optimizer_result_exists "$optimizer_stage_dir" "$model_name" "$rep"; then
                log "Optimization already complete: ${model_name}/Rep_${rep}"
                continue
            fi

            model_ckpt="${run_dir}/${model_name}/Rep_${rep}/final_model.pt"
            if [[ ! -f "$model_ckpt" ]]; then
                model_ckpt="${run_dir}/${model_name}/Rep_${rep}/best_model.pt"
            fi

            if [[ -f "$model_ckpt" ]]; then
                log "Optimization missing: ${model_name}/Rep_${rep} -> queued"
                missing_model_paths+=("$model_ckpt")
            else
                log "Optimization missing but no checkpoint found: ${model_name}/Rep_${rep}"
            fi
        done
    done

    if [[ "${#missing_model_paths[@]}" -eq 0 ]]; then
        log "All ablation optimization outputs already exist for $dataset_name. Skipping optimizer rerun."
    else
        log "Running optimizer for ${#missing_model_paths[@]} missing model replicate(s)."
        cmd=(python3 optimizer_script.py
            --plant "$PLANT_NAME"
            --run_dir "$run_dir"
            --output_dir "$optimizer_stage_dir"
            --restarts "$OPT_RESTARTS"
            --steps "$OPT_STEPS"
            --models sinkhorn)
        for model_path in "${missing_model_paths[@]}"; do
            cmd+=(--model "$model_path")
        done
        "${cmd[@]}" | tee -a "$LOG_FILE" || {
            log "Optimization failed for ablation summary on $dataset_name"
        }
    fi

    python3 optimizer_script.py \
        --summary_only \
        --run_dir "$run_dir" \
        --output_dir "$optimizer_stage_dir" | tee -a "$LOG_FILE" || {
            log "Summary-only aggregation failed for ablation optimizer outputs on $dataset_name"
    }
done

log "=== STAGE 4 COMPLETE ==="
