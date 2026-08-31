#!/bin/bash
# Stage 5: Full experiment orchestrator
set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENT_TAG="${1:-$(date +%m%d%y_%H%M%S)}"
REPLICATES=2
EXPERIMENT_EPOCHS=20

# If set to 1, Stage 5 always starts from scratch for this experiment tag.
STAGE05_FORCE_RERUN=0

# Best HP sets for Stage 5 training (edit these as needed).
MLP_HP_LR="1e-3"
MLP_HP_BS="8"
SINKHORN_HP_LR="5e-4"
SINKHORN_HP_BS="8"

PLANTS=(
    "Plant_001-9"
    "Plant_006-25"
    "Plant_008-19"
    "Plant_016-20"
    "Plant_023-1"
    "Plant_045-1"
    "Plant_047-25"
    "Plant_063-32"
    "Plant_070-11"
    "Plant_071-8"
    "Plant_076-24"
    "Plant_104-24"
    "Plant_191-28"
)

DATASET_SPECS=(
    "50000 10000 25000"
)

# Use standard sinkhorn for final experiments.
MODELS=(baseline sinkhorn)
OPT_RESTARTS=10
OPT_STEPS=1000

PIPELINE_ROOT="Research Pipeline"
LOG_FILE="${PIPELINE_ROOT}/05_experiment.log"
mkdir -m 777 -p "$PIPELINE_ROOT"

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

training_checkpoint_path() {
    local run_dir="$1"
    local model_name="$2"
    local replicate_id="$3"

    local ckpt="${run_dir}/${model_name}/Rep_${replicate_id}/final_model.pt"
    if [[ -f "$ckpt" ]]; then
        echo "$ckpt"
        return 0
    fi

    ckpt="${run_dir}/${model_name}/Rep_${replicate_id}/best_model.pt"
    if [[ -f "$ckpt" ]]; then
        echo "$ckpt"
        return 0
    fi

    return 1
}

training_complete() {
    local run_dir="$1"

    for model_name in "${MODELS[@]}"; do
        for rep in $(seq 1 "$REPLICATES"); do
            if ! training_checkpoint_path "$run_dir" "$model_name" "$rep" >/dev/null; then
                return 1
            fi
        done
    done
    return 0
}

optimizer_result_exists() {
    local optimizer_root="$1"
    local model_name="$2"
    local replicate_id="$3"

    if find "$optimizer_root" -type f -path "*/Run_*/${model_name}/Rep_${replicate_id}/opt_result.csv" -print -quit 2>/dev/null | grep -q .; then
        return 0
    fi
    if find "$optimizer_root" -type f -path "*/Run_*/${model_name}/*/Rep_${replicate_id}/opt_result.csv" -print -quit 2>/dev/null | grep -q .; then
        return 0
    fi
    return 1
}

dataset_complete() {
    local dataset_dir="$1"
    [[ -f "${dataset_dir}/Train.csv" && -f "${dataset_dir}/Validation.csv" && -f "${dataset_dir}/Test.csv" ]]
}

require_dataset() {
    local plant_name="$1"
    local train_size="$2"
    local val_size="$3"
    local test_size="$4"

    local dataset_name="${plant_name}-${train_size}_${val_size}_${test_size}"
    local dataset_dir="Datasets/${dataset_name}"

    if ! dataset_complete "$dataset_dir"; then
        log "ERROR: Required dataset missing: $dataset_name"
        log "Run 00_dataset.sh before Stage 5."
        exit 1
    fi

    log "Dataset present: $dataset_name"
}

log "=== STAGE 5: Full Experiment Orchestration ==="
log "Experiment Tag: $EXPERIMENT_TAG"
log "Plants: ${PLANTS[*]}"
log "Models: ${MODELS[*]}"
log "Replicates: $REPLICATES"
log "Epochs: $EXPERIMENT_EPOCHS"
log "MLP HP Set: lr=${MLP_HP_LR}, bs=${MLP_HP_BS}"
log "Sinkhorn HP Set: lr=${SINKHORN_HP_LR}, bs=${SINKHORN_HP_BS}"
log "Dataset Specs: ${DATASET_SPECS[*]}"

for plant_name in "${PLANTS[@]}"; do
    optimizer_root_rel="Optimizer Data/Experiment_${EXPERIMENT_TAG}/${plant_name}"
    mkdir -m 777 -p "$optimizer_root_rel"

    log "Processing plant: $plant_name"
    log "Optimizer root: $optimizer_root_rel"

    for spec in "${DATASET_SPECS[@]}"; do
        read -r train_size val_size test_size <<< "$spec"
        dataset_name="${plant_name}-${train_size}_${val_size}_${test_size}"
        rerun_suffix=""
        if [[ "$STAGE05_FORCE_RERUN" == "1" ]]; then
            rerun_suffix="_rerun_$(date +%Y%m%d_%H%M%S)"
        fi
        training_run_name="Experiment_${EXPERIMENT_TAG}_${dataset_name}${rerun_suffix}"
        training_run_dir="Training Data/${training_run_name}"
        optimizer_out_rel="${optimizer_root_rel}/${dataset_name}${rerun_suffix}"

        log "------------------------------------------------------------"
        log "Config: Plant=${plant_name} | Dataset=${dataset_name}"

        require_dataset "$plant_name" "$train_size" "$val_size" "$test_size"

        if [[ "$STAGE05_FORCE_RERUN" == "1" ]]; then
            log "Force rerun enabled; keeping prior artifacts and using new run paths for ${dataset_name}"
            log "New training run: ${training_run_dir}"
            log "New optimizer output: ${optimizer_out_rel}"
        fi

        if training_complete "$training_run_dir"; then
            log "[1/2] Training already complete for ${dataset_name}. Skipping retrain."
        else
            if [[ -d "$training_run_dir" ]]; then
                log "[1/2] Found partial training directory for ${dataset_name}: ${training_run_dir}"
                log "[1/2] Keeping existing partial artifacts and continuing with available checkpoints."
            else
                log "[1/2] Training models on dataset: ${dataset_name}"
                python3 train_models.py \
                    --dataset "$dataset_name" \
                    --plant "$plant_name" \
                    --run-name "$training_run_name" \
                    --replicates "$REPLICATES" \
                    --epochs "$EXPERIMENT_EPOCHS" \
                    --models "${MODELS[@]}" \
                    --mlp-learning-rate "$MLP_HP_LR" \
                    --mlp-batch-size "$MLP_HP_BS" \
                    --sinkhorn-learning-rate "$SINKHORN_HP_LR" \
                    --sinkhorn-batch-size "$SINKHORN_HP_BS" \
                    --skip-evaluation | tee -a "$LOG_FILE"
            fi
        fi

        if training_complete "$training_run_dir"; then
            log "Training run dir ready: ${training_run_dir}"
        else
            log "Training is still partial for ${dataset_name}; continuing with any available checkpoints."
        fi

        missing_model_paths=()
        for model_name in "${MODELS[@]}"; do
            for rep in $(seq 1 "$REPLICATES"); do
                if optimizer_result_exists "$optimizer_out_rel" "$model_name" "$rep"; then
                    log "[2/2] Optimization already complete: ${model_name}/Rep_${rep}"
                    continue
                fi

                ckpt_path="$(training_checkpoint_path "$training_run_dir" "$model_name" "$rep" || true)"
                if [[ -n "$ckpt_path" && -f "$ckpt_path" ]]; then
                    missing_model_paths+=("$ckpt_path")
                    log "[2/2] Optimization missing: ${model_name}/Rep_${rep} -> queued"
                else
                    log "[2/2] Optimization skipped (checkpoint missing): ${model_name}/Rep_${rep}"
                fi
            done
        done

        if [[ "${#missing_model_paths[@]}" -eq 0 ]]; then
            log "[2/2] Optimization already complete for ${dataset_name}. Skipping rerun."
        else
            log "[2/2] Running optimizer for ${#missing_model_paths[@]} missing model replicate(s)."
            cmd=(python3 optimizer_script.py
                --run_dir "$training_run_dir"
                --plant "$plant_name"
                --models "${MODELS[@]}"
                --output_dir "$optimizer_out_rel"
                --restarts "$OPT_RESTARTS"
                --steps "$OPT_STEPS")
            for model_path in "${missing_model_paths[@]}"; do
                cmd+=(--model "$model_path")
            done
            "${cmd[@]}" | tee -a "$LOG_FILE"
        fi
    done
done

log "[3/3] Building cross-plant summary CSV"
python3 optimizer_script.py \
    --summary_only \
    --run_dir "Optimizer Data/Experiment_${EXPERIMENT_TAG}" \
    --output_dir "Optimizer Data/Experiment_${EXPERIMENT_TAG}" | tee -a "$LOG_FILE"

log "=== STAGE 5 COMPLETE ==="
log "Root Training runs directory: ${SCRIPT_DIR}/Training Data"
log "Root Optimizer outputs: ${SCRIPT_DIR}/Optimizer Data/Experiment_${EXPERIMENT_TAG}"
