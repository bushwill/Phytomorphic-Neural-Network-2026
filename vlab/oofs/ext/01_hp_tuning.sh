#!/bin/bash
# Stage 1: Baseline HP Tuning
set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# User config (same as research pipeline)
PLANT_NAME="Plant_023-1"
DATASET_SPECS=(
    "1000 200 500"
    "20000 4000 10000"
)

RUN_BASE_HP_IF_MISSING="${RUN_BASE_HP_IF_MISSING:-0}"

PIPELINE_ROOT="Research Pipeline"
STATE_DIR="${PIPELINE_ROOT}/.state"
LOG_FILE="${PIPELINE_ROOT}/01_hp_tuning.log"
mkdir -m 777 -p "$PIPELINE_ROOT" "$STATE_DIR"

log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"; }

dataset_complete() { local d="$1"; [[ -f "${d}/Train.csv" && -f "${d}/Validation.csv" && -f "${d}/Test.csv" ]]; }

ensure_dataset_available() {
    local plant="$1"; local train_size="$2"; local val_size="$3"; local test_size="$4"
    local dataset_name="${plant}-${train_size}_${val_size}_${test_size}"
    local main_dir="Datasets/${dataset_name}"
    local alt_dir="Convergence Tests/Datasets/${dataset_name}"

    if dataset_complete "$main_dir"; then
        log "Dataset ready in main store: $main_dir"
        return 0
    fi
    if dataset_complete "$alt_dir"; then
        log "Copying dataset from alternate store: $alt_dir -> $main_dir"
        mkdir -m 777 -p "$main_dir" && cp -a "${alt_dir}/." "$main_dir/"
        return 0
    fi

    log "ERROR: Missing required dataset: $dataset_name"
    log "Run 00_dataset.sh before this stage."
    exit 1
}

log "=== STAGE 1: Baseline HP Tuning ==="
base_hp_found=1
for spec in "${DATASET_SPECS[@]}"; do
    read -r train_size val_size test_size <<< "$spec"
    ensure_dataset_available "$PLANT_NAME" "$train_size" "$val_size" "$test_size"
    dataset_name="${PLANT_NAME}-${train_size}_${val_size}_${test_size}"
    if ! find "Hyperparameter Tuning" -maxdepth 2 -type f -name "tuning_summary.csv" 2>/dev/null | grep -q "$dataset_name"; then
        base_hp_found=0
        log "Baseline HP results not found for dataset: $dataset_name"
    else
        log "Baseline HP results found for dataset: $dataset_name"
    fi
done

if [[ "$base_hp_found" -eq 1 ]]; then
    log "Baseline HP stage already complete. Skipping rerun."
elif [[ "$RUN_BASE_HP_IF_MISSING" == "1" ]]; then
    log "Running full baseline HP tuning because RUN_BASE_HP_IF_MISSING=1"
    bash ./run_hp_tuning.sh | tee -a "$LOG_FILE"
else
    log "Skipping baseline HP rerun (set RUN_BASE_HP_IF_MISSING=1 to force)."
fi

log "=== STAGE 1 COMPLETE ==="
