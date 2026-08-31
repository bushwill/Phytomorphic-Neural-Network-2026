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

# If set to 1, Stage 1 always starts a fresh HP tuning run for each dataset.
STAGE01_FORCE_RERUN=1

HP_MODELS=(mlp sinkhorn)
HP_LEARNING_RATES=(1e-3 5e-4 1e-4)
HP_BATCH_SIZES=(1 16 32)
HP_REPLICATES=3
HP_DATASET_FRACTION=1.0
HP_EPOCHS=100
HP_PATIENCE=5
HP_OPT_RESTARTS=3
HP_OPT_STEPS=250

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

find_tuning_dirs_for_dataset() {
    local dataset_name="$1"
    find "Hyperparameter Tuning" -maxdepth 1 -mindepth 1 -type d -name "Tuning_*_${dataset_name}_*" 2>/dev/null | sort
}

latest_tuning_dir_for_dataset() {
    local dataset_name="$1"
    local latest=""
    latest="$(find "Hyperparameter Tuning" -maxdepth 1 -mindepth 1 -type d -name "Tuning_*_${dataset_name}_*" -print0 2>/dev/null | xargs -0 -r ls -1td 2>/dev/null | head -n 1 || true)"
    [[ -n "$latest" ]] && echo "$latest"
}

dataset_hp_complete() {
    local tuning_dir="$1"
    local expected="$2"
    local summary_csv="${tuning_dir}/tuning_summary.csv"

    [[ -f "$summary_csv" ]] || return 1

    python3 - "$summary_csv" "$expected" <<'PY'
import csv
import sys

summary_path = sys.argv[1]
expected = int(sys.argv[2])

keys = set()
with open(summary_path, "r", newline="") as f:
    for row in csv.DictReader(f):
        model = (row.get("model") or "").strip()
        rep = str(row.get("replicate") or "").strip()
        lr = str(row.get("learning_rate") or "").strip()
        bs = str(row.get("batch_size") or "").strip()
        if model and rep and lr and bs:
            keys.add((model, rep, lr, bs))

sys.exit(0 if len(keys) >= expected else 1)
PY
}

print_test_r2_reports_for_dir() {
    local run_dir="$1"
    local dataset_name="$2"
    local count=0

    while IFS= read -r report_path; do
        [[ -z "$report_path" ]] && continue
        count=$((count + 1))
        log "Test R2 report file (${dataset_name}): ${report_path}"
        r2_line="$(grep -E '^r2=' "$report_path" | head -n 1 || true)"
        ss_tot_line="$(grep -E '^ss_tot=' "$report_path" | head -n 1 || true)"
        ss_res_line="$(grep -E '^ss_res=' "$report_path" | head -n 1 || true)"
        if [[ -n "$r2_line" || -n "$ss_tot_line" || -n "$ss_res_line" ]]; then
            log "  score_terms: ${r2_line:-r2=NA} | ${ss_tot_line:-ss_tot=NA} | ${ss_res_line:-ss_res=NA}"
        fi
    done < <(find "$run_dir" -type f -name "test_r2_report.txt" 2>/dev/null | sort)

    if [[ "$count" -eq 0 ]]; then
        log "No test_r2_report.txt files found in ${run_dir}"
    fi
}

log "=== STAGE 1: Baseline HP Tuning ==="
expected_runs=$(( ${#HP_MODELS[@]} * ${#HP_LEARNING_RATES[@]} * ${#HP_BATCH_SIZES[@]} * HP_REPLICATES ))

for spec in "${DATASET_SPECS[@]}"; do
    read -r train_size val_size test_size <<< "$spec"
    ensure_dataset_available "$PLANT_NAME" "$train_size" "$val_size" "$test_size"
    dataset_name="${PLANT_NAME}-${train_size}_${val_size}_${test_size}"

    mapfile -t dataset_dirs < <(find_tuning_dirs_for_dataset "$dataset_name")
    complete_dir=""
    for existing_dir in "${dataset_dirs[@]}"; do
        if dataset_hp_complete "$existing_dir" "$expected_runs"; then
            complete_dir="$existing_dir"
            break
        fi
    done

    run_dir=""
    resume_mode=0

    if [[ "$STAGE01_FORCE_RERUN" == "1" ]]; then
        log "Force rerun enabled for dataset: $dataset_name"
        log "Keeping existing tuning directories and starting a new run."
        run_dir="Hyperparameter Tuning/Tuning_$(date +%Y%m%d_%H%M%S)_${dataset_name}_mlp-sinkhorn"
    elif [[ -n "$complete_dir" ]]; then
        log "Baseline HP already complete for dataset: $dataset_name"
        log "Using existing complete run: $complete_dir"
        print_test_r2_reports_for_dir "$complete_dir" "$dataset_name"
        continue
    else
        latest_dir="$(latest_tuning_dir_for_dataset "$dataset_name")"
        if [[ -n "$latest_dir" ]]; then
            run_dir="$latest_dir"
            resume_mode=1
            log "Resuming existing partial HP run for dataset: $dataset_name"
            log "Resume directory: $run_dir"
        else
            run_dir="Hyperparameter Tuning/Tuning_$(date +%Y%m%d_%H%M%S)_${dataset_name}_mlp-sinkhorn"
            log "No prior run found for dataset: $dataset_name"
            log "Starting new HP run: $run_dir"
        fi
    fi

    mkdir -m 777 -p "$run_dir"

    cmd=(python3 tune_models.py
        --dataset "$dataset_name"
        --plant "$PLANT_NAME"
        --models "${HP_MODELS[@]}"
        --learning-rates "${HP_LEARNING_RATES[@]}"
        --batch-sizes "${HP_BATCH_SIZES[@]}"
        --replicates "$HP_REPLICATES"
        --dataset-fraction "$HP_DATASET_FRACTION"
        --epochs "$HP_EPOCHS"
        --patience "$HP_PATIENCE"
        --opt-restarts "$HP_OPT_RESTARTS"
        --opt-steps "$HP_OPT_STEPS"
        --tuning-dir "$run_dir")

    if [[ "$resume_mode" -eq 1 && "$STAGE01_FORCE_RERUN" != "1" ]]; then
        cmd+=(--resume)
        log "Running baseline HP tuning in resume mode for dataset: $dataset_name"
    else
        log "Running baseline HP tuning from start for dataset: $dataset_name"
    fi

    "${cmd[@]}" | tee -a "$LOG_FILE"
    print_test_r2_reports_for_dir "$run_dir" "$dataset_name"
done

log "=== STAGE 1 COMPLETE ==="
