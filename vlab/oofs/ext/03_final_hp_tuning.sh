# Stage 3: Quick Final HP Tuning
set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PLANT_NAME="Plant_023-1"
DATASET_SPECS=(
    "50000 10000 25000"
)

FINAL_HP_REPLICATES=2
FINAL_HP_EPOCHS=50
FINAL_HP_PATIENCE=10
FINAL_HP_FRACTION=1.0

FINAL_MLP_LRS=(1e-3 5e-4)
FINAL_MLP_BS=(1 4 8)
FINAL_SINKHORN_LRS=(5e-4 1e-3)
FINAL_SINKHORN_BS=(8 16 32)

OPT_RESTARTS=3
OPT_STEPS=250

FINAL_HP_ROOT="Final HP Tune/${PLANT_NAME}"
LOG_FILE="Research Pipeline/03_final_hp_tuning.log"
mkdir -m 777 -p "${FINAL_HP_ROOT}" "$(dirname "$LOG_FILE")"

log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"; }

stage3_complete() {
    local summary_csv="${FINAL_HP_ROOT}/${PLANT_NAME}-50000_10000_25000/tuning_summary.csv"
    local expected=$(( (${#FINAL_MLP_LRS[@]} * ${#FINAL_MLP_BS[@]} + ${#FINAL_SINKHORN_LRS[@]} * ${#FINAL_SINKHORN_BS[@]}) * FINAL_HP_REPLICATES ))

    [[ -f "$summary_csv" ]] || return 1

    python3 - "$summary_csv" "$expected" <<'PY'
import csv
import sys

summary_path = sys.argv[1]
expected = int(sys.argv[2])

pairs = set()
with open(summary_path, "r", newline="") as f:
    for row in csv.DictReader(f):
        model = row.get("model", "")
        rep = str(row.get("replicate", ""))
        if model and rep and (model.startswith("mlp_") or model.startswith("sinkhorn_")):
            pairs.add((model, rep))

sys.exit(0 if len(pairs) >= expected else 1)
PY
}

log "=== STAGE 3: Quick Final HP Tune ==="

if stage3_complete; then
    log "Stage 3 already complete (all expected final HP jobs found). Skipping rerun."
    log "=== STAGE 3 COMPLETE ==="
    exit 0
fi

for spec in "${DATASET_SPECS[@]}"; do
    read -r train_size val_size test_size <<< "$spec"
    dataset_name="${PLANT_NAME}-${train_size}_${val_size}_${test_size}"
    stage_dir="${FINAL_HP_ROOT}/${dataset_name}"
    mkdir -m 777 -p "$stage_dir"
    log "Final HP quick tune dataset: $dataset_name"

    python3 tune_models.py \
        --dataset "$dataset_name" \
        --plant "$PLANT_NAME" \
        --models mlp \
        --learning-rates "${FINAL_MLP_LRS[@]}" \
        --batch-sizes "${FINAL_MLP_BS[@]}" \
        --replicates "$FINAL_HP_REPLICATES" \
        --dataset-fraction "$FINAL_HP_FRACTION" \
        --epochs "$FINAL_HP_EPOCHS" \
        --patience "$FINAL_HP_PATIENCE" \
        --opt-restarts "$OPT_RESTARTS" \
        --opt-steps "$OPT_STEPS" \
        --tuning-dir "$stage_dir" \
        --resume | tee -a "$LOG_FILE"

    python3 tune_models.py \
        --dataset "$dataset_name" \
        --plant "$PLANT_NAME" \
        --models sinkhorn \
        --learning-rates "${FINAL_SINKHORN_LRS[@]}" \
        --batch-sizes "${FINAL_SINKHORN_BS[@]}" \
        --replicates "$FINAL_HP_REPLICATES" \
        --dataset-fraction "$FINAL_HP_FRACTION" \
        --epochs "$FINAL_HP_EPOCHS" \
        --patience "$FINAL_HP_PATIENCE" \
        --opt-restarts "$OPT_RESTARTS" \
        --opt-steps "$OPT_STEPS" \
        --tuning-dir "$stage_dir" \
        --resume | tee -a "$LOG_FILE"

    python3 convergence_test.py --summary-file "${stage_dir}/final_hp_summary.txt" --tuning-dir "$stage_dir" | tee -a "$LOG_FILE"
done

log "=== STAGE 3 COMPLETE ==="
