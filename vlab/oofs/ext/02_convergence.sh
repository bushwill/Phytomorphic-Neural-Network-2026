# Stage 2: Quick Convergence
set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PLANT_NAME="Plant_023-1"
CONV_FRACTIONS=(0.25 0.5 1.0)
CONV_EPOCH_CONFIGS=("25 5" "50 10")
CONV_REPLICATES=2

# hp sets from step 1
MLP_LR="1e-3"
MLP_BS="1"
SINKHORN_LR="5e-4"
SINKHORN_BS="16"

OPT_RESTARTS=3
OPT_STEPS=250

CONV_DATASET_NAME="${PLANT_NAME}-50000_10000_25000"

QUICK_CONV_DIR="Convergence Tests/Convergence_${CONV_DATASET_NAME}"
LOG_FILE="Research Pipeline/02_convergence.log"
mkdir -m 777 -p "$(dirname "$LOG_FILE")" "$QUICK_CONV_DIR"

log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"; }

stage2_complete() {
    local summary_csv="${QUICK_CONV_DIR}/tuning_summary.csv"
    local expected=$(( ${#CONV_FRACTIONS[@]} * ${#CONV_EPOCH_CONFIGS[@]} * CONV_REPLICATES * 2 ))

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

log "=== STAGE 2: Quick Convergence ==="

if stage2_complete; then
    log "Stage 2 already complete (all expected convergence jobs found). Skipping rerun."
    log "=== STAGE 2 COMPLETE ==="
    exit 0
fi

for frac in "${CONV_FRACTIONS[@]}"; do
    for cfg in "${CONV_EPOCH_CONFIGS[@]}"; do
        read -r max_epochs patience <<< "$cfg"
        log "Convergence config: frac=${frac}, epochs=${max_epochs}, patience=${patience}"

        python3 tune_models.py \
            --dataset "$CONV_DATASET_NAME" \
            --plant "$PLANT_NAME" \
            --models mlp \
            --learning-rates "$MLP_LR" \
            --batch-sizes "$MLP_BS" \
            --replicates "$CONV_REPLICATES" \
            --dataset-fraction "$frac" \
            --epochs "$max_epochs" \
            --patience "$patience" \
            --opt-restarts "$OPT_RESTARTS" \
            --opt-steps "$OPT_STEPS" \
            --tuning-dir "$QUICK_CONV_DIR" \
            --resume | tee -a "$LOG_FILE"

        python3 tune_models.py \
            --dataset "$CONV_DATASET_NAME" \
            --plant "$PLANT_NAME" \
            --models sinkhorn \
            --learning-rates "$SINKHORN_LR" \
            --batch-sizes "$SINKHORN_BS" \
            --replicates "$CONV_REPLICATES" \
            --dataset-fraction "$frac" \
            --epochs "$max_epochs" \
            --patience "$patience" \
            --opt-restarts "$OPT_RESTARTS" \
            --opt-steps "$OPT_STEPS" \
            --tuning-dir "$QUICK_CONV_DIR" \
            --resume | tee -a "$LOG_FILE"
    done
done

python3 convergence_test.py --summary-file "${QUICK_CONV_DIR}/convergence_summary.txt" --tuning-dir "$QUICK_CONV_DIR" | tee -a "$LOG_FILE"

log "=== STAGE 2 COMPLETE ==="
