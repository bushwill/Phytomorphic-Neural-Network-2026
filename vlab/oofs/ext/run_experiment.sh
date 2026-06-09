#!/bin/bash
set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENT_TAG="${1:-$(date +%m%d%y_%H%M%S)}"
LOG_FILE="Research Pipeline/run_experiment.log"
mkdir -m 777 -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

STAGES=(
    "00_dataset.sh"
    "01_hp_tuning.sh"
    "02_convergence.sh"
    "03_final_hp_tuning.sh"
    "04_ablation.sh"
    "05_experiment.sh"
)

log "=== Full Research Pipeline Start ==="
log "Experiment Tag: $EXPERIMENT_TAG"

for stage in "${STAGES[@]}"; do
    log "Running stage: $stage"
    if [[ "$stage" == "05_experiment.sh" ]]; then
        bash "./$stage" "$EXPERIMENT_TAG" | tee -a "$LOG_FILE"
    else
        bash "./$stage" | tee -a "$LOG_FILE"
    fi
    log "Completed stage: $stage"
done

log "=== Full Research Pipeline Complete ==="
