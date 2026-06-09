#!/bin/bash
# ==============================================================================
# Research Pipeline Orchestrator
# ==============================================================================
# Runs numbered stage scripts in order:
#   00_dataset.sh
#   01_hp_tuning.sh
#   02_convergence.sh
#   03_final_hp_tuning.sh
#   04_ablation.sh
#
# Optional: choose stages via STAGES env var, e.g.
#   STAGES="02_convergence.sh 03_final_hp_tuning.sh" bash run_research_pipeline.sh

set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIPELINE_ROOT="Research Pipeline"
STATE_DIR="${PIPELINE_ROOT}/.state"
LOG_FILE="${PIPELINE_ROOT}/pipeline.log"
mkdir -m 777 -p "$PIPELINE_ROOT" "$STATE_DIR"

log() {
    local msg="$1"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $msg" | tee -a "$LOG_FILE"
}

mark_stage() {
    local stage="$1"
    local marker="${STATE_DIR}/${stage}.done"
    date +"%Y-%m-%d %H:%M:%S" > "$marker"
}

run_stage() {
    local script_name="$1"
    local stage_key="${script_name%.sh}"

    if [[ ! -f "$script_name" ]]; then
        log "ERROR: Missing stage script: $script_name"
        exit 1
    fi

    log "========== RUNNING ${script_name} =========="
    bash "$script_name" | tee -a "$LOG_FILE"
    mark_stage "$stage_key"
    log "========== COMPLETED ${script_name} =========="
}

DEFAULT_STAGES=(
    "00_dataset.sh"
    "01_hp_tuning.sh"
    "02_convergence.sh"
    "03_final_hp_tuning.sh"
    "04_ablation.sh"
)

if [[ -n "${STAGES:-}" ]]; then
    # shellcheck disable=SC2206
    STAGE_LIST=(${STAGES})
else
    STAGE_LIST=("${DEFAULT_STAGES[@]}")
fi

log "========== PIPELINE START =========="
log "Stages: ${STAGE_LIST[*]}"

for stage_script in "${STAGE_LIST[@]}"; do
    run_stage "$stage_script"
done

log "========== PIPELINE COMPLETE =========="
log "Central log: ${LOG_FILE}"
log "Per-stage logs: Research Pipeline/00_dataset.log, Research Pipeline/01_hp_tuning.log, Research Pipeline/02_convergence.log, Research Pipeline/03_final_hp_tuning.log, Research Pipeline/04_ablation.log"
