#!/usr/bin/env bash
set -euo pipefail

# Keep generated files/dirs editable and removable from host-mounted volumes.
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SMOKE_TAG="${SMOKE_TAG:-SMOKE_$(date +%m%d%H%M%S)}"
PLANT_NAME="${PLANT_NAME:-Plant_063-32}"
TRAIN_SIZE="${TRAIN_SIZE:-8}"
VAL_SIZE="${VAL_SIZE:-2}"
TEST_SIZE="${TEST_SIZE:-4}"
REPLICATES="${REPLICATES:-1}"
OPT_STEPS="${OPT_STEPS:-20}"
OPT_RESTARTS="${OPT_RESTARTS:-1}"
OPT_DRY_RUN="${OPT_DRY_RUN:-0}"

DATASET_OUT_REL="Datasets/${SMOKE_TAG}"
TRAIN_RUN_REL="Training Data/${SMOKE_TAG}"
OPT_OUT_REL="Optimizer Data/${SMOKE_TAG}"

on_error() {
  local exit_code=$?
  echo
  echo "[SMOKE] FAILED (exit ${exit_code}) at line ${BASH_LINENO[0]}"
  echo "[SMOKE] Tag: ${SMOKE_TAG}"
  exit "${exit_code}"
}
trap on_error ERR

echo "=========================================="
echo "[SMOKE] Tag: ${SMOKE_TAG}"
echo "[SMOKE] Plant: ${PLANT_NAME}"
echo "[SMOKE] Sizes: train=${TRAIN_SIZE} val=${VAL_SIZE} test=${TEST_SIZE}"
echo "[SMOKE] Replicates: ${REPLICATES}"
echo "[SMOKE] Optimizer: steps=${OPT_STEPS} restarts=${OPT_RESTARTS} dry_run=${OPT_DRY_RUN}"
echo "=========================================="

echo "[SMOKE 1/3] Generate tiny dataset"
python3 generate_dataset.py \
  --plant "$PLANT_NAME" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --output_dir "$DATASET_OUT_REL"

echo "[SMOKE 2/3] Train all model families"
python3 train_models.py \
  --dataset "$SMOKE_TAG" \
  --plant "$PLANT_NAME" \
  --run-name "$SMOKE_TAG" \
  --replicates "$REPLICATES" \
  --no-multiprocessing \
  --skip-evaluation

if [[ ! -d "$TRAIN_RUN_REL" ]]; then
  echo "[SMOKE] ERROR: Could not find training output directory."
  echo "[SMOKE] Expected: $TRAIN_RUN_REL"
  exit 1
fi

echo "[SMOKE] Training run dir: $TRAIN_RUN_REL"

echo "[SMOKE 3/3] Optimize trained models"
optimizer_args=(
  --run_dir "$TRAIN_RUN_REL"
  --models baseline hungarian sinkhorn
  --output_dir "$OPT_OUT_REL"
  --steps "$OPT_STEPS"
  --restarts "$OPT_RESTARTS"
)
if [[ "$OPT_DRY_RUN" == "1" ]]; then
  optimizer_args+=(--dry_run)
fi

python3 optimizer_script.py "${optimizer_args[@]}"

echo
echo "[SMOKE] PASSED"
echo "[SMOKE] Dataset: ${SCRIPT_DIR}/${DATASET_OUT_REL}"
echo "[SMOKE] Training run: ${TRAIN_RUN_REL}"
echo "[SMOKE] Optimizer output root: ${SCRIPT_DIR}/${OPT_OUT_REL}"
