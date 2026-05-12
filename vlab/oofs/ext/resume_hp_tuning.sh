#!/bin/bash
# Resume an existing hyperparameter tuning run created by tune_models.py.
# Edit the variables below to match the run you want to resume, then run ./resume_hp_tuning.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Path to the tuning directory you want to resume (edit this if needed)
TUNING_DIR="${SCRIPT_DIR}/Hyperparameter Tuning/Tuning_20260428_021852_Plant_023-1-20000_4000_10000_mlp-sinkhorn"

# Dataset and plant arguments (edit if needed)
DATASET="Plant_023-1-20000_4000_10000"
PLANT="Plant_023-1"

# Models and grid (edit if you changed these when you started the run)
MODELS=(mlp sinkhorn)
LEARNING_RATES=(0.001 0.0005 0.0001)
BATCH_SIZES=(1 16 32)

# Other tuning args (edit to match original run)
REPLICATES=3
DATASET_FRACTION=1.0
EPOCHS=100
PATIENCE=5
OPT_RESTARTS=3
OPT_STEPS=250

cd "${SCRIPT_DIR}" || exit 1

python3 tune_models.py \
  --tuning-dir "${TUNING_DIR}" \
  --resume \
  --dataset "${DATASET}" \
  --plant "${PLANT}" \
  --models "${MODELS[@]}" \
  --learning-rates "${LEARNING_RATES[@]}" \
  --batch-sizes "${BATCH_SIZES[@]}" \
  --replicates ${REPLICATES} \
  --dataset-fraction ${DATASET_FRACTION} \
  --epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --opt-restarts ${OPT_RESTARTS} \
  --opt-steps ${OPT_STEPS}

exit $?
