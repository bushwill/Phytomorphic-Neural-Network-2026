#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Phytomorphic Neural Network Orchestration Pipeline
# Ablation 2: Internal Module Ablation (Encoder, Scaler, Aggregator)
# ==============================================================================
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENT_TAG="${1:-$(date +%m%d%y_%H%M%S)}"
REPLICATES="${REPLICATES:-3}"

PLANTS=(
  "Plant_063-32"
)

# Ablation 2: Module Differences 
DATASET_SPECS=(
  "50000 5000 10000"
)

echo "Starting Phytomorphic Surrogate Pipeline Orchestration (Ablation 2)..."

for PLANT_NAME in "${PLANTS[@]}"; do
  
  OPT_ROOT_REL="Optimizer Data/Experiment_${EXPERIMENT_TAG}/${PLANT_NAME}"

  mkdir -m 777 -p "${OPT_ROOT_REL}"

  echo "=================================================="
  echo "Experiment Tag: ${EXPERIMENT_TAG}"
  echo "Processing Plant: ${PLANT_NAME}"
  echo "Replicates per model: ${REPLICATES}"
  echo "Optimizer root: ${OPT_ROOT_REL}"
  echo "=================================================="

  for spec in "${DATASET_SPECS[@]}"; do
    read -r train_size val_size test_size <<< "$spec"
    dataset_name="${PLANT_NAME}-${train_size}_${val_size}_${test_size}"
    dataset_dir="Datasets/${dataset_name}"
    training_run_name="Ablation2_${EXPERIMENT_TAG}_${dataset_name}"
    training_run_dir="Training Data/${training_run_name}"
    optimizer_out_rel="${OPT_ROOT_REL}/${dataset_name}"

    echo
    echo "=================================================="
    echo "Checking for dataset: ${dataset_name}..."
    
    if [[ ! -f "${dataset_dir}/Train.csv" || ! -f "${dataset_dir}/Validation.csv" || ! -f "${dataset_dir}/Test.csv" ]]; then
      echo "Dataset missing or incomplete. Automatically generating it now..."
      mkdir -m 777 -p "${dataset_dir}"
      python3 generate_dataset.py \
        --plant "$PLANT_NAME" \
        --train_size "$train_size" \
        --val_size "$val_size" \
        --test_size "$test_size" \
        --output_dir "${dataset_dir}"
    else
      echo "Dataset fully generated. Skipping generation step."
    fi

    echo
    echo "[1/2] Training ablation models on dataset: ${dataset_name}"
    python3 train_models.py \
      --dataset "$dataset_name" \
      --plant "$PLANT_NAME" \
      --run-name "$training_run_name" \
      --replicates "$REPLICATES" \
      --models sinkhorn_full sinkhorn_no_encoder sinkhorn_no_scaler sinkhorn_no_aggregator \
      --skip-evaluation

    echo
    echo "[2/2] Optimizing trained ablation models from: ${training_run_dir}"
    python3 optimizer_script.py \
      --run_dir "$training_run_dir" \
      --plant "$PLANT_NAME" \
      --models sinkhorn_full sinkhorn_no_encoder sinkhorn_no_scaler sinkhorn_no_aggregator \
      --output_dir "$optimizer_out_rel"

  done
done

echo "Ablation 2 experiments complete."
