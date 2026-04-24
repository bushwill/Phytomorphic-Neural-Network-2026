#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Phytomorphic Neural Network Orchestration Pipeline
#
# Stage 1: Procedural Simulation & Point Cloud Dataset Generation
# Stage 2: Surrogate Model Training (Baseline, Hungarian, Sinkhorn)
# Stage 3: Surrogate-Driven Hierarchical Parameter Optimization
#
# Forces global Umask to 000 so Docker-generated outputs remain editable by the 
# host Linux user within mounted repository volumes `Datasets/` and `/Optimizer Data`.
# ==============================================================================
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENT_TAG="${1:-$(date +%m%d%y_%H%M%S)}"
REPLICATES="${REPLICATES:-3}"

# Define the biological targets to reverse-model via L-system proxy.
PLANTS=(
  "Plant_023-1"
  "Plant_063-32"
  "Plant_191-28"
)

# Experimental scales mapping to: Train / Val / Test
# Validates scaling convergence and data-efficiency profiles.
DATASET_SPECS=(
  "1000 200 500"
)
# "10000 1000 2000"
# "50000 5000 10000"
echo "Starting Phytomorphic Surrogate Pipeline Orchestration..."

for PLANT_NAME in "${PLANTS[@]}"; do
  
  OPT_ROOT_REL="Optimizer Data/Experiment_${EXPERIMENT_TAG}/${PLANT_NAME}"

  # Allocate persistence trees and force native full read-write (-m 777)
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
    training_run_name="Experiment_${EXPERIMENT_TAG}_${dataset_name}"
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
    echo "[1/2] Training models on dataset: ${dataset_name}"
    python3 train_models.py \
      --dataset "$dataset_name" \
      --plant "$PLANT_NAME" \
      --run-name "$training_run_name" \
      --replicates "$REPLICATES" \
      --models baseline hungarian sinkhorn \
      --skip-evaluation

    if [[ -z "$training_run_dir" || ! -d "$training_run_dir" ]]; then
      echo "ERROR: Could not determine training output directory for ${dataset_name}."
      echo "Expected training directory: ${training_run_dir}"
      exit 1
    fi

    echo "Training run dir: ${training_run_dir}"

    echo
    echo "[2/2] Optimizing trained models from: ${training_run_dir}"
    python3 optimizer_script.py \
      --run_dir "$training_run_dir" \
      --plant "$PLANT_NAME" \
      --models baseline hungarian sinkhorn \
      --output_dir "$optimizer_out_rel"

  done
done

echo
echo "All experiments complete."
echo "Root Training runs directory: ${SCRIPT_DIR}/Training Data"
echo "Root Optimizer outputs: ${SCRIPT_DIR}/Optimizer Data/Experiment_${EXPERIMENT_TAG}"
