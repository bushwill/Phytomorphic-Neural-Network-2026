#!/bin/bash

# ==============================================================================
# End-to-End Hyperparameter Tuning & Convergence Pipeline
# ==============================================================================
# This script automates running the tune_models.py script, allowing multi-plant,
# multi-model, and multi-fraction testing for surrogate optimization convergence.

# --- 1. CONFIGURATION ---

# Define the biological targets to target for the structure reference
PLANTS=(
    "Plant_023-1"
    "Plant_063-32"
    "Plant_191-28"
)

# Experimental scales mapping to: Train Val Test
DATASET_SPECS=(
    "1000 200 500"
    "20000 4000 10000"
)

# Models to evaluate
# Options: mlp sinkhorn hungarian
MODELS="mlp sinkhorn"

# Hyperparameter search grid
# List space-separated combinations you want tested per model
LEARNING_RATES="1e-3 5e-4 1e-4"
BATCH_SIZES="1 16 32"

# Number of Replicate runs per configuration for statistical smoothing
REPLICATES=3

# Epochs and Patience (Early stopping based on true LPFG cost)
MAX_EPOCHS=100
PATIENCE=5

# Optimizer parameters for the lightweight mid-training check
OPT_RESTARTS=3
OPT_STEPS=250

# Fractions of the training set to use (e.g. 0.25 = 25% of training data)
FRACTIONS=("1.0")

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
set -e # Exit immediately on error
umask 000 # Forces global Umask to 000 so outputs remain editable by host user

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Starting End-to-End Tuning and Convergence Test Pipeline"
echo "============================================================"
echo "Plants:        ${PLANTS[*]}"
echo "Dataset Specs: ${DATASET_SPECS[*]}"
echo "Models:        $MODELS"
echo "Grid LR:       $LEARNING_RATES"
echo "Grid Batch:    $BATCH_SIZES"
echo "Fractions:     ${FRACTIONS[*]}"
echo "Replicates:    $REPLICATES"
echo "Patience:      $PATIENCE"
echo "============================================================"

for PLANT_NAME in "${PLANTS[@]}"; do
    for spec in "${DATASET_SPECS[@]}"; do
        read -r train_size val_size test_size <<< "$spec"
        dataset_name="${PLANT_NAME}-${train_size}_${val_size}_${test_size}"
        dataset_dir="Datasets/${dataset_name}"
        
        # --- Stage 1: Dataset Generation ---
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

        for FRAC in "${FRACTIONS[@]}"; do
            echo "------------------------------------------------------------"
            echo " Running Config -> Plant: $PLANT_NAME | Dataset: $dataset_name | Fraction: $FRAC"
            echo "------------------------------------------------------------"
            
            # Execute Python Script with unquoted $MODELS so it expands to multiple arguments
            python3 tune_models.py \
                --dataset "$dataset_name" \
                --plant "$PLANT_NAME" \
                --models $MODELS \
                --learning-rates $LEARNING_RATES \
                --batch-sizes $BATCH_SIZES \
                --replicates "$REPLICATES" \
                --dataset-fraction "$FRAC" \
                --epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --opt-restarts "$OPT_RESTARTS" \
                --opt-steps "$OPT_STEPS"
                
            sleep 2
        done
    done
done

echo ""
echo "============================================================"
echo " Pipeline Complete! Ensure you check the tuning_summary.csv"
echo " logs inside the Hyperparameter Tuning/Tuning_* folders."
echo "============================================================"
