#!/bin/bash

# ==============================================================================
# Convergence Analysis Pipeline
# ==============================================================================
# This script systematically tests convergence of both MLP and Sinkhorn models
# by varying data fractions and epoch counts to determine optimal training requirements.
#
# Purpose:
#   - Find minimum data required for model convergence
#   - Find optimal epoch count (early stopping effectiveness)
#   - Compare convergence behavior between MLP and Sinkhorn
#   - Identify overfitting onset points using SAME dataset (fair comparison)
#
# Methodology:
#   - Uses SINGLE large dataset (20000 training samples)
#   - Tests fractions (10%, 25%, 50%, 75%, 100%) of same dataset
#   - Allows cross-comparison of validation and test performance
#   - Uses BEST HP set for each model (from HP tuning stage)

set -e
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Use same plant as HP tuning for consistency
PLANT_NAME="Plant_023-1"

# Use SINGLE dataset for convergence testing (allows fair cross-comparison)
# Using large dataset size for better statistical representation
DATASET_SIZE="20000 4000 10000"  # Train Val Test
read -r TRAIN_SIZE VAL_SIZE TEST_SIZE <<< "$DATASET_SIZE"
DATASET_NAME="${PLANT_NAME}-${TRAIN_SIZE}_${VAL_SIZE}_${TEST_SIZE}"

# Check for dataset in original location first, then convergence tests location
ORIGINAL_DATASET_DIR="Datasets/${DATASET_NAME}"
CONVERGENCE_DATASET_DIR="Convergence Tests/Datasets/${DATASET_NAME}"

# Use original location if it exists, otherwise use convergence tests location
if [[ -f "${ORIGINAL_DATASET_DIR}/Train.csv" ]]; then
    DATASET_DIR="$ORIGINAL_DATASET_DIR"
else
    DATASET_DIR="$CONVERGENCE_DATASET_DIR"
fi

# Best HP sets from tuning (one per model)
# MLP best: lr0.001_bs1 (R²=0.988128±0.001123, cost=57,327±5,331 on large data)
# Sinkhorn best: lr0.0005_bs16 (R²=0.970142±0.007185, cost=59,346±405 on large data)
MLP_LR="1e-3"
MLP_BS="1"
SINKHORN_LR="5e-4"
SINKHORN_BS="16"

# Fractions of training set to test convergence
# Tests how model scales with data quantity from same distribution
FRACTIONS=(0.1 0.25 0.5 0.75 1.0)

# Epoch configurations to test convergence behavior
# Tests where early stopping is effective and overfitting onset
EPOCH_CONFIGS=(
    "10 3"    # Early: 10 epochs, patience 3
    "25 5"    # Medium: 25 epochs, patience 5
    "50 10"   # Full: 50 epochs, patience 10
)

# Number of replicates for statistical stability
REPLICATES=3

# Optimizer parameters (same as HP tuning)
OPT_RESTARTS=3
OPT_STEPS=250

# ==============================================================================
# LOGGING & OUTPUT
# ==============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONVERGENCE_DIR="Convergence Tests/Convergence_${TIMESTAMP}_${PLANT_NAME}"
mkdir -m 777 -p "$CONVERGENCE_DIR"

# Convergence tuning results go directly in convergence folder (no HP Tuning subdirectory)
TUNING_BASE_DIR="$CONVERGENCE_DIR"

# Verify directory was created
if [[ ! -d "$CONVERGENCE_DIR" ]]; then
    echo "ERROR: Failed to create convergence directory: $CONVERGENCE_DIR"
    exit 1
fi

echo "✓ Created convergence directory: $CONVERGENCE_DIR"

CONVERGENCE_SUMMARY="${CONVERGENCE_DIR}/convergence_summary.txt"

echo ""
echo "============================================================"
echo "        CONVERGENCE ANALYSIS PIPELINE"
echo "============================================================"
echo "Plant:              $PLANT_NAME"
echo "Dataset:            $DATASET_NAME (${TRAIN_SIZE} training samples)"
echo "Data Fractions:     ${FRACTIONS[@]}"
echo "Epoch Configs:      ${#EPOCH_CONFIGS[@]} configurations"
echo "MLP Config:         lr${MLP_LR} bs${MLP_BS}"
echo "Sinkhorn Config:    lr${SINKHORN_LR} bs${SINKHORN_BS}"
echo "Replicates:         $REPLICATES"
echo "Output Dir:         $CONVERGENCE_DIR"
echo "Tuning Dir:         $TUNING_BASE_DIR"
echo "============================================================"
echo ""

# ==============================================================================
# MAIN EXECUTION LOOP
# ==============================================================================

# Generate dataset once for all convergence tests
echo "────────────────────────────────────────────────────────────"
echo "Dataset: $DATASET_NAME (Train: $TRAIN_SIZE | Val: $VAL_SIZE | Test: $TEST_SIZE)"
echo "────────────────────────────────────────────────────────────"

if [[ ! -f "${DATASET_DIR}/Train.csv" ]] || [[ ! -f "${DATASET_DIR}/Validation.csv" ]] || [[ ! -f "${DATASET_DIR}/Test.csv" ]]; then
    echo "  [DATASET] Generating dataset..."
    mkdir -m 777 -p "${DATASET_DIR}"
    python3 generate_dataset.py \
        --plant "$PLANT_NAME" \
        --train_size "$TRAIN_SIZE" \
        --val_size "$VAL_SIZE" \
        --test_size "$TEST_SIZE" \
        --output_dir "${DATASET_DIR}"
else
    echo "  [DATASET] Using existing dataset"
fi

echo ""

# Test each fraction and epoch configuration
for frac in "${FRACTIONS[@]}"; do
    for epoch_config in "${EPOCH_CONFIGS[@]}"; do
        read -r max_epochs patience <<< "$epoch_config"
        
        frac_percent=$(awk "BEGIN {printf \"%.0f\", $frac * 100}")
        
        echo "────────────────────────────────────────────────────────────"
        echo "Config: Fraction=${frac_percent}% | Epochs=${max_epochs} | Patience=${patience}"
        echo "────────────────────────────────────────────────────────────"
        
        # Test MLP with best HP set
        echo "  [MLP] Testing with lr${MLP_LR}_bs${MLP_BS}..."
        python3 tune_models.py \
            --dataset "$DATASET_NAME" \
            --plant "$PLANT_NAME" \
            --models mlp \
            --learning-rates "$MLP_LR" \
            --batch-sizes "$MLP_BS" \
            --replicates "$REPLICATES" \
            --dataset-fraction "$frac" \
            --epochs "$max_epochs" \
            --patience "$patience" \
            --opt-restarts "$OPT_RESTARTS" \
            --opt-steps "$OPT_STEPS" \
            --tuning-dir "$TUNING_BASE_DIR" || {
            echo "  [ERROR] MLP test failed"
            exit 1
        }
        
        sleep 1
        
        # Test Sinkhorn with best HP set
        echo "  [SINKHORN] Testing with lr${SINKHORN_LR}_bs${SINKHORN_BS}..."
        python3 tune_models.py \
            --dataset "$DATASET_NAME" \
            --plant "$PLANT_NAME" \
            --models sinkhorn \
            --learning-rates "$SINKHORN_LR" \
            --batch-sizes "$SINKHORN_BS" \
            --replicates "$REPLICATES" \
            --dataset-fraction "$frac" \
            --epochs "$max_epochs" \
            --patience "$patience" \
            --opt-restarts "$OPT_RESTARTS" \
            --opt-steps "$OPT_STEPS" \
            --tuning-dir "$TUNING_BASE_DIR" || {
            echo "  [ERROR] Sinkhorn test failed"
            exit 1
        }
        
        sleep 1
        
        echo ""
    done
done

echo ""
echo "Checking convergence directory contents after tuning runs:"
find "$CONVERGENCE_DIR" -type f -name "tuning_summary.csv" 2>/dev/null | head -20 || echo "  (No tuning_summary.csv files found)"
echo ""

# ==============================================================================
# POST-PROCESSING: GENERATE CONVERGENCE SUMMARY
# ==============================================================================

echo ""
echo "============================================================"
echo "  Generating Convergence Summary"
echo "============================================================"
echo ""

export CONVERGENCE_SUMMARY
export TUNING_BASE_DIR

python3 convergence_test.py --summary-file "$CONVERGENCE_SUMMARY" --tuning-dir "$TUNING_BASE_DIR"

# ==============================================================================
# FINAL REPORT
# ==============================================================================

echo ""
echo "============================================================"
echo "          CONVERGENCE ANALYSIS COMPLETE"
echo "============================================================"
echo "Output Directory:   $CONVERGENCE_DIR"
echo "Summary Report:     $CONVERGENCE_SUMMARY"
echo "============================================================"
echo ""
