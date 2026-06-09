# Stage 0: Dataset preparation for preliminary tests
set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PRELIM_PLANT="Plant_023-1"

FINAL_PLANTS=(
    "Plant_001-9"
    "Plant_006-25"
    "Plant_008-19"
    "Plant_016-20"
    "Plant_023-1"
    "Plant_045-1"
    "Plant_047-25"
    "Plant_063-32"
    "Plant_070-11"
    "Plant_071-8"
    "Plant_076-24"
    "Plant_104-24"
    "Plant_191-28"
)

# Datasets needed by the numbered stages:
# - 01_hp_tuning.sh            -> 1k and 20k
# - 02_convergence.sh          -> 50k (covered by FINAL_PLANTS)
# - 03_final_hp_tuning.sh      -> 50k
# - 04_ablation.sh             -> 50k
PRELIM_DATASET_SPECS=(
    "1000 200 500"
    "20000 4000 10000"
)

FINAL_DATASET_SPEC="50000 10000 25000"

LOG_FILE="Research Pipeline/00_dataset.log"
mkdir -m 777 -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

dataset_complete() {
    local d="$1"
    [[ -f "${d}/Train.csv" && -f "${d}/Validation.csv" && -f "${d}/Test.csv" ]]
}

ensure_dataset() {
    local plant_name="$1"
    local train_size="$2"
    local val_size="$3"
    local test_size="$4"

    local dataset_name="${plant_name}-${train_size}_${val_size}_${test_size}"
    local main_dir="Datasets/${dataset_name}"
    local alt_dir="Convergence Tests/Datasets/${dataset_name}"

    if dataset_complete "$main_dir"; then
        log "Dataset already present: $main_dir"
        return 0
    fi

    if dataset_complete "$alt_dir"; then
        log "Dataset already present in alternate store: $alt_dir -> $main_dir"
        mkdir -m 777 -p "$main_dir"
        cp -a "${alt_dir}/." "$main_dir/"
        return 0
    fi

    log "Generating dataset: $dataset_name"
    mkdir -m 777 -p "$main_dir"
    python3 generate_dataset.py \
        --plant "$plant_name" \
        --train_size "$train_size" \
        --val_size "$val_size" \
        --test_size "$test_size" \
        --output_dir "$main_dir" | tee -a "$LOG_FILE"
}

log "=== STAGE 0: Dataset Prep ==="
for spec in "${PRELIM_DATASET_SPECS[@]}"; do
    read -r train_size val_size test_size <<< "$spec"
    ensure_dataset "$PRELIM_PLANT" "$train_size" "$val_size" "$test_size"
done

read -r final_train final_val final_test <<< "$FINAL_DATASET_SPEC"
for plant_name in "${FINAL_PLANTS[@]}"; do
    ensure_dataset "$plant_name" "$final_train" "$final_val" "$final_test"
done

log "=== STAGE 0 COMPLETE ==="
