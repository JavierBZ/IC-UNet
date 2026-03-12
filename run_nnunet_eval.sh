#!/usr/bin/env bash
# =============================================================================
# run_nnunet_eval.sh
# nnUNet inference + TTA post-processing for IC-UNet (Dataset100)
#
# Usage:
#   bash run_nnunet_eval.sh
#
# Configure the variables below before running.
# Requires nnUNet to be installed and the following env vars set:
#   nnUNet_raw, nnUNet_preprocessed, nnUNet_results
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
DATASET="Dataset100_ICAD"
TRAINER="nnUNetTrainerNoMirroring"
PLANS="nnUNetResEncUNetLPlans"
CONFIG="3d_fullres"

# Folds to evaluate (space-separated, e.g. "0 1 2 3 4" for full CV)
FOLDS="0 1 2 3 4"

# Number of TTA versions (0-indexed, script will run tta0 .. tta{N_TTA-1})
N_TTA=7

# Path to nnUNet_models directory containing per-fold checkpoints
# Expected layout: nnUNet_models/f{fold}/checkpoint_final.pth
CKPT_DIR="nnUNet_models"

# ---------------------------------------------------------------------------
# Derived paths (no need to edit below this line)
# ---------------------------------------------------------------------------
RESULTS_BASE="nnUNet_results/${DATASET}/${TRAINER}__${PLANS}__${CONFIG}"
RAW_BASE="nnUNet_raw/${DATASET}"

# ---------------------------------------------------------------------------
# Step 1: Run nnUNet inference for each fold x TTA combination
# ---------------------------------------------------------------------------
for fold in $FOLDS; do
    for tta_idx in $(seq 0 $((N_TTA - 1))); do
        INPUT_DIR="${RAW_BASE}/imagesTs_fold_${fold}_test_tta${tta_idx}"
        OUTPUT_DIR="${RESULTS_BASE}/tta${tta_idx}_f${fold}"
        CKPT="${CKPT_DIR}/f${fold}/checkpoint_final.pth"

        if [ ! -d "$INPUT_DIR" ]; then
            echo "[SKIP] Input not found: $INPUT_DIR"
            continue
        fi

        echo ">>> fold=${fold}  tta=${tta_idx}"
        echo "    input:  $INPUT_DIR"
        echo "    output: $OUTPUT_DIR"
        echo "    ckpt:   $CKPT"

        nnUNetv2_predict \
            -d "$DATASET" \
            -i "$INPUT_DIR" \
            -o "$OUTPUT_DIR" \
            -f "$fold" \
            -tr "$TRAINER" \
            -c "$CONFIG" \
            -p "$PLANS" \
            --disable_tta \
            -chk checkpoint_final.pth

        echo ""
    done
done

# ---------------------------------------------------------------------------
# Step 2: TTA post-processing (aggregation across TTA passes per fold)
# ---------------------------------------------------------------------------
echo ">>> Running TTA post-processing with custom_tta_nnunet.py ..."
python custom_tta_nnunet.py

echo ">>> Done."