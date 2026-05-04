#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RhythmGuassian launcher (4D-rPPG).
#
# Usage:
#   ./run.sh                  # run with the defaults below
#   GPU=1 TARGET=V4V ./run.sh # override a single setting
#   ./run.sh -t V4V -mi 5000  # forward extra flags to train.py
#
# First-time setup:
#   1. Activate the conda env that has PyTorch + 2DGS rasterizer installed.
#   2. Point RPPG_DATA_ROOT to your pre-processed STMap directory
#      (must contain the per-dataset folders listed in train.py:FILE_NAME).
#   3. Set REDATA=1 to (re)build the STMap index on the first run; revert to 0
#      after the index files exist under $RPPG_DATA_ROOT/STMap_Index/.
# ---------------------------------------------------------------------------
set -euo pipefail

# ----- User-tunable settings -----------------------------------------------
GPU="${GPU:-0}"                                    # CUDA device index
TARGET="${TARGET:-VIPL}"                           # leave-one-out target domain
REDATA="${REDATA:-0}"                              # 1 = rebuild STMap index
MAX_ITER="${MAX_ITER:-20000}"                      # total training iterations
BATCH_SIZE="${BATCH_SIZE:-32}"                     # per-source batch size
NUM_WORKERS="${NUM_WORKERS:-8}"                    # DataLoader workers

# Path to the pre-processed STMap dataset root.
export RPPG_DATA_ROOT="${RPPG_DATA_ROOT:-./Data/STMap/}"

# Optional: activate your conda env here.
# source activate rhythmgs

# ----- Sanity checks --------------------------------------------------------
if [[ ! -d "$RPPG_DATA_ROOT" ]]; then
    echo "[run.sh] WARNING: RPPG_DATA_ROOT='$RPPG_DATA_ROOT' does not exist."
    echo "         Either create it or export RPPG_DATA_ROOT to your STMap path."
fi

# ----- Launch ---------------------------------------------------------------
echo "[run.sh] GPU=$GPU TARGET=$TARGET REDATA=$REDATA"
echo "[run.sh] RPPG_DATA_ROOT=$RPPG_DATA_ROOT"

python train.py \
    -g  "$GPU" \
    -t  "$TARGET" \
    -rD "$REDATA" \
    -mi "$MAX_ITER" \
    -b  "$BATCH_SIZE" \
    -p  "$NUM_WORKERS" \
    "$@"
