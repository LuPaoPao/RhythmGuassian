#!/bin/bash
# Reference launcher for RhythmGaussian (4D-rPPG).
# Edit the conda env name and STMap data root to match your machine.
set -e

gpu_ids=0

# Activate your environment, e.g.:
# source activate rhythmgs

# Point to your pre-processed STMap dataset root (must contain the per-dataset
# subdirectories listed in train.py:FILE_NAME).
export RPPG_DATA_ROOT="${RPPG_DATA_ROOT:-./Data/STMap/}"

python train.py -g "$gpu_ids" -t VIPL -rD 0
