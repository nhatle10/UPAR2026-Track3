#!/usr/bin/env bash
# Fine-tune the HRNet-W48 (COCO -> PoseTrack18) partner for 10 epochs on all training annotations.
set -e
export CUDA_VISIBLE_DEVICES=${1:-0}
python train.py --config configs/final_hrnetpt384.yaml --fold final --epochs 10
