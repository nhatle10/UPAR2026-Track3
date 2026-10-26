#!/usr/bin/env bash
# Train the 22 ViTPose-L soup members on all provided training annotations (fold `final`).
# Each member: one config + number of epochs. ~30 min/epoch on one 20 GB GPU. Usage: bash scripts/02_train_vit_members.sh [GPU]
set -e
export CUDA_VISIBLE_DEVICES=${1:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MEMBERS="e4:4 e4s2:4 nocs:4 e10:10 hw:4 qual:4 dp:4 d1:8 d2:6 d3:10 d4:5 v2a:4 v2b:4 v2c:4 v2d:4 v2d10:10 v2a10:10 v2c10:10 v2e:4 v2d6:6 v2all:6 v2all2:6"
for m in $MEMBERS; do
  n=${m%%:*}; ep=${m##*:}
  [ -f runs/final_vitl384_$n/final.pt ] && { echo "skip $n"; continue; }
  python train.py --config configs/final_vitl384_$n.yaml --fold final --epochs $ep
done
