#!/usr/bin/env bash
# Soup the 22 members, re-estimate BatchNorm, export both networks to TorchScript and build the final package.
set -e
export CUDA_VISIBLE_DEVICES=${1:-0}
N="e4 e4s2 nocs e10 hw qual dp d1 d2 d3 d4 v2a v2b v2c v2d v2d10 v2a10 v2c10 v2e v2d6 v2all v2all2"
python tools/build_soup.py runs/soup22/final.pt $(for n in $N; do echo runs/final_vitl384_$n/final.pt; done)
# ViT soup: re-estimate BN on 4000 training crops, export with flip-TTA, 384x288, padding 1.1
python recalibrate_bn.py --checkpoint runs/soup22/final.pt --out runs/soup22/final_bn.pt --config configs/final_vitl384_e4.yaml \
  --backbone pretrained/vitpose-l-hf --input_size 384 288 --padding_factor 1.1 --n 4000 --batch 16
python export_submission.py --checkpoint runs/soup22/final_bn.pt --backbone pretrained/vitpose-l-hf \
  --out submission_vit_soup22.zip --tta_flip --input_size 384 288 --padding_factor 1.1
# HRNet partner: same procedure
python recalibrate_bn.py --checkpoint runs/final_hrnetpt384/final.pt --out runs/final_hrnetpt384/final_bn.pt --config configs/final_hrnetpt384.yaml \
  --backbone hrnet_w48_pt18 --input_size 384 288 --padding_factor 1.1 --n 4000 --batch 16
python export_submission.py --checkpoint runs/final_hrnetpt384/final_bn.pt --backbone hrnet_w48_pt18 \
  --out submission_hrnet.zip --tta_flip --input_size 384 288 --padding_factor 1.1
# Final package: heatmap fusion 0.8 / 0.2, power-mean (p=4) confidence over 13 joints
python build_ensemble.py --fusion heatmap --score_power 4 --out submission_final.zip \
  --member submission_vit_soup22.zip:0.8:1.1 --member submission_hrnet.zip:0.2:1.1
echo "built submission_final.zip"
