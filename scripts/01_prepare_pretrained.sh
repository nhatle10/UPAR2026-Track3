#!/usr/bin/env bash
# Prepare the permitted pretrained weights in ./pretrained (ImageNet-1K / COCO / PoseTrack18 only).
set -e
mkdir -p pretrained
# 1) ViTPose-L, original release (MAE ImageNet-1K -> COCO, classic decoder). Download `vitpose-l.pth` manually from
#    https://github.com/ViTAE-Transformer/ViTPose (MS COCO table, ViTPose-L, classic decoder) and place it in ./pretrained.
test -f pretrained/vitpose-l.pth || { echo "missing pretrained/vitpose-l.pth (see comment above)"; exit 1; }
python tools/convert_vitpose_l.py pretrained/vitpose-l.pth pretrained/vitpose-l-hf     # -> HuggingFace VitPose format
# 2) HRNet-W48, ImageNet-1K -> COCO -> PoseTrack18 (MMPose)
wget -nc -O pretrained/hrnet_w48_posetrack18_384x288.pth \
  https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288-5fd6d3ff_20211130.pth
python tools/clean_mmpose_ckpt.py pretrained/hrnet_w48_posetrack18_384x288.pth pretrained/hrnet_w48_posetrack18_clean.pth
