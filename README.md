# UPAR 2027 Challenge — Track 3 (Human Pose Estimation)

Code, training scripts and parameters for our final Track 3 submission:
**a recipe-grid weight soup of 22 ViTPose-L fine-tunes, fused at the heatmap level with an HRNet-W48 (COCO → PoseTrack18) partner, with a power-mean confidence.**

| Submission | AP | AP50 | AP75 | AP_M | AP_L | AR |
|---|---|---|---|---|---|---|
| Soup-22 + HRNet-W48 + power mean | 0.9068 | 0.9708 | 0.9396 | 0.8787 | 0.6971 | 0.9252 |

Only permitted pretraining sources are used (ImageNet-1K, COCO, PoseTrack18). Fine-tuning uses only the provided Market-1501, PA100k and PETA annotations. The hidden test images are used for inference only.

---

## 1. Quick start: reproduce the leaderboard submission (inference only)

The submitted package is self-contained: TorchScript weights of the soup and the HRNet partner, plus `run.py` (the server API `predict_image(sample)`).

1. Download `submission_track3_ens_soup22_hrpt18_sp4.zip` (1.38 GB, MD5 `54f91d079b3125ac348f4badd4bf8f7c`): https://drive.google.com/file/d/1UEkah4AcEfZLWVmEnK5XxV3qI4KCPSLi/view?usp=sharing
2. Unzip it and run it locally the same way the evaluation server does:

```bash
unzip submission_track3_ens_soup22_hrpt18_sp4.zip -d final_submission
python tools/predict_local.py final_submission \
    <ANNOTATIONS.json with images + person boxes> <IMAGE_ROOT> --out predictions.json
```

`tools/predict_local.py` imports `run.py`, calls `predict_image` once per image and writes COCO keypoint results. If the annotation file contains ground-truth keypoints, it also prints the OKS AP. The OKS sigmas are those of PoseTrack18, which equal the COCO sigmas.

Sanity check on 50 provided PETA training images (these are training images, so the AP is optimistic):
```bash
python tools/predict_local.py final_submission \
    datasets/UPAR-Challenge-2027-Track3/data/annotations/specialization/train_peta.json \
    datasets/UPAR-Challenge-2027-Track3/data --limit 50
```

The same zip is exactly the file submitted to Codabench, so it can also be uploaded there unchanged (`run.py` is at the archive root).

Run it in the official runtime image, without building anything (pycocotools is only needed for the optional AP printout):
```bash
docker run --gpus all --rm -v $PWD:/work -w /work pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  bash -c "pip install -q pycocotools && python tools/predict_local.py final_submission \
           datasets/UPAR-Challenge-2027-Track3/data/annotations/specialization/train_peta.json \
           datasets/UPAR-Challenge-2027-Track3/data --limit 50"
```

Or call the server API directly from Python:
```python
import sys; sys.path.insert(0, "final_submission")
import os; os.chdir("final_submission")      # run.py loads its weights from ./assets
import run
out = run.predict_image({"image_id": 1, "file_name": "img.jpg", "image_path": "/abs/path/img.jpg",
                         "width": 64, "height": 128, "boxes": [[x, y, w, h]], "box_ids": [1]})
# out = {"keypoints": [[[x, y, v] * 17] per box], "scores": [confidence per box]}, absolute image coordinates
```

The package needs only torch, numpy and Pillow, and makes no network access. It uses a GPU if one is available and falls back to CPU otherwise (slow). On the server it took 5,405 s for the 30,245 test crops, including flip testing of both networks.

## 2. Environment

```bash
docker build -t upar-track3 .
docker run --gpus all -it --rm \
  -v /path/to/UPAR-Challenge-2027-Track3/data:/workspace/upar2027_track3/datasets/UPAR-Challenge-2027-Track3/data \
  -v /path/to/pretrained:/workspace/upar2027_track3/pretrained \
  upar-track3 bash
```

Alternatively, use `pip install -r requirements.txt` with Python 3.11. We trained with torch 2.4.1 (CUDA 12.1) and transformers 4.57.6 on single NVIDIA RTX 3080 (20 GB) GPUs.

Data layout: the official challenge repository's `data/` folder (images and `annotations/specialization/train_{market,pa100k,peta}.json`), placed at `datasets/UPAR-Challenge-2027-Track3/data`. Use the official `download_datasets.py` to create it.

## 3. Full training pipeline

All steps are run from the repository root.

```bash
bash scripts/01_prepare_pretrained.sh          # permitted pretrained weights -> ./pretrained
bash scripts/02_train_vit_members.sh 0         # 22 ViTPose-L members (GPU 0), ~30 min/epoch each
bash scripts/03_train_hrnet.sh 0               # HRNet-W48 partner, 10 epochs
bash scripts/04_build_submission.sh 0          # soup + BN re-estimation + export + final package
```

The output is `submission_final.zip`, which has the same structure as the leaderboard package. Members are independent, so step 2 can be split across several GPUs by editing the `MEMBERS` list.

### 3.1 Pretrained weights (permitted sources only)

| Model | Pretraining | Source |
|---|---|---|
| ViTPose-L (original release, classic decoder) | MAE on ImageNet-1K → COCO keypoints | `vitpose-l.pth` from https://github.com/ViTAE-Transformer/ViTPose. Converted to the HuggingFace `VitPose` format by `tools/convert_vitpose_l.py` (the script fetches only the architecture config and preprocessor files from the Hub, no weights). The multi-dataset ViTPose+ models are **not** used. |
| HRNet-W48 | ImageNet-1K → COCO → PoseTrack18 | MMPose `hrnet_w48_posetrack18_384x288-5fd6d3ff_20211130.pth`. Metadata is stripped by `tools/clean_mmpose_ckpt.py`. |

### 3.2 Base fine-tuning recipe (`configs/final_vitl384_e4.yaml`)

- Input 384×288. The box is expanded to aspect ratio 3:4 with padding 1.25 (train) / 1.1 (test). Gaussian heatmaps with σ = 2 at stride 4 (96×72), 17 PoseTrack18 slots. The COCO output channels are copied to the target slots, with head_bottom/head_top initialized from the eye channels.
- Loss: MSE on heatmaps, masked by visibility. Ear slots are never labeled.
- AdamW: lr 3.5e-5 (backbone) / 3.5e-4 (decoder), weight decay 1e-4, batch 16, AMP, gradient checkpointing, constant learning rate.
- EMA of weights (decay 0.9998). The **EMA of the last epoch** is used; no checkpoint selection on validation data.
- Augmentation:
  - horizontal flip with left/right swap;
  - ±25 % scale jitter;
  - with p = 0.3, a tighter and shifted crop (scale 0.5–0.85, center jitter 0.15).
- Trained on the union of the three training sets (`--fold final`).

### 3.3 The 22 soup members

Each member changes one factor of the base recipe and has its own seed. Config: `configs/final_vitl384_<name>.yaml`.

| Recipe | Change | Members (epochs) |
|---|---|---|
| Base | — | e4 (4), e4s2 (4), e10 (10), d1 (8) |
| No crop-shrink | shifted-crop augmentation off | nocs (4), d2 (6) |
| Head weighting | loss ×2 on nose, head_bottom, head_top | hw (4), d4 (5) |
| Quality | resize/blur/JPEG (p 0.3) + color jitter 0.25 | qual (4), d3 (10) |
| Drop-path | stochastic depth 0.2 | dp (4) |
| Illumination | gain [0.3, 1.5], gamma [0.6, 1.8], color temperature, grayscale, noise (p 0.4) | v2a (4), v2a10 (10) |
| Motion / low resolution | motion blur 5–21 px (p 0.3), downscale to 12–35 % (p 0.3) | v2b (4) |
| Occlusion | boxes near joints (p 0.5), pasted persons (p 0.3) | v2c (4), v2c10 (10) |
| Combined | all of the above at lower rates | v2d (4), v2d6 (6), v2d10 (10) |
| Lower-body occlusion | band from the hips down (p 0.4), pasted persons (p 0.5) | v2e (4) |
| All-in-one | combined + head weighting + drop-path | v2all (6), v2all2 (6) |

The augmentations are implemented in `data/transforms.py` (`degrade_crop`, `lower_occlude`) and `data/pose_dataset.py`.

### 3.4 Soup, partner and inference

1. **Soup:** uniform average of the 22 members' weights (`tools/build_soup.py`). BatchNorm statistics (decoder only; the ViT uses LayerNorm) are then re-estimated on 4,000 training crops (`recalibrate_bn.py`).
2. **Partner:** HRNet-W48 PoseTrack18, fine-tuned 10 epochs (lr 1e-4, batch 8, base augmentation, `configs/final_hrnetpt384.yaml`), with BN re-estimated in the same way.
3. **Inference** (`submission_template/run_ensemble.py` + `run.py`):
   - Both networks see the crop and its mirror. Mirrored heatmaps are flipped back, left/right channels are swapped, and the result is shifted by 1 px and averaged.
   - The flip-averaged heatmaps are fused: `H = 0.8 · H_soup + 0.2 · H_hrnet`.
   - One DARK decoding, then the inverse affine transform maps coordinates back to the image.
   - Confidence: `c = (mean_j s_j^4)^(1/4)` over the fused peaks of the 13 labeled joints (nose and the 12 body joints, excluding head_top and head_bottom).

## 4. Repository layout

```
train.py               training loop (EMA, AMP, per-epoch OKS-AP when a validation fold is used)
run_loo.py             leave-one-dataset-out / held-out experiments (not needed for the submission)
recalibrate_bn.py      BatchNorm re-estimation after souping
export_submission.py   export one network (TorchScript + preprocessing json) as a submission package
build_ensemble.py      combine packages: heatmap fusion with weights, power-mean score
eval_ckpt.py           evaluate a checkpoint on held-out data
configs/               base.yaml, 22 member configs, HRNet partner config
data/ models/ eval/    dataset + augmentations, ViTPose-L / HRNet builders, COCO OKS evaluation
submission_template/   run.py (single model), run_ensemble.py (final entry point), metadata.yaml, assets/mean_pose.json
tools/                 convert_vitpose_l.py, clean_mmpose_ckpt.py, build_soup.py, predict_local.py
scripts/               01–04 end-to-end pipeline
```

Code comments are partly in Vietnamese.

## 5. Notes

- Training is deterministic for a fixed seed on our hardware. Retraining the HRNet partner reproduced identical weights. Small numerical differences across GPUs or library versions are possible.
- Some optional code paths (SimCC/FPN heads, presence head, OKS targets, OHKM, UDP, LOO utilities) are disabled in all final configs.
- Contact: uitsecretorganization@gmail.com
