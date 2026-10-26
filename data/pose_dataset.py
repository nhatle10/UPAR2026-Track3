"""Đọc annotation COCO-format thật của Track 3 (train_<dataset>.json từ
UPAR-Challenge-2027-Track3/data/annotations/specialization/, đã verify cấu
trúc thật 01/10/2026 — xem docs_3/strategy.md mục 5) + Dataset top-down pose
(crop quanh box, warp keypoint, sinh heatmap target).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.splits import DATASET_TO_JSON
from data.transforms import (
    degrade_crop, occlude_crop, lower_occlude, paste_distractor, truncnorm_unit, half_body_box, edge_blackout,
    box_to_center_scale, generate_oks_target_heatmap, generate_target_heatmap,
    warp_affine, warp_matrix, warp_point,
)

NUM_KEYPOINTS = 17


def _is_dyadic(v: float, scale: int = 1024, tol: float = 2e-3) -> bool:
    """toạ độ là bội của 1/1024 (dấu hiệu nhãn do quy trình tự động sinh khi kém chắc chắn)"""
    r = abs(float(v)) * scale
    return abs(r - round(r)) < tol
DS_INDEX = {"Market1501": 0, "PA100k": 1, "PETA": 2}


def scale_feature(bbox) -> float:
    """log(sqrt(diện tích box gốc)) - 4.5 (~ 90px => 0). Dùng cho model scale_cond (FiLM)."""
    return float(np.log(np.sqrt(max(float(bbox[2]), 1.0) * max(float(bbox[3]), 1.0)))) - 4.5
# Cặp keypoint trái/phải để lật ngang lúc augment (flip) — nose(0)/head_bottom(1)/
# head_top(2) không có cặp (đối xứng trên trục dọc), ear/shoulder/elbow/wrist/
# hip/knee/ankle đều có cặp trái-phải, đúng thứ tự slot trong overview.md.
FLIP_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


@dataclass
class PoseSample:
    image_path: str
    width: int
    height: int
    bbox: list[float]  # [x, y, w, h], ảnh gốc
    keypoints: np.ndarray  # (17, 3) x, y, v — ảnh gốc
    source_dataset: str


def _resolve_image_path(root: str, file_name: str) -> str:
    """Giống _resolve_image_path trong run.py mẫu của BTC — PA100k có quirk
    đường dẫn lồng 'images/images/' tuỳ lúc tải, thử vài candidate."""
    candidates = [
        os.path.join(root, file_name),
        os.path.join(root, file_name.replace("/images/images/", "/images/")),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]  # để lỗi FileNotFoundError tự nhiên nếu không tìm thấy, dễ debug hơn im lặng


def load_pose_annotations(root: str, dataset: str) -> list[PoseSample]:
    """root: thư mục `data/` của UPAR-Challenge-2027-Track3 (chứa
    annotations/specialization/, market/, pa100k/, peta/)."""
    json_path = os.path.join(root, "annotations", "specialization", DATASET_TO_JSON[dataset])
    with open(json_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    samples = []
    for ann in coco["annotations"]:
        img = images_by_id.get(ann["image_id"])
        if img is None:
            continue
        kpts = np.asarray(ann["keypoints"], dtype=np.float64).reshape(NUM_KEYPOINTS, 3)
        samples.append(PoseSample(
            image_path=_resolve_image_path(root, img["file_name"]),
            width=img["width"], height=img["height"],
            bbox=list(ann["bbox"]), keypoints=kpts, source_dataset=dataset,
        ))
    return samples


def load_datasets(root: str, datasets: list[str]) -> list[PoseSample]:
    samples = []
    for d in datasets:
        samples.extend(load_pose_annotations(root, d))
    return samples


class PoseDataset(Dataset):
    """Top-down pose dataset: mỗi item = 1 box (không phải 1 ảnh — 1 ảnh có
    thể cho nhiều box, nhưng dữ liệu Track 3 chủ yếu 1 box/ảnh vì ảnh đã là
    crop pedestrian sẵn, xem docs_3/strategy.md mục 5)."""

    def __init__(self, samples: list[PoseSample], image_mean, image_std, rescale_factor: float,
                 input_h: int, input_w: int, heatmap_h: int, heatmap_w: int,
                 augment: bool = False, padding_factor: float = 1.25, normalize_factor: float = 200.0,
                 sigma: float = 2.0, flip_prob: float = 0.5, scale_jitter: float = 0.25,
                 rotation_prob: float = 0.5, rotation_degrees: float = 30.0,
                 crop_shrink_prob: float = 0.0, crop_shrink_range: tuple[float, float] = (0.5, 0.85),
                 center_jitter: float = 0.0, target_type: str = "gaussian",
                 quality_prob: float = 0.0, color_jitter: float = 0.0,
                 illum_prob: float = 0.0, motion_prob: float = 0.0, lowres_prob: float = 0.0,
                 occ_prob: float = 0.0, paste_prob: float = 0.0, lowocc_prob: float = 0.0, aspect_jitter: float = 0.0,
                 target_encoding: str = "legacy", sigma_occluded: float | None = None,
                 noisy_label_weight: float = 1.0,
                 stock_affine: bool = False, half_body_prob: float = 0.0, blackout_prob: float = 0.0,
                 blackout_zero_weight: bool = False, shear_prob: float = 0.0, shear_degrees: float = 15.0,
                 fda_prob: float = 0.0, fda_beta: float = 0.1, randconv_prob: float = 0.0):
        self.samples = samples
        self.mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        self.rescale_factor = rescale_factor
        self.input_h, self.input_w = input_h, input_w
        self.heatmap_h, self.heatmap_w = heatmap_h, heatmap_w
        self.augment = augment
        self.padding_factor = padding_factor
        self.normalize_factor = normalize_factor
        self.sigma = sigma
        self.flip_prob = flip_prob
        self.scale_jitter = scale_jitter
        self.rotation_prob = rotation_prob
        self.rotation_degrees = rotation_degrees
        # TUỲ CHỌN (docs_3/strategy.md mục 7.7, lấy cảm hứng ProbPose) — mặc
        # định TẮT (crop_shrink_prob=0.0, center_jitter=0.0 = hành vi cũ):
        # chủ động crop chặt hơn + lệch tâm, đẩy một số keypoint ra NGOÀI
        # vùng crop, dạy model tình huống "mất 1 phần cơ thể" (biên crop cắt
        # cụt chi) — khác scale_jitter (chỉ đổi tỷ lệ đều, hiếm khi đẩy hẳn
        # keypoint ra ngoài).
        self.crop_shrink_prob = crop_shrink_prob
        self.crop_shrink_range = crop_shrink_range
        self.center_jitter = center_jitter
        self.illum_prob, self.motion_prob, self.lowres_prob = illum_prob, motion_prob, lowres_prob
        self.quality_prob = quality_prob  # TUỲ CHỌN: augmentation chất lượng ảnh (data/transforms.py::degrade_crop)
        self.color_jitter = color_jitter
        self.stock_affine = stock_affine      # TUỲ CHỌN: công thức affine kiểu MMPose/ProbPose (xem README); thay scale_jitter/rotation/crop_shrink
        self.half_body_prob = half_body_prob
        self.blackout_prob = blackout_prob    # TUỲ CHỌN: che đen mép (ProbPose); giữ giám sát vị trí trừ khi blackout_zero_weight
        self.blackout_zero_weight = blackout_zero_weight
        self.noisy_label_weight = noisy_label_weight  # TUỲ CHỌN: trọng số loss cho toạ độ nhãn 'dyadic' (kém tin cậy, xem eda/)
        self.sigma_occluded = sigma_occluded  # TUỲ CHỌN: sigma target cho khớp v=1 (bị che)
        self.udp = (target_encoding == "udp")  # xem generate_target_heatmap(udp=True)
        self.fda_prob = fda_prob          # TUỲ CHỌN: Fourier Domain Augmentation (hoán biên độ tần số thấp với 1 ảnh train khác)
        self.fda_beta = fda_beta
        self.randconv_prob = randconv_prob  # TUỲ CHỌN: random convolution
        self.shear_prob = shear_prob          # TUỲ CHỌN: kéo nghiêng ngang (shear) crop quanh tâm, góc U(-shear_degrees,+shear_degrees)
        self.shear_degrees = shear_degrees
        self.aspect_jitter = aspect_jitter  # TUỲ CHỌN: kéo giãn crop theo 1 trục (box sai tỷ lệ), r=exp(U(-j,j)), scale*[r,1/r]
        self.occ_prob = occ_prob      # TUỲ CHỌN: che khuất tổng hợp (data/transforms.py::occlude_crop)
        self.lowocc_prob = lowocc_prob  # TUỲ CHỌN: che nửa dưới (data/transforms.py::lower_occlude)
        self.paste_prob = paste_prob  # TUỲ CHỌN: dán người gây nhiễu (data/transforms.py::paste_distractor)
        assert target_type in ("gaussian", "oks"), target_type
        self.target_type = target_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = np.asarray(Image.open(s.image_path).convert("RGB"), dtype=np.float32)
        keypoints = s.keypoints.copy()

        center, scale = box_to_center_scale(s.bbox, self.input_w, self.input_h,
                                             self.normalize_factor, self.padding_factor)

        flip = False
        rotation = 0.0
        blackout_rect = None
        if self.augment and self.stock_affine:
            bbox_used = list(s.bbox)
            hb = half_body_box(s.keypoints, s.bbox, prob=self.half_body_prob)
            if hb is not None:
                bbox_used = hb
                center, scale = box_to_center_scale(hb, self.input_w, self.input_h, self.normalize_factor, self.padding_factor)
            tn = truncnorm_unit(4)
            if np.random.random() < 0.3:                                  # dịch tâm: 0.16 * kích thước, xác suất 0.3
                center = center.copy()
                center[0] += tn[0] * 0.16 * scale[0] * 200.0
                center[1] += tn[1] * 0.16 * scale[1] * 200.0
            scale = scale * (1.0 + 0.5 * tn[2])                           # scale ~ 1 +- 0.5 (chuẩn cắt cụt), LUÔN áp dụng
            if np.random.random() < 0.6:
                rotation = float(tn[3] * 80.0)                            # xoay +-80 độ, xác suất 0.6
            if np.random.random() < self.flip_prob:
                flip = True
        elif self.augment:
            scale = scale * np.random.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)
            if self.aspect_jitter > 0:
                r = float(np.exp(np.random.uniform(-self.aspect_jitter, self.aspect_jitter)))
                scale = scale * np.array([r, 1.0 / r], dtype=scale.dtype)
            if np.random.random() < self.flip_prob:
                flip = True
            if np.random.random() < self.rotation_prob:
                rotation = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
            if np.random.random() < self.crop_shrink_prob:
                shrink = np.random.uniform(self.crop_shrink_range[0], self.crop_shrink_range[1])
                scale = scale * shrink
                if self.center_jitter > 0:
                    box_w, box_h = s.bbox[2], s.bbox[3]
                    center = center.copy()
                    center[0] += np.random.uniform(-self.center_jitter, self.center_jitter) * box_w
                    center[1] += np.random.uniform(-self.center_jitter, self.center_jitter) * box_h

        if flip:
            image = image[:, ::-1, :].copy()
            keypoints[:, 0] = s.width - 1 - keypoints[:, 0]
            for a, b in FLIP_PAIRS:
                keypoints[[a, b]] = keypoints[[b, a]]
            center = center.copy()
            center[0] = s.width - 1 - center[0]

        if self.augment and self.blackout_prob > 0:
            bb = list(s.bbox)
            if flip:
                bb[0] = s.width - 1 - (bb[0] + bb[2])
            image, blackout_rect = edge_blackout(image, bb, prob=self.blackout_prob)
        matrix = warp_matrix(center, scale, self.input_w, self.input_h, rotation=rotation)
        if self.augment and self.shear_prob > 0 and np.random.random() < self.shear_prob:
            k = float(np.tan(np.deg2rad(np.random.uniform(-self.shear_degrees, self.shear_degrees))))
            shear = np.array([[1.0, k, -k * self.input_h * 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            matrix = (shear @ np.vstack([matrix, np.array([[0.0, 0.0, 1.0]], dtype=np.float32)]))[:2].astype(np.float32)
        crop = warp_affine(image, matrix, self.input_h, self.input_w)
        if self.augment and (self.quality_prob > 0 or self.color_jitter > 0 or self.illum_prob > 0 or self.motion_prob > 0 or self.lowres_prob > 0):
            crop = degrade_crop(crop, self.quality_prob, self.color_jitter, self.illum_prob, self.motion_prob, self.lowres_prob)
        if self.augment and self.fda_prob > 0 and np.random.random() < self.fda_prob:
            from data.transforms import fourier_swap
            sj = self.samples[np.random.randint(len(self.samples))]
            oimg = np.asarray(Image.open(sj.image_path).convert("RGB"), dtype=np.float32)
            oc, osc = box_to_center_scale(sj.bbox, self.input_w, self.input_h, self.normalize_factor, self.padding_factor)
            crop = fourier_swap(crop, warp_affine(oimg, warp_matrix(oc, osc, self.input_w, self.input_h), self.input_h, self.input_w), self.fda_beta)
        if self.augment and self.randconv_prob > 0 and np.random.random() < self.randconv_prob:
            from data.transforms import random_conv
            crop = random_conv(crop)

        kpts_crop = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        for i in range(NUM_KEYPOINTS):
            kpts_crop[i] = warp_point(matrix, keypoints[i, 0], keypoints[i, 1])

        if self.augment and self.paste_prob > 0 and np.random.random() < self.paste_prob:
            j = np.random.randint(len(self.samples))
            sj = self.samples[j]
            oimg = np.asarray(Image.open(sj.image_path).convert("RGB"), dtype=np.float32)
            oc, osc = box_to_center_scale(sj.bbox, self.input_w, self.input_h, self.normalize_factor, self.padding_factor)
            other = warp_affine(oimg, warp_matrix(oc, osc, self.input_w, self.input_h), self.input_h, self.input_w)
            crop = paste_distractor(crop, other, 1.0)
        if self.augment and self.occ_prob > 0:
            crop = occlude_crop(crop, kpts_crop, keypoints[:, 2], self.occ_prob)
        if self.augment and self.lowocc_prob > 0:
            crop = lower_occlude(crop, kpts_crop, keypoints[:, 2], self.lowocc_prob)

        # scale_x/scale_y THUẦN (không lẫn rotation) — TÍNH LẠI theo đúng công
        # thức nội bộ của warp_matrix (dòng size_dst/size_target), vì matrix[0,0]/
        # matrix[1,1] sau khi ghép rotation không còn là scale thuần nữa. Dùng cho
        # generate_oks_target_heatmap (cần quy đổi khoảng cách heatmap-space về
        # đúng đơn vị ảnh gốc, không phụ thuộc góc xoay).
        size_dst = np.array([self.input_w, self.input_h], dtype=np.float32) - 1.0
        size_target = scale * 200.0
        warp_scale_xy = size_dst / size_target

        tensor = crop.transpose(2, 0, 1) * self.rescale_factor
        tensor = ((tensor - self.mean) / self.std).astype(np.float32)

        if self.target_type == "oks":
            bbox_area = float(s.bbox[2]) * float(s.bbox[3])
            target, target_weight = generate_oks_target_heatmap(
                kpts_crop, keypoints[:, 2], self.heatmap_h, self.heatmap_w,
                self.input_h, self.input_w, bbox_area=bbox_area, warp_scale=warp_scale_xy)
        else:
            target, target_weight = generate_target_heatmap(
                kpts_crop, keypoints[:, 2], self.heatmap_h, self.heatmap_w,
                self.input_h, self.input_w, sigma=self.sigma, udp=self.udp,
                sigma_occluded=self.sigma_occluded)

        if self.augment and self.blackout_zero_weight and blackout_rect is not None:
            x0, y0, x1, y1 = blackout_rect
            inside = (keypoints[:, 0] >= x0) & (keypoints[:, 0] < x1) & (keypoints[:, 1] >= y0) & (keypoints[:, 1] < y1)
            target_weight[~inside] = 0.0   # biến thể so sánh: tắt giám sát khớp nằm trong vùng bị che đen
        if self.augment and self.noisy_label_weight != 1.0:
            for kk in range(NUM_KEYPOINTS):
                if kk != 0 and target_weight[kk] > 0 and _is_dyadic(s.keypoints[kk, 0]) and _is_dyadic(s.keypoints[kk, 1]):
                    target_weight[kk] *= self.noisy_label_weight
        return {
            "image": tensor,
            "target": target,
            "target_weight": target_weight,
            # keypoints_crop: toạ độ RAW trong không gian crop (không raster
            # hoá) — Gaussian/OKS target không cần, nhưng SimCC loss cần toạ
            # độ liên tục trực tiếp (xem train.py::simcc_loss). Trả LUÔN
            # (không chỉ khi target_type=simcc) để code đơn giản, tốn thêm
            # không đáng kể (17×2 float).
            "keypoints_crop": kpts_crop,
            "ds_index": np.int64(DS_INDEX.get(s.source_dataset, 0)),   # chỉ số expert theo dataset nguồn (model dataset_experts)
            "scale_feat": np.float32(scale_feature(s.bbox)),   # log kích thước box GỐC (crop cố định kích thước làm mất thông tin này)
            "source_dataset": s.source_dataset,
        }
