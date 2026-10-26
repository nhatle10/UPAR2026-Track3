"""Tính COCO OKS AP thật — dùng `pycocotools` (KHÔNG tự viết lại thuật toán
COCOeval bằng tay). Khác BTC (Docker chấm của họ thiếu mạng nên phải có bundled
numpy fallback), môi trường train/LOO của ta tự kiểm soát được, `pip install
pycocotools` bình thường — không có lý do chấp nhận rủi ro đúng-sai của một
bản tự viết lại thuật toán matching/AP phức tạp (greedy OKS matching theo
score, tích phân precision-recall 101 điểm, maxDets...). Xem docs_3/strategy.md
mục 3.

Sigma 17 keypoint ĐÃ VERIFY khớp CHÍNH XÁC `pycocotools` default
`kpt_oks_sigmas` (PoseTrack18 dùng lại đúng hằng số COCO, mượn sigma left_eye/
right_eye cho head_bottom/head_top) — vẫn set tường minh ở đây thay vì dựa vào
default ngầm định, để code tự giải thích và không phụ thuộc hành vi nội bộ của
thư viện có thể đổi giữa các phiên bản.
"""
from __future__ import annotations

import numpy as np
import torch

from data.pose_dataset import FLIP_PAIRS, PoseSample
from data.transforms import (
    KPT_OKS_SIGMAS, box_to_center_scale, decode_heatmaps, decode_simcc, flip_heatmaps, warp_affine, warp_matrix,
)
from PIL import Image

NUM_KEYPOINTS = 17
KEYPOINT_NAMES = [
    "nose", "head_bottom", "head_top", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]  # 1-indexed, chỉ để hợp lệ hoá COCO category, không ảnh hưởng OKS
# KPT_OKS_SIGMAS giờ định nghĩa ở data/transforms.py (tránh trùng lặp — hàm
# generate_oks_target_heatmap mới cũng cần dùng, xem docs_3/strategy.md mục 7.7).


def build_coco_gt(samples: list[PoseSample], area_scale: float = 1.0) -> dict:
    """image_id = index trong `samples` + 1 — PHẢI dùng CÙNG list (cùng thứ
    tự) khi gọi predict_and_decode() để khớp image_id giữa GT và detections.

    QUAN TRỌNG đã debug thật (01/10/2026): id BẮT ĐẦU TỪ 1, không phải 0 —
    pycocotools dùng 0 làm sentinel nội bộ "chưa khớp" trong dtMatches/
    gtMatches (`accumulate()` so sánh == 0 để tính false positive). Nếu GT
    thật có id=0, một detection khớp ĐÚNG với gt id=0 sẽ bị đọc nhầm thành
    "chưa khớp" vì dtMatches=0 trông giống hệt sentinel "no match" — khiến AP
    sai hoàn toàn (đã tái hiện: AP=0 dù detection trùng khít GT tuyệt đối, chỉ
    vì annotation id=0). Lỗi kinh điển của pycocotools, không phải lỗi logic
    OKS — đã verify bằng cách đổi offset +1 thì AP quay về đúng ~1.0."""
    images, annotations = [], []
    for i, s in enumerate(samples):
        img_id = i + 1
        images.append({"id": img_id, "width": s.width, "height": s.height, "file_name": s.image_path})
        kpts = s.keypoints
        num_kp = int((kpts[:, 2] > 0).sum())
        annotations.append({
            "id": img_id, "image_id": img_id, "category_id": 1,
            "keypoints": kpts.reshape(-1).tolist(),
            "num_keypoints": num_kp,
            "bbox": list(s.bbox), "area": float(max(s.bbox[2], 1.0) * max(s.bbox[3], 1.0)) * area_scale,
            "iscrowd": 0,
        })
    categories = [{"id": 1, "name": "person", "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON}]
    return {"images": images, "annotations": annotations, "categories": categories}


@torch.no_grad()
def predict_and_decode(model, samples: list[PoseSample], preprocess_cfg: dict, device: str,
                        batch_size: int = 32, tta_flip: bool = False, num_workers: int = 16,
                        decode_fix: bool = False) -> list[dict]:
    """Forward + DARK decode cho TOÀN BỘ samples, trả về list COCO-detection
    dict (image_id khớp index trong `samples`, giống build_coco_gt). Dùng
    center-scale CHUẨN (không augment) — giống hệt đường inference thật của
    submission_template/run.py.

    `tta_flip` (TUỲ CHỌN, mặc định False — docs_3/strategy.md mục 7.4):
    forward thêm ảnh lật ngang, trung bình heatmap (đã căn chỉnh UDP-style
    qua `flip_heatmaps`) trước khi decode — tốn gấp đôi compute, thường
    +0.5-1 AP. So LOO có/không `tta_flip` trước khi quyết định bật lúc nộp.
    CHƯA hỗ trợ cho `model.head_type == "simcc"` (cơ chế lật khác hẳn, cần
    lật thứ tự bin — xem docs_3/strategy.md mục 7.1), ném lỗi rõ ràng thay vì
    âm thầm cho kết quả sai nếu bật nhầm."""
    if tta_flip and model.head_type == "simcc":
        raise NotImplementedError(
            "tta_flip chưa hỗ trợ cho head_type='simcc' (cơ chế lật khác heatmap, "
            "cần lật thứ tự bin x — chưa implement). Tắt tta_flip hoặc dùng head khác."
        )
    model.eval()
    in_h, in_w = preprocess_cfg["input_height"], preprocess_cfg["input_width"]
    mean = np.asarray(preprocess_cfg["image_mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(preprocess_cfg["image_std"], dtype=np.float32).reshape(3, 1, 1)
    rescale = preprocess_cfg["rescale_factor"]
    norm_factor = preprocess_cfg["normalize_factor"]
    pad_factor = preprocess_cfg["padding_factor"]

    class _EvalSet(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            s = samples[i]
            image = np.asarray(Image.open(s.image_path).convert("RGB"), dtype=np.float32)
            center, scale = box_to_center_scale(s.bbox, in_w, in_h, norm_factor, pad_factor)
            matrix = warp_matrix(center, scale, in_w, in_h)
            crop = warp_affine(image, matrix, in_h, in_w)
            tensor = ((crop.transpose(2, 0, 1) * rescale - mean) / std).astype(np.float32)
            from data.pose_dataset import scale_feature
            return tensor, center.astype(np.float64), np.asarray(scale, dtype=np.float64), np.float32(scale_feature(s.bbox))

    # Tiền xử lý ảnh song song bằng process (warp numpy thuần là nút thắt eval khi chạy tuần tự)
    loader = torch.utils.data.DataLoader(_EvalSet(), batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)

    detections = []
    start = 0
    for tensors_b, centers_b, scales_b, sf_b in loader:
        chunk = range(len(tensors_b))
        centers = list(centers_b.numpy())
        scales = list(scales_b.numpy())
        tensors = tensors_b.numpy()

        batch = torch.from_numpy(tensors).to(device)
        sf_dev = sf_b.to(device)

        def _forward(b):
            out = model(b, sf_dev) if getattr(model, "scale_cond", False) else model(b)
            # use_presence=True: forward() trả (main_output, presence_logits) —
            # presence CHƯA tích hợp vào decode/score (xem models/build.py::
            # PresenceHead docstring), chỉ bỏ qua ở đây.
            return out[0] if getattr(model, "use_presence", False) else out

        if model.head_type == "simcc":
            x_logits, y_logits = _forward(batch)
            coords, kp_scores = decode_simcc(x_logits.float().cpu().numpy(), y_logits.float().cpu().numpy(),
                                              np.stack(centers), np.stack(scales),
                                              split_ratio=model.simcc_split_ratio)
        else:
            heatmaps = _forward(batch).float().cpu().numpy()
            if tta_flip:
                batch_flipped = torch.flip(batch, dims=[-1])  # lật ngang (trục W)
                heatmaps_flipped = _forward(batch_flipped).float().cpu().numpy()
                heatmaps_flipped = flip_heatmaps(heatmaps_flipped, FLIP_PAIRS, shift=1 if decode_fix else 0)
                heatmaps = (heatmaps + heatmaps_flipped) * 0.5
            coords, kp_scores = decode_heatmaps(heatmaps, np.stack(centers), np.stack(scales),
                                                 input_wh=(in_w, in_h) if decode_fix else None)

        for j, s_idx in enumerate(range(start, start + len(chunk))):
            flat = []
            for k in range(NUM_KEYPOINTS):
                flat.extend([float(coords[j, k, 0]), float(coords[j, k, 1]), 1.0])
            detections.append({
                "image_id": s_idx + 1, "category_id": 1,  # +1: khớp offset id trong build_coco_gt (xem docstring ở đó)
                "keypoints": flat, "score": float(np.mean(kp_scores[j])),
            })
        start += len(chunk)

    return detections


def evaluate_oks_ap(samples: list[PoseSample], detections: list[dict], area_scale: float = 1.0) -> dict:
    """AP/AP50/AP_medium/AP_large qua pycocotools — ĐÚNG protocol
    `evaluation.md` (maxDets=20). Trả {} với giá trị NaN nếu samples rỗng."""
    if not samples:
        return {"AP": float("nan"), "AP50": float("nan"), "AP_medium": float("nan"), "AP_large": float("nan")}
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise ImportError(
            "pycocotools chưa cài — `pip install pycocotools` (xem code_3/requirements.txt). "
            "Không có fallback tự viết trong code_3 (xem docstring module này)."
        ) from e

    gt = build_coco_gt(samples, area_scale)
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])

    ev = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    ev.params.kpt_oks_sigmas = KPT_OKS_SIGMAS
    ev.params.maxDets = [20]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats
    return {
        "AP": float(stats[0]), "AP50": float(stats[1]),
        "AP_medium": float(stats[3]), "AP_large": float(stats[4]),
    }
