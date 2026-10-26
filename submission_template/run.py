"""[TRACK 3] Entry point nộp lên Codabench.

COPY GẦN NGUYÊN VĂN từ baseline mẫu CHÍNH THỨC của BTC
(UPAR-Challenge-2027-Track3/examples/task3/vitpose_submission/run.py, đã đọc
trực tiếp 01/10/2026) — preprocessing (center-scale crop, affine warp) và
decode heatmap (DARK, Zhang et al. CVPR 2020) giữ NGUYÊN, đã verify đúng qua
test tay (xem code_3/README.md). KHÁC bản gốc: `assets/vitpose.torchscript`
ở đây là checkpoint ĐÃ FINE-TUNE trên dữ liệu UPAR-Pose (qua
code_3/export_submission.py), không phải checkpoint off-the-shelf của BTC.

API đúng theo `docs_3/submission.md`:
    load_model() -> None
    predict_image(sample: dict) -> dict  # {"keypoints": [...], "scores": [...]}

QUAN TRỌNG: file này KHÔNG được import từ code_3/data, code_3/models... — phải
tự đứng độc lập trong submission.zip (không có mạng, không có code_3/ khác
ngoài assets/). Mọi hàm preprocessing/decode viết lại trực tiếp ở đây, trùng
với code_3/data/transforms.py (dùng cho train/eval nội bộ) NHƯNG 2 bản này
KHÔNG tự động đồng bộ — sửa 1 bên phải kiểm tra bên kia (giống quy ước
code/code_2 cho các file COPY).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "assets", "vitpose.torchscript")
META_PATH = os.path.join(HERE, "assets", "preprocess.json")
PRIOR_PATH = os.path.join(HERE, "assets", "mean_pose.json")

BATCH_SIZE = 8
DARK_KERNEL = 11
# Cặp keypoint trái/phải để hoán vị lúc flip-test TTA — PHẢI khớp
# data/pose_dataset.py::FLIP_PAIRS (code_3/), trùng lặp có chủ đích vì file
# này đứng độc lập trong submission.zip, không import được data/.
FLIP_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

_MODEL = None
_DEVICE = None
_META: dict[str, Any] = {}
_PRIOR: list[list[float]] | None = None
_MEAN: np.ndarray | None = None
_STD: np.ndarray | None = None
_TTA_FLIP = False
_FLIP_SHIFT = 0
_DECODE_WH = None  # (crop_w, crop_h) khi preprocess.json có decode_fix=true
_HEAD_TYPE = "default"
_SIMCC_SPLIT_RATIO = 2
_SCORE_JOINTS = None  # chỉ số khớp dùng để tính điểm tin cậy; None = tất cả 17 khớp (hành vi cũ)


def load_model() -> None:
    global _MODEL, _DEVICE, _META, _PRIOR, _MEAN, _STD, _TTA_FLIP, _HEAD_TYPE, _SIMCC_SPLIT_RATIO, _FLIP_SHIFT, _DECODE_WH, _SCORE_JOINTS
    if _MODEL is not None:
        return

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run `python export_model.py` before zipping: "
            "the evaluation container cannot download anything."
        )

    with open(META_PATH, "r", encoding="utf-8") as fh:
        _META = json.load(fh)
    with open(PRIOR_PATH, "r", encoding="utf-8") as fh:
        _PRIOR = json.load(fh)["mean_pose_normalized"]

    _MEAN = np.asarray(_META["image_mean"], dtype=np.float32).reshape(3, 1, 1)
    _STD = np.asarray(_META["image_std"], dtype=np.float32).reshape(3, 1, 1)
    _TTA_FLIP = bool(_META.get("tta_flip", False))  # TUỲ CHỌN, mặc định tắt — đã bake vào preprocess.json lúc export
    _HEAD_TYPE = str(_META.get("head_type", "default"))
    _FLIP_SHIFT = int(_META.get("flip_shift", 0))  # export mới đặt 1; zip cũ không có khoá này => 0 (hành vi cũ)
    _DECODE_WH = (int(_META["input_width"]), int(_META["input_height"])) if _META.get("decode_fix", False) else None
    _SIMCC_SPLIT_RATIO = int(_META.get("simcc_split_ratio", 2))
    _sj = _META.get("score_joints")   # TUỲ CHỌN: bỏ tai (không có nhãn) và đầu khỏi điểm tin cậy; không có khoá => như cũ
    _SCORE_JOINTS = [int(k) for k in _sj] if _sj else None

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    _MODEL = torch.jit.load(MODEL_PATH, map_location=_DEVICE)
    _MODEL.eval()
    print(f"[submission] ViTPose ready on {_DEVICE} "
          f"({_META['input_width']}x{_META['input_height']} input, tta_flip={_TTA_FLIP}, "
          f"head_type={_HEAD_TYPE})", flush=True)


def flip_heatmaps(heatmaps: np.ndarray, shift: int = 0) -> np.ndarray:
    """Lật heatmap + hoán vị kênh trái/phải cho flip-test TTA — KHÔNG dịch
    pixel (xem code_3/data/transforms.py::flip_heatmaps cho lý do + số đo; 2
    bản TRÙNG LẶP CÓ CHỦ Ý vì file này đứng độc lập trong submission.zip)."""
    flipped = heatmaps[..., ::-1].copy()
    if shift:
        flipped[..., shift:] = flipped[..., :-shift].copy()
        flipped[..., :shift] = 0.0
    for a, b in FLIP_PAIRS:
        flipped[:, [a, b]] = flipped[:, [b, a]]
    return flipped


def box_to_center_scale(box, out_w: int, out_h: int,
                        normalize_factor: float, padding_factor: float):
    x, y, w, h = [float(v) for v in box[:4]]
    aspect = out_w / out_h
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w / normalize_factor, h / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor


def warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    size_input = center * 2.0
    size_dst = np.array([out_w, out_h], dtype=np.float32) - 1.0
    size_target = scale * 200.0
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] + 0.5 * size_target[0])
    matrix[1, 1] = scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[1] + 0.5 * size_target[1])
    return matrix


def warp_affine(image: np.ndarray, matrix: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    inverse = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32),
                         np.arange(out_w, dtype=np.float32), indexing="ij")
    src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]

    height, width = image.shape[:2]
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = src_x - x0, src_y - y0

    inside = (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)

    top = image[y0c, x0c] * (1 - wx)[..., None] + image[y0c, x1c] * wx[..., None]
    bottom = image[y1c, x0c] * (1 - wx)[..., None] + image[y1c, x1c] * wx[..., None]
    return (top * (1 - wy)[..., None] + bottom * wy[..., None]) * inside[..., None]


def preprocess(image: np.ndarray, box) -> tuple:
    out_w, out_h = int(_META["input_width"]), int(_META["input_height"])
    center, scale = box_to_center_scale(
        box, out_w, out_h, float(_META["normalize_factor"]), float(_META["padding_factor"]))
    crop = warp_affine(image, warp_matrix(center, scale, out_w, out_h), out_h, out_w)
    tensor = crop.transpose(2, 0, 1) * float(_META["rescale_factor"])
    return (((tensor - _MEAN) / _STD).astype(np.float32), center, scale)


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 / (sigma * sigma) * x * x)
    return (kernel / kernel.sum()).astype(np.float32)


def _gaussian_blur(heatmaps: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = (kernel_size - 1) // 2
    kernel = _gaussian_kernel1d(0.8, radius)
    padded = np.pad(heatmaps, ((0, 0), (0, 0), (radius, radius), (radius, radius)),
                    mode="symmetric")
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 2, padded)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 3, blurred)


def decode_heatmaps(heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray, input_wh=None):
    batch, num_kpts, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, num_kpts, -1)
    idx = np.argmax(flat, axis=2)
    scores = np.max(flat, axis=2)

    coords = np.zeros((batch, num_kpts, 2), dtype=np.float32)
    coords[:, :, 0] = idx % width
    coords[:, :, 1] = idx // width
    coords = np.where((scores > 0.0)[..., None], coords, -1.0)

    blurred = np.clip(_gaussian_blur(heatmaps, DARK_KERNEL), 0.001, 50.0)
    log_hm = np.log(blurred)
    padded = np.pad(log_hm, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").reshape(-1)

    stride = (width + 2) * (height + 2)
    base = (coords[..., 0] + 1 + (coords[..., 1] + 1) * (width + 2))
    base = base + stride * np.arange(batch * num_kpts).reshape(batch, num_kpts)
    index = base.astype(np.int64).reshape(-1, 1)

    center_v = padded[index]
    x_plus = padded[index + 1]
    x_minus = padded[index - 1]
    y_plus = padded[index + width + 2]
    y_minus = padded[index - 2 - width]
    xy_plus = padded[index + width + 3]
    xy_minus = padded[index - width - 3]

    dx = 0.5 * (x_plus - x_minus)
    dy = 0.5 * (y_plus - y_minus)
    derivative = np.concatenate([dx, dy], axis=1).reshape(batch, num_kpts, 2, 1)

    dxx = x_plus - 2 * center_v + x_minus
    dyy = y_plus - 2 * center_v + y_minus
    dxy = 0.5 * (xy_plus - x_plus - y_plus + center_v + center_v - x_minus - y_minus + xy_minus)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(batch, num_kpts, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords = coords - np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze(-1)

    # input_wh=(crop_w, crop_h): ánh xạ chỉ số heatmap -> ảnh khớp với target huấn luyện (xem data/transforms.py::decode_heatmaps). None = cũ.
    if input_wh is None:
        kx, ky = 1.0 / (width - 1.0), 1.0 / (height - 1.0)
    else:
        kx = input_wh[0] / (width * (input_wh[0] - 1.0))
        ky = input_wh[1] / (height * (input_wh[1] - 1.0))
    out = np.zeros_like(coords)
    for i in range(batch):
        scale = scales[i] * 200.0
        out[i, :, 0] = coords[i, :, 0] * scale[0] * kx + centers[i][0] - scale[0] * 0.5
        out[i, :, 1] = coords[i, :, 1] * scale[1] * ky + centers[i][1] - scale[1] * 0.5
    return out, scores


def decode_simcc(x_logits: np.ndarray, y_logits: np.ndarray, centers: np.ndarray, scales: np.ndarray,
                  split_ratio: int):
    """Giải mã output SimCCHead — TRÙNG LẶP CÓ CHỦ Ý với
    data/transforms.py::decode_simcc (file này đứng độc lập trong
    submission.zip, không import được data/). Xem docstring bản gốc cho giải
    thích đầy đủ (kỳ vọng softmax mỗi trục, quy đổi bin->heatmap-grid->ảnh
    gốc dùng CHUNG công thức với decode_heatmaps)."""
    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    px = softmax(x_logits.astype(np.float64))
    py = softmax(y_logits.astype(np.float64))
    num_x_bins, num_y_bins = px.shape[-1], py.shape[-1]

    exp_x = (px * np.arange(num_x_bins, dtype=np.float64)[None, None, :]).sum(axis=-1)
    exp_y = (py * np.arange(num_y_bins, dtype=np.float64)[None, None, :]).sum(axis=-1)
    scores = (px.max(axis=-1) + py.max(axis=-1)) / 2.0

    heatmap_w = num_x_bins / split_ratio
    heatmap_h = num_y_bins / split_ratio
    coord_x = exp_x / split_ratio
    coord_y = exp_y / split_ratio

    batch = x_logits.shape[0]
    out = np.zeros((batch, x_logits.shape[1], 2), dtype=np.float64)
    for i in range(batch):
        scale = scales[i] * 200.0
        out[i, :, 0] = coord_x[i] * scale[0] / (heatmap_w - 1.0) + centers[i][0] - scale[0] * 0.5
        out[i, :, 1] = coord_y[i] * scale[1] / (heatmap_h - 1.0) + centers[i][1] - scale[1] * 0.5
    return out, scores


def _prior_pose(box) -> list[list[float]]:
    bx, by, bw, bh = [float(v) for v in box]
    return [[bx + u * bw, by + v * bh, 1.0] for u, v in _PRIOR]


def _resolve_image_path(sample: dict[str, Any]) -> str:
    image_path = str(sample.get("image_path", ""))
    if image_path and os.path.isfile(image_path):
        return image_path

    file_name = str(sample.get("file_name", ""))
    if image_path:
        candidates = [
            image_path.replace("/images/images/", "/images/"),
            image_path.replace("\\images\\images\\", "\\images\\"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    if file_name:
        candidates = [
            file_name,
            file_name.removeprefix("images/"),
            file_name.removeprefix("images\\"),
        ]
        for candidate in candidates:
            candidate_path = os.path.join(HERE, candidate)
            if os.path.isfile(candidate_path):
                return candidate_path

    if image_path:
        return image_path
    raise FileNotFoundError("Could not resolve image path for sample.")


def _load_image_matching_boxes(sample: dict[str, Any], boxes: list):
    """Đọc ảnh và BẢO ĐẢM hệ toạ độ khớp với `boxes` (khai báo qua sample["width"/"height"]). Phòng thủ cho subset test lạ:
    ảnh xoay theo EXIF, hoặc kích thước thật khác khai báo. Trả (ảnh float32 HWC, boxes đã quy đổi, (sx, sy) để đổi toạ độ dự đoán về hệ ban đầu).
    Với dữ liệu train (kích thước khớp 100%, không EXIF) đường này không đổi gì."""
    from PIL import ImageOps
    img = Image.open(_resolve_image_path(sample))
    dw, dh = sample.get("width"), sample.get("height")
    if dw and dh and (img.width, img.height) != (int(dw), int(dh)):
        try:
            t = ImageOps.exif_transpose(img)
            if (t.width, t.height) == (int(dw), int(dh)):
                img = t
        except Exception:
            pass
    img = img.convert("RGB")
    sx = sy = 1.0
    if dw and dh and (img.width, img.height) != (int(dw), int(dh)):
        sx, sy = img.width / float(dw), img.height / float(dh)   # ảnh đã đổi kích thước: quy đổi box sang hệ ảnh thật
        boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
    return np.asarray(img, dtype=np.float32), boxes, (sx, sy)


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    if _MODEL is None:
        load_model()

    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}

    orig_boxes = boxes
    image, boxes, inv_scale = _load_image_matching_boxes(sample, boxes)

    keypoints: list[Any] = [None] * len(boxes)
    scores: list[float] = [0.0] * len(boxes)

    usable = [i for i, b in enumerate(boxes) if float(b[2]) > 0 and float(b[3]) > 0]
    for i in set(range(len(boxes))) - set(usable):
        keypoints[i] = _prior_pose(orig_boxes[i])   # box suy biến: dùng box GỐC (không quy đổi)
        scores[i] = 0.01

    for start in range(0, len(usable), BATCH_SIZE):
        chunk = usable[start:start + BATCH_SIZE]
        tensors, centers, scales = [], [], []
        for i in chunk:
            tensor, center, scale = preprocess(image, boxes[i])
            tensors.append(tensor)
            centers.append(center)
            scales.append(scale)

        batch = torch.from_numpy(np.stack(tensors)).to(_DEVICE)
        if _HEAD_TYPE == "simcc":
            x_logits, y_logits = _MODEL(batch)
            coords, joint_scores = decode_simcc(x_logits.float().cpu().numpy(), y_logits.float().cpu().numpy(),
                                                 np.stack(centers), np.stack(scales), _SIMCC_SPLIT_RATIO)
        else:
            heatmaps = _MODEL(batch).float().cpu().numpy()
            if _TTA_FLIP:
                batch_flipped = torch.flip(batch, dims=[-1])
                heatmaps_flipped = _MODEL(batch_flipped).float().cpu().numpy()
                heatmaps = (heatmaps + flip_heatmaps(heatmaps_flipped, _FLIP_SHIFT)) * 0.5
            coords, joint_scores = decode_heatmaps(heatmaps, np.stack(centers), np.stack(scales), _DECODE_WH)

        for j, i in enumerate(chunk):
            keypoints[i] = [[float(x) / inv_scale[0], float(y) / inv_scale[1], 1.0] for x, y in coords[j]]
            scores[i] = float(np.mean(joint_scores[j] if _SCORE_JOINTS is None else joint_scores[j][_SCORE_JOINTS]))

    return {"keypoints": keypoints, "scores": scores}
