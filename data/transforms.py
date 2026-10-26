"""Tiền xử lý ảnh/keypoint cho top-down pose (center-scale crop, affine warp,
sinh heatmap target, giải mã heatmap->toạ độ).

`box_to_center_scale`, `warp_matrix`, `warp_affine`, `_gaussian_kernel1d`,
`_gaussian_blur`, `decode_heatmaps` COPY GẦN NGUYÊN VĂN từ
`UPAR-Challenge-2027-Track3/examples/task3/vitpose_submission/run.py` (baseline
mẫu CHÍNH THỨC của BTC, đã đọc trực tiếp 01/10/2026) — đây là implementation
DARK (Zhang et al. CVPR 2020) đã test/dùng thật bởi BTC, KHÔNG viết lại logic
giải mã, chỉ thêm phần BTC không cần (vì họ chỉ inference): `warp_point` (áp
affine THUẬN cho keypoint lúc train, khác `warp_affine` áp NGHỊCH cho ảnh) và
`generate_target_heatmap` (sinh nhãn Gaussian để tính loss).
"""
from __future__ import annotations

import numpy as np

# Sigma OKS chuẩn 17 keypoint PoseTrack18 — ĐÃ VERIFY khớp CHÍNH XÁC
# pycocotools default `kpt_oks_sigmas` (xem docs_3/strategy.md mục 3). Định
# nghĩa Ở ĐÂY (không phải eval/coco_eval.py) để `generate_oks_target_heatmap`
# dùng được mà không tạo import vòng (coco_eval.py import ngược từ
# data.pose_dataset) — coco_eval.py import hằng số này TỪ ĐÂY, không định
# nghĩa bản riêng nữa.
KPT_OKS_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
], dtype=np.float64)

DARK_KERNEL = 11


def box_to_center_scale(box, out_w: int, out_h: int,
                         normalize_factor: float = 200.0, padding_factor: float = 1.25):
    """box: [x, y, w, h] (COCO xywh, toạ độ ảnh gốc). Trả về (center, scale)
    đã pad để khớp đúng tỷ lệ khung hình out_w/out_h của model (bbox Track 3
    thường LỆCH tỷ lệ model — xem docs_3/strategy.md mục 5)."""
    x, y, w, h = [float(v) for v in box[:4]]
    aspect = out_w / out_h
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w / normalize_factor, h / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor


def warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int,
                 rotation: float = 0.0) -> np.ndarray:
    """Ma trận affine 2x3 THUẬN: toạ độ ảnh gốc -> toạ độ crop (out_w, out_h).

    `rotation` (độ, MỚI — baseline mẫu BTC không có tham số này, chỉ
    scale+translate) xoay quanh `center` TRƯỚC khi áp scale+translate gốc —
    ghép bằng hợp 2 ma trận affine đồng nhất (homogeneous). Khi `rotation=0.0`
    (mặc định, ĐÚNG mọi lời gọi hiện có kể cả `submission_template/run.py`),
    ma trận xoay suy biến về ma trận đơn vị CHÍNH XÁC (cos=1, sin=0 → R=I) nên
    kết quả giống hệt bản gốc — đã verify bằng test tay, xem code_3/README.md."""
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

    if rotation != 0.0:
        theta = np.deg2rad(rotation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = float(center[0]), float(center[1])
        rot = np.array([
            [cos_t, -sin_t, cx - cos_t * cx + sin_t * cy],
            [sin_t, cos_t, cy - sin_t * cx - cos_t * cy],
        ], dtype=np.float32)
        m3 = np.vstack([matrix, [0.0, 0.0, 1.0]])
        r3 = np.vstack([rot, [0.0, 0.0, 1.0]])
        matrix = (m3 @ r3)[:2].astype(np.float32)
    return matrix


def warp_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Áp affine THUẬN (từ warp_matrix) cho MỘT điểm — dùng để warp keypoint
    ground-truth sang không gian crop lúc sinh heatmap target. KHÔNG có trong
    baseline mẫu (họ chỉ cần inference, không cần warp keypoint GT) — tự thêm,
    nhưng dùng ĐÚNG matrix thuận giống warp_matrix ở trên, nhất quán 100% với
    affine đã áp cho ảnh (qua nghịch đảo trong warp_affine bên dưới)."""
    return (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2],
            matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])


def warp_affine(image: np.ndarray, matrix: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resample ảnh vào không gian crop (out_h, out_w) bằng matrix THUẬN (áp
    nghịch đảo nội bộ để tra ngược từ dst về src, song song bilinear)."""
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


def generate_target_heatmap(keypoints_crop: np.ndarray, visibility: np.ndarray,
                             heatmap_h: int, heatmap_w: int, crop_h: int, crop_w: int,
                             sigma: float = 2.0, udp: bool = False,
                             sigma_occluded: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sinh Gaussian heatmap target + target_weight cho loss.

    keypoints_crop: (K, 2) toạ độ keypoint ĐÃ warp sang không gian crop (pixel,
        cùng hệ với ảnh input model, TRƯỚC khi chia cho scale_factor).
    visibility: (K,) 0/1/2 — v=0 bị loại khỏi target_weight (không tính loss).
    sigma: độ rộng Gaussian trong không gian HEATMAP (không phải crop) — 2.0 là
        giá trị chuẩn phổ biến cho heatmap 64x48 (xem HRNet/SimpleBaseline),
        CHƯA tune riêng cho bài toán này, so LOO nếu muốn đổi.
    Trả về (target (K,heatmap_h,heatmap_w) float32, target_weight (K,) float32).
    """
    k = keypoints_crop.shape[0]
    target = np.zeros((k, heatmap_h, heatmap_w), dtype=np.float32)
    target_weight = (visibility > 0).astype(np.float32)

    if udp:
        # Mã hoá THỐNG NHẤT với giải mã chuẩn (kiểu UDP): chỉ số heatmap = toạ_độ_crop * (W-1)/(crop_w-1) (KHÔNG làm tròn),
        # tâm Gaussian là SỐ THỰC -> không mất thông tin dưới-pixel. Giải mã dùng 1/(W-1) (decode_fix=False), flip không cần dịch.
        radius = int(sigma * 3)
        ys = np.arange(heatmap_h, dtype=np.float32)[:, None]
        xs = np.arange(heatmap_w, dtype=np.float32)[None, :]
        for i in range(k):
            if target_weight[i] == 0:
                continue
            mu_x = keypoints_crop[i, 0] * (heatmap_w - 1.0) / (crop_w - 1.0)
            mu_y = keypoints_crop[i, 1] * (heatmap_h - 1.0) / (crop_h - 1.0)
            if mu_x < -radius or mu_y < -radius or mu_x >= heatmap_w + radius or mu_y >= heatmap_h + radius:
                target_weight[i] = 0.0
                continue
            g = np.exp(-((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / (2 * sigma ** 2))
            g[g < np.exp(-4.5)] = 0.0   # cắt đuôi ở 3 sigma như bản cũ
            target[i] = g
        return target, target_weight

    feat_stride_x = crop_w / heatmap_w
    feat_stride_y = crop_h / heatmap_h
    def _gauss(sg: float):
        r = int(sg * 3)
        sz = 2 * r + 1
        xr = np.arange(0, sz, dtype=np.float32)
        yr = xr[:, None]
        return r, sz, np.exp(-((xr - sz // 2) ** 2 + (yr - sz // 2) ** 2) / (2 * sg ** 2))

    radius, size, gaussian = _gauss(sigma)
    gaussian_main = gaussian
    g_occ = _gauss(sigma_occluded) if sigma_occluded else None

    for i in range(k):
        if target_weight[i] == 0:
            continue
        if g_occ is not None and visibility[i] == 1:   # khớp bị che (v=1): target rộng hơn -> bớt tự tin quá mức vào 1 pixel
            radius, size, gaussian = g_occ
        else:
            radius, size, gaussian = _gauss(sigma) if False else (int(sigma * 3), 2 * int(sigma * 3) + 1, gaussian_main)
        mu_x = int(keypoints_crop[i, 0] / feat_stride_x + 0.5)
        mu_y = int(keypoints_crop[i, 1] / feat_stride_y + 0.5)
        # keypoint warp ra ngoài vùng heatmap (box crop không bao hết ảnh gốc) -> bỏ qua
        if mu_x < -radius or mu_y < -radius or mu_x >= heatmap_w + radius or mu_y >= heatmap_h + radius:
            target_weight[i] = 0.0
            continue

        gx_min, gx_max = max(0, -mu_x + radius), min(size, heatmap_w - mu_x + radius)
        gy_min, gy_max = max(0, -mu_y + radius), min(size, heatmap_h - mu_y + radius)
        hx_min, hx_max = max(0, mu_x - radius), min(heatmap_w, mu_x + radius + 1)
        hy_min, hy_max = max(0, mu_y - radius), min(heatmap_h, mu_y + radius + 1)
        target[i, hy_min:hy_max, hx_min:hx_max] = gaussian[gy_min:gy_max, gx_min:gx_max]

    return target, target_weight


def generate_oks_target_heatmap(keypoints_crop: np.ndarray, visibility: np.ndarray,
                                 heatmap_h: int, heatmap_w: int, crop_h: int, crop_w: int,
                                 bbox_area: float, warp_scale: np.ndarray,
                                 kpt_sigmas: np.ndarray = KPT_OKS_SIGMAS,
                                 eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Sinh target heatmap theo ĐÚNG hình dạng OKS thật (LẤY CẢM HỨNG ProbPose,
    Purkrabek & Matas, CVPR 2025, arXiv 2412.02254 — xem docs_3/strategy.md
    mục 7.7) THAY VÌ Gaussian đẳng hướng sigma cố định của
    `generate_target_heatmap`.

    KHÔNG phải port y hệt OKSLoss gốc của ProbPose (họ dùng probability map
    chuẩn hoá Sparsemax + risk minimization + Sobel regularizer) — chỉ mượn
    Ý TƯỞNG CỐT LÕI: target nên phản ánh ĐÚNG dung sai OKS thật theo TỪNG
    LOẠI keypoint (vai/hông dung sai lớn hơn cổ tay/mắt cá — xem sigma khác
    nhau trong `KPT_OKS_SIGMAS`) và THEO TỪNG MẪU (object scale = căn bậc hai
    diện tích box GỐC, không phải 1 sigma chung cho mọi keypoint/mọi mẫu như
    bản Gaussian). Vẫn dùng chung cơ chế loss (MSE có trọng số) đã có — chỉ
    đổi HÌNH DẠNG target, không đổi kiến trúc loss.

    bbox_area: diện tích box GỐC (ảnh gốc, w*h) — "s^2" trong công thức OKS
        chuẩn (s = sqrt(area)).
    warp_scale: (2,) = [matrix[0,0], matrix[1,1]] từ `warp_matrix` — tỷ lệ
        pixel ẢNH GỐC trên 1 pixel CROP theo trục x,y (để quy đổi khoảng
        cách heatmap-space về đúng đơn vị ảnh gốc, nhất quán với OKS chuẩn
        — KHÁC Gaussian bản cũ vốn đo khoảng cách trực tiếp trong không gian
        heatmap, không quy đổi về ảnh gốc).
    """
    k = keypoints_crop.shape[0]
    target = np.zeros((k, heatmap_h, heatmap_w), dtype=np.float32)
    target_weight = (visibility > 0).astype(np.float32)

    feat_stride_x = crop_w / heatmap_w
    feat_stride_y = crop_h / heatmap_h
    orig_per_hm_x = feat_stride_x / max(warp_scale[0], eps)
    orig_per_hm_y = feat_stride_y / max(warp_scale[1], eps)
    s2 = max(bbox_area, 1.0)

    ys, xs = np.meshgrid(np.arange(heatmap_h, dtype=np.float32),
                          np.arange(heatmap_w, dtype=np.float32), indexing="ij")

    for i in range(k):
        if target_weight[i] == 0:
            continue
        mu_x = keypoints_crop[i, 0] / feat_stride_x
        mu_y = keypoints_crop[i, 1] / feat_stride_y
        if mu_x < -2 or mu_y < -2 or mu_x >= heatmap_w + 2 or mu_y >= heatmap_h + 2:
            target_weight[i] = 0.0
            continue
        dx_orig = (xs - mu_x) * orig_per_hm_x
        dy_orig = (ys - mu_y) * orig_per_hm_y
        d2 = dx_orig ** 2 + dy_orig ** 2
        k_i = 2.0 * kpt_sigmas[i]
        target[i] = np.exp(-d2 / (2.0 * s2 * k_i ** 2 + eps)).astype(np.float32)

    return target, target_weight


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


def flip_heatmaps(heatmaps: np.ndarray, flip_pairs: list[tuple[int, int]], shift: int = 0) -> np.ndarray:
    """Lật heatmap theo chiều ngang + hoán vị kênh trái/phải — dùng cho
    flip-test TTA (docs_3/strategy.md mục 7.4).

    KHÔNG dịch 1 pixel. `warp_matrix` dùng quy ước `size-1` (kiểu UDP), nên
    lật ảnh crop là đối xứng quanh (W-1)/2 và lật heatmap `[..., ::-1]` đã căn
    đúng. Bản cũ có `np.roll(shift=1)` (quy ước mmpose cũ, không hợp với
    quy ước này) — đo trên checkpoint LOO thật làm TTA GIẢM AP (PETA 0.9216 ->
    0.9146, PA100k 0.8941 -> 0.8890), bỏ shift thì TĂNG (0.9317 / 0.9013)."""
    flipped = heatmaps[..., ::-1].copy()
    if shift:  # dịch sang phải `shift` pixel; =1 khi dùng decode_fix (đã đo: tốt nhất/ngang nhất ở mọi fold)
        flipped[..., shift:] = flipped[..., :-shift].copy()
        flipped[..., :shift] = 0.0
    for a, b in flip_pairs:
        flipped[:, [a, b]] = flipped[:, [b, a]]
    return flipped


def decode_heatmaps(heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray, input_wh: tuple | None = None):
    """DARK decode (Zhang et al. CVPR 2020) — COPY NGUYÊN VĂN từ baseline mẫu
    BTC. heatmaps: (B, K, H, W). Trả về (coords (B,K,2) toạ độ ảnh gốc, scores (B,K))."""
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

    # Hệ số quy đổi chỉ số heatmap -> toạ độ ảnh. `input_wh=(crop_w, crop_h)` (SỬA LỖI, đã đo trên LOO +0.003..+0.010 AP):
    # target huấn luyện đặt đỉnh tại chỉ số = toạ_độ_crop * W / crop_w (generate_target_heatmap), nên giải mã đúng là
    # crop_w / (W * (crop_w - 1)) chứ KHÔNG phải 1 / (W - 1) (sai số hệ thống ~+1.5 px crop, tăng theo vị trí). None = hành vi cũ.
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
                  split_ratio: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Giải mã output `models.build.SimCCHead` (x_logits, y_logits) thành
    toạ độ ẢNH GỐC — CÙNG interface với `decode_heatmaps` (centers/scales ->
    coords, scores) để `predict_and_decode()`/submission dùng được cho cả 2
    loại head mà không cần nhánh code riêng ở tầng gọi.

    Decode bằng KỲ VỌNG (expected value) trên phân phối softmax mỗi trục —
    cho độ chính xác dưới-bin (sub-bin) đơn giản, không cần đạo hàm Hessian
    như DARK (SimCC vốn đã tránh lỗi lượng tử hoá bằng cách chia bin mịn hơn
    `split_ratio` lần — vấn đề DARK giải quyết cho heatmap KHÔNG còn nghiêm
    trọng ở đây). KHÔNG phải cách decode DUY NHẤT hợp lệ (paper gốc SimCC
    cũng dùng argmax thuần) — đây là lựa chọn triển khai, so LOO nếu muốn đổi.

    x_logits/y_logits: (B, K, num_x_bins)/(B, K, num_y_bins) — numpy, CHƯA
    qua softmax. Trả về (coords (B,K,2) toạ độ ảnh gốc, scores (B,K))."""
    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    px = softmax(x_logits.astype(np.float64))  # (B, K, num_x_bins)
    py = softmax(y_logits.astype(np.float64))
    num_x_bins, num_y_bins = px.shape[-1], py.shape[-1]

    exp_x = (px * np.arange(num_x_bins, dtype=np.float64)[None, None, :]).sum(axis=-1)  # (B, K)
    exp_y = (py * np.arange(num_y_bins, dtype=np.float64)[None, None, :]).sum(axis=-1)
    scores = (px.max(axis=-1) + py.max(axis=-1)) / 2.0

    # bin index -> "heatmap-grid pixel" (num_bins = heatmap_size * split_ratio,
    # xem SimCCHead) -> tái dùng CHÍNH XÁC công thức quy đổi heatmap-grid ->
    # ảnh gốc giống decode_heatmaps, để nhất quán 2 đường decode.
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


def degrade_crop(crop: np.ndarray, quality_prob: float = 0.0, color_jitter: float = 0.0,
                 illum_prob: float = 0.0, motion_prob: float = 0.0, lowres_prob: float = 0.0) -> np.ndarray:
    """Augmentation chất lượng ảnh trên crop (HWC float32, 0..255) — mô phỏng khác biệt camera/độ phân giải
    giữa các domain (xem docs_3/strategy.md): giảm rồi phóng lại độ phân giải, blur Gaussian, nén JPEG, nhiễu,
    và jitter màu (độ sáng/tương phản/bão hoà). Chỉ đổi giá trị pixel, KHÔNG đổi hình học nên không ảnh hưởng
    keypoint/target. quality_prob=0 và color_jitter=0 -> trả nguyên crop (hành vi cũ)."""
    import io
    from PIL import Image, ImageFilter
    h, w = crop.shape[:2]
    if color_jitter > 0 and np.random.random() < 0.8:
        b = 1.0 + np.random.uniform(-color_jitter, color_jitter)
        c = 1.0 + np.random.uniform(-color_jitter, color_jitter)
        s = 1.0 + np.random.uniform(-color_jitter, color_jitter)
        crop = crop * b
        crop = (crop - crop.mean()) * c + crop.mean()
        gray = crop.mean(axis=2, keepdims=True)
        crop = (crop - gray) * s + gray
        crop = np.clip(crop, 0.0, 255.0)
    if illum_prob > 0 and np.random.random() < illum_prob:      # ÁNH SÁNG mạnh: thiếu/thừa sáng, gamma, nhiệt độ màu, đôi khi ảnh xám; tối thì nhiễu cảm biến nhiều hơn
        g = float(np.exp(np.random.uniform(np.log(0.3), np.log(1.5)))); gam = np.random.uniform(0.6, 1.8)
        crop = 255.0 * np.power(np.clip(crop, 0.0, 255.0) / 255.0, gam) * g
        crop = crop * np.random.uniform(0.85, 1.15, size=(1, 1, 3)).astype(np.float32)
        if np.random.random() < 0.15:
            crop = np.repeat(crop.mean(axis=2, keepdims=True), 3, axis=2)
        crop = crop + np.random.normal(0.0, np.random.uniform(1.0, 8.0) / max(g, 0.3) ** 0.5, crop.shape)
        crop = np.clip(crop, 0.0, 255.0).astype(np.float32)
    if motion_prob > 0 and np.random.random() < motion_prob:   # MỜ DO CHUYỂN ĐỘNG: trung bình L bản dịch dọc một hướng ngẫu nhiên
        L = int(np.random.randint(5, 22)); ang = np.random.uniform(0.0, np.pi)
        ph, pw = h, w; pad = L // 2 + 1
        P = np.pad(crop, ((pad, pad), (pad, pad), (0, 0)), mode="edge"); acc = np.zeros_like(crop)
        for t in np.linspace(-L / 2.0, L / 2.0, L):
            dy, dx = int(round(t * np.sin(ang))), int(round(t * np.cos(ang)))
            acc += P[pad + dy:pad + dy + ph, pad + dx:pad + dx + pw]
        crop = (acc / L).astype(np.float32)
    if lowres_prob > 0 and np.random.random() < lowres_prob:   # ĐỘ PHÂN GIẢI CỰC THẤP (người rất nhỏ phóng to): thu còn 12–35% rồi phóng lại
        img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)); f = float(np.exp(np.random.uniform(np.log(0.12), np.log(0.35))))
        small = img.resize((max(int(w * f), 6), max(int(h * f), 6)), Image.BILINEAR)
        img = small.resize((w, h), Image.BICUBIC if np.random.random() < 0.5 else Image.BILINEAR)
        crop = np.asarray(img, dtype=np.float32)
    if quality_prob > 0 and np.random.random() < quality_prob:
        img = Image.fromarray(crop.astype(np.uint8))
        if np.random.random() < 0.5:  # giảm rồi phóng lại độ phân giải
            f = np.random.uniform(0.3, 0.9)
            small = img.resize((max(int(w * f), 8), max(int(h * f), 8)), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        if np.random.random() < 0.3:  # blur
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))
        if np.random.random() < 0.5:  # nén JPEG
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(np.random.randint(30, 91)))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        crop = np.asarray(img, dtype=np.float32)
        if np.random.random() < 0.3:  # nhiễu Gaussian nhẹ
            crop = np.clip(crop + np.random.normal(0.0, np.random.uniform(2.0, 8.0), crop.shape), 0.0, 255.0).astype(np.float32)
    return crop


def lower_occlude(crop: np.ndarray, kpts_crop: np.ndarray, vis: np.ndarray, prob: float = 0.0) -> np.ndarray:
    """CHE NỬA DƯỚI (mô phỏng bàn/ghế/xe/người khác che từ thắt lưng xuống — đặc điểm giám sát trong nhà, đám đông).
    Với xác suất `prob`: dải ngang từ quanh hông (hoặc 45–70% chiều cao crop nếu không có hông) tới đáy (hoặc một phần),
    phủ 60–100% bề ngang, tô bằng màu+nhiễu hoặc một mảng nền lấy từ chính crop đã làm mờ. NHÃN GIỮ NGUYÊN. prob=0 -> giữ nguyên."""
    if prob <= 0 or np.random.random() >= prob:
        return crop
    h, w = crop.shape[:2]
    hips = [k for k in (11, 12) if vis[k] > 0 and 0 <= kpts_crop[k, 1] < h]
    ank = [k for k in (13, 14, 15, 16) if vis[k] > 0 and 0 <= kpts_crop[k, 1] < h]
    if hips:
        hy = float(np.mean([kpts_crop[k, 1] for k in hips])); ly = float(np.max([kpts_crop[k, 1] for k in ank])) if ank else h * 0.95
        y0 = int(np.clip(hy + np.random.uniform(-0.1, 0.4) * max(ly - hy, 1.0), 0, h - 8))
    else:
        y0 = int(h * np.random.uniform(0.45, 0.7))
    y1 = h if np.random.random() < 0.7 else int(np.clip(y0 + np.random.uniform(0.3, 0.8) * (h - y0), y0 + 8, h))
    bw = int(w * np.random.uniform(0.6, 1.0)); x0 = int(np.random.randint(0, w - bw + 1))
    c = crop.copy()
    if np.random.random() < 0.5:
        c[y0:y1, x0:x0 + bw] = np.random.uniform(20, 220, size=(1, 1, 3)) + np.random.normal(0, 8, (y1 - y0, bw, 3))
    else:   # mảng nền: lấy vùng phía trên của crop, lật dọc, làm mờ thô
        src = c[: max(y1 - y0, 1)][::-1, x0:x0 + bw]
        src = src[: y1 - y0]
        if src.shape[0] < y1 - y0:
            src = np.resize(src, (y1 - y0, bw, 3))
        k = 5; pad = np.pad(src, ((k, k), (k, k), (0, 0)), mode="edge"); acc = np.zeros_like(src)
        for dy in range(-k, k + 1, 2):
            for dx in range(-k, k + 1, 2):
                acc += pad[k + dy:k + dy + src.shape[0], k + dx:k + dx + src.shape[1]]
        c[y0:y1, x0:x0 + bw] = acc / ((k + 1) ** 2)
    return np.clip(c, 0, 255).astype(np.float32)


LIMB_KPTS = [7, 8, 9, 10, 13, 14, 15, 16]  # khuỷu, cổ tay, đầu gối, mắt cá — khớp khó, hay bị che


def occlude_crop(crop: np.ndarray, kpts_crop: np.ndarray, vis: np.ndarray, prob: float = 0.0) -> np.ndarray:
    """Che khuất tổng hợp trên crop (HWC float32, 0..255): với xác suất `prob`, hoặc che vuông quanh 1-2 khớp tay/chân
    (đúng khớp khó), hoặc che chữ nhật ngẫu nhiên. NHÃN GIỮ NGUYÊN (model phải suy ra khớp bị che).
    prob=0 -> trả nguyên crop (hành vi cũ)."""
    if prob <= 0 or np.random.random() >= prob:
        return crop
    h, w = crop.shape[:2]
    crop = crop.copy()
    if np.random.random() < 0.6:
        cand = [k for k in LIMB_KPTS if vis[k] > 0]
        if cand:
            for k in np.random.choice(cand, size=min(len(cand), np.random.randint(1, 3)), replace=False):
                half = int(np.random.uniform(0.06, 0.16) * w)
                x, y = int(kpts_crop[k, 0]), int(kpts_crop[k, 1])
                x0, x1, y0, y1 = max(x - half, 0), min(x + half, w), max(y - half, 0), min(y + half, h)
                if x1 > x0 and y1 > y0:
                    crop[y0:y1, x0:x1] = np.random.uniform(0, 255, size=(1, 1, 3))
            return crop
    rw, rh = int(np.random.uniform(0.15, 0.4) * w), int(np.random.uniform(0.1, 0.3) * h)
    x0, y0 = np.random.randint(0, max(w - rw, 1)), np.random.randint(0, max(h - rh, 1))
    crop[y0:y0 + rh, x0:x0 + rw] = np.random.uniform(0, 255, size=(1, 1, 3))
    return crop


def paste_distractor(crop: np.ndarray, other: np.ndarray, prob: float = 0.0) -> np.ndarray:
    """Dán một người gây nhiễu (lấy từ crop của mẫu khác, mặt nạ elip mờ viền) lên 1 bên crop, che một phần người mục tiêu —
    mô phỏng crop chứa nhiều người, nhãn VẪN là người mục tiêu ở giữa (xem EDA: ~1.5% ca nhầm người = 15-18% OKS bị mất).
    Người dán nhỏ hơn và lệch sang trái/phải, người mục tiêu giữ đúng vị trí/khung của box. prob=0 -> nguyên crop."""
    if prob <= 0 or np.random.random() >= prob:
        return crop
    from PIL import Image
    h, w = crop.shape[:2]
    ph = int(np.random.uniform(0.45, 0.85) * h)
    pw = int(ph * np.random.uniform(0.45, 0.7))
    patch = Image.fromarray(np.clip(other, 0, 255).astype(np.uint8)).resize((pw, ph), Image.BILINEAR)
    if np.random.random() < 0.5:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    patch = np.asarray(patch, dtype=np.float32)
    yy, xx = np.mgrid[0:ph, 0:pw]
    d = (((xx - pw / 2) / (pw / 2)) ** 2 + ((yy - ph / 2) / (ph / 2)) ** 2)
    mask = np.clip((1.0 - d) * 4.0, 0.0, 1.0)[..., None]  # elip, viền mờ
    side = -1 if np.random.random() < 0.5 else 1
    cx = int(w / 2 + side * np.random.uniform(0.18, 0.4) * w)
    cy = int(np.random.uniform(0.42, 0.65) * h)
    x0, y0 = cx - pw // 2, cy - ph // 2
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + pw, w), min(y0 + ph, h)
    if x1c <= x0c or y1c <= y0c:
        return crop
    crop = crop.copy()
    pm = mask[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    pp = patch[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    crop[y0c:y1c, x0c:x1c] = crop[y0c:y1c, x0c:x1c] * (1 - pm) + pp * pm
    return crop


def truncnorm_unit(size=None) -> np.ndarray:
    """Chuẩn tắc cắt cụt trên [-1, 1] (độ lệch chuẩn hiệu dụng ~0.54) — cách lấy mẫu của công thức affine "stock" (MMPose/ProbPose)."""
    from scipy.stats import truncnorm
    return truncnorm.rvs(-1.0, 1.0, size=size).astype(np.float32)


UPPER_KPTS = list(range(0, 11))   # mũi, đầu, vai, khuỷu, cổ tay (slot tai không có nhãn)
LOWER_KPTS = list(range(11, 17))  # hông, đầu gối, mắt cá


def half_body_box(keypoints: np.ndarray, bbox, prob: float = 0.3, upper_prob: float = 0.7, padding: float = 1.5,
                  min_total: int = 9, min_upper: int = 2, min_lower: int = 3):
    """Với xác suất `prob`, thay box bằng box bao nửa thân trên/dưới (từ keypoint có nhãn) * padding, cắt trong box gốc.
    Trả bbox [x,y,w,h] mới hoặc None (giữ box cũ). Tự cài đặt theo mô tả của công thức half-body trong MMPose."""
    if np.random.random() >= prob:
        return None
    vis = keypoints[:, 2] > 0
    up = [k for k in UPPER_KPTS if vis[k]]
    lo = [k for k in LOWER_KPTS if vis[k]]
    if vis.sum() < min_total:
        return None
    if len(up) >= min_upper and len(lo) >= min_lower:
        sel = up if np.random.random() < upper_prob else lo
    elif len(up) >= min_upper:
        sel = up
    elif len(lo) >= min_lower:
        sel = lo
    else:
        return None
    pts = keypoints[sel, :2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = max((x1 - x0) * padding, 8.0), max((y1 - y0) * padding, 8.0)
    nx0, ny0 = max(cx - w / 2, bbox[0]), max(cy - h / 2, bbox[1])
    nx1, ny1 = min(cx + w / 2, bbox[0] + bbox[2]), min(cy + h / 2, bbox[1] + bbox[3])
    if nx1 - nx0 < 8 or ny1 - ny0 < 8:
        return None
    return [nx0, ny0, nx1 - nx0, ny1 - ny0]


def edge_blackout(image: np.ndarray, bbox, prob: float = 0.8, ratio_range=(0.1, 0.3), context: float = 1.25):
    """Che ĐEN mép ảnh nguồn (mô phỏng người bị cắt ở biên ảnh), GIỮ NGUYÊN hình học crop. Trong vùng box*context (cắt vào ảnh) giữ lại
    một hình chữ nhật chiếm U(1-max,1-min) mỗi chiều, đặt ngẫu nhiên; mọi pixel ngoài hình chữ nhật thành đen.
    Trả (ảnh, (x0,y0,x1,y1) vùng giữ) — vùng giữ = toàn ảnh nếu không áp dụng. Tự cài đặt theo mô tả của ProbPose (CVPR'25)."""
    h, w = image.shape[:2]
    keep = (0, 0, w, h)
    if np.random.random() >= prob:
        return image, keep
    cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
    bw, bh = bbox[2] * context, bbox[3] * context
    x0, y0 = int(max(cx - bw / 2, 0)), int(max(cy - bh / 2, 0))
    x1, y1 = int(min(cx + bw / 2, w)), int(min(cy + bh / 2, h))
    rh, rw = np.random.uniform(1 - ratio_range[1], 1 - ratio_range[0], 2)
    dh, dw = max(int((y1 - y0) * rh), 2), max(int((x1 - x0) * rw), 2)
    xs = np.random.randint(x0, max(x1 - dw, x0 + 1))
    ys = np.random.randint(y0, max(y1 - dh, y0 + 1))
    out = np.zeros_like(image)
    out[ys:ys + dh, xs:xs + dw] = image[ys:ys + dh, xs:xs + dw]
    return out, (xs, ys, xs + dw, ys + dh)


def fourier_swap(crop: np.ndarray, other: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """Fourier Domain Augmentation: thay biên độ TẦN SỐ THẤP (cửa sổ tương đối beta) của `crop` bằng hỗn hợp với biên độ của `other` (pha giữ nguyên) —
    đổi 'style' camera/ánh sáng mà giữ cấu trúc người, nên nhãn không đổi. HWC float32 0..255."""
    h, w = crop.shape[:2]
    bh, bw = max(int(h * beta), 1), max(int(w * beta), 1)
    lam = np.random.uniform(0.0, 1.0)
    cy, cx = h // 2, w // 2
    out = np.empty_like(crop)
    for c in range(crop.shape[2]):
        fa = np.fft.fftshift(np.fft.fft2(crop[:, :, c]))
        fb = np.fft.fftshift(np.fft.fft2(other[:, :, c]))
        amp, pha = np.abs(fa), np.angle(fa)
        sl = (slice(cy - bh, cy + bh + 1), slice(cx - bw, cx + bw + 1))
        amp[sl] = (1.0 - lam) * amp[sl] + lam * np.abs(fb)[sl]
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha))))
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def random_conv(crop: np.ndarray) -> np.ndarray:
    """Random convolution (domain generalization): lọc ảnh bằng nhân ngẫu nhiên k x k (k in {3,5}) rồi trộn với ảnh gốc; đổi texture, giữ hình học."""
    from scipy.ndimage import convolve
    k = int(np.random.choice([3, 5]))
    ker = np.random.randn(k, k).astype(np.float32)
    ker /= max(np.abs(ker).sum(), 1e-6)
    ker[k // 2, k // 2] += 1.0                       # gần phép đồng nhất + nhiễu cấu trúc
    ker /= ker.sum() if abs(ker.sum()) > 1e-3 else 1.0
    out = np.stack([convolve(crop[:, :, c], ker, mode="nearest") for c in range(crop.shape[2])], axis=2)
    a = np.random.uniform(0.3, 1.0)
    return np.clip(a * out + (1.0 - a) * crop, 0.0, 255.0).astype(np.float32)
