"""[TRACK 3] Điểm vào của bản nộp GHÉP NHIỀU MODEL (ensemble). Trong zip, file này là `run.py`; file `single_impl.py`
là bản sao của `submission_template/run.py` (chỉ dùng lại các hàm thuần: box_to_center_scale/warp/decode/flip...).

assets/ensemble.json: {"members": [{"model": "assets/m0.torchscript", "meta": "assets/m0_preprocess.json", "weight": 0.5}, ...]}
Mỗi member có preprocess riêng (kích thước crop, padding, decode_fix, flip_shift, tta_flip). Với mỗi box: chạy từng member
(flip-TTA nếu bật) -> giải mã ra toạ độ ảnh gốc -> trung bình có trọng số TOẠ ĐỘ và điểm tin cậy.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # nền tảng có thể nạp run.py mà không thêm thư mục zip vào sys.path
import single_impl as S

HERE = os.path.dirname(os.path.abspath(__file__))
ENS_PATH = os.path.join(HERE, "assets", "ensemble.json")
BATCH_SIZE = 8

_MEMBERS: list[dict] | None = None
_FUSION = "coords"   # "coords": trộn toạ độ đã giải mã (mặc định cũ); "heatmap": trộn heatmap rồi giải mã 1 lần (cần CÙNG hình học crop)
_SCORE = None
_DEVICE = None
_PRIOR = None


def _fwd(mb, x):
    """Forward 1 member; member fp16 chạy dưới torch.autocast (fp16), trả float32."""
    if mb.get("half"):
        with torch.autocast("cuda", dtype=torch.float16):
            return mb["model"](x).float()
    return mb["model"](x).float()


def load_model() -> None:
    global _SCORE, _MEMBERS, _DEVICE, _PRIOR, _FUSION
    if _MEMBERS is not None:
        return
    with open(ENS_PATH, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    with open(os.path.join(HERE, "assets", "mean_pose.json"), "r", encoding="utf-8") as fh:
        _PRIOR = json.load(fh)["mean_pose_normalized"]
    S._PRIOR = _PRIOR  # single_impl._prior_pose dùng biến này khi box suy biến
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    members = []
    total_w = sum(float(m.get("weight", 1.0)) for m in spec["members"])
    for m in spec["members"]:
        with open(os.path.join(HERE, m["meta"]), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        model = torch.jit.load(os.path.join(HERE, m["model"]), map_location=_DEVICE)
        model.eval()
        half = bool(m.get("fp16", False)) and _DEVICE.type == "cuda"   # TUỲ CHỌN: chạy member ở fp16 (mặc định tắt -> hành vi cũ)
        members.append({
            "model": model, "half": half, "meta": meta, "w": float(m.get("weight", 1.0)) / total_w,
            "mean": np.asarray(meta["image_mean"], dtype=np.float32).reshape(3, 1, 1),
            "std": np.asarray(meta["image_std"], dtype=np.float32).reshape(3, 1, 1),
            "decode_wh": (int(meta["input_width"]), int(meta["input_height"])) if meta.get("decode_fix", False) else None,
            "flip_shift": int(meta.get("flip_shift", 0)), "tta": bool(meta.get("tta_flip", False)),
        })
    _FUSION = str(spec.get("fusion", "coords"))
    _sc = spec.get("score")
    _SCORE = None if not _sc else {"joints": [int(k) for k in _sc["joints"]], "power": float(_sc["power"])}
    if _FUSION == "heatmap":
        keys = ("input_width", "input_height", "padding_factor", "normalize_factor", "rescale_factor", "image_mean", "image_std", "decode_fix", "flip_shift")
        for mb in members[1:]:
            for k in keys:
                if mb["meta"].get(k) != members[0]["meta"].get(k):
                    raise ValueError(f"fusion=heatmap yêu cầu cùng hình học/mã hoá: khác ở '{k}'")
    _MEMBERS = members
    print(f"[ensemble] {len(members)} members on {_DEVICE}: " + ", ".join(
        f"{m['meta']['input_width']}x{m['meta']['input_height']} pad={m['meta']['padding_factor']} w={m['w']:.2f}"
        for m in members), flush=True)


def _prep(image: np.ndarray, box, mb: dict):
    meta = mb["meta"]
    out_w, out_h = int(meta["input_width"]), int(meta["input_height"])
    center, scale = S.box_to_center_scale(box, out_w, out_h, float(meta["normalize_factor"]), float(meta["padding_factor"]))
    crop = S.warp_affine(image, S.warp_matrix(center, scale, out_w, out_h), out_h, out_w)
    tensor = crop.transpose(2, 0, 1) * float(meta["rescale_factor"])
    return ((tensor - mb["mean"]) / mb["std"]).astype(np.float32), center, scale


def _member_predict(image: np.ndarray, boxes: list, mb: dict):
    """Trả (coords (N,17,2), joint_scores (N,17)) của 1 member cho N box hợp lệ."""
    coords_all, scores_all = [], []
    for start in range(0, len(boxes), BATCH_SIZE):
        chunk = boxes[start:start + BATCH_SIZE]
        tens, cs, ss = [], [], []
        for b in chunk:
            t, c, s = _prep(image, b, mb)
            tens.append(t); cs.append(c); ss.append(s)
        batch = torch.from_numpy(np.stack(tens)).to(_DEVICE)
        hm = _fwd(mb, batch).float().cpu().numpy()
        if mb["tta"]:
            hf = _fwd(mb, torch.flip(batch, dims=[-1])).float().cpu().numpy()
            hm = (hm + S.flip_heatmaps(hf, mb["flip_shift"])) * 0.5
        co, sc = S.decode_heatmaps(hm, np.stack(cs), np.stack(ss), mb["decode_wh"])
        coords_all.append(co); scores_all.append(sc)
    return np.concatenate(coords_all), np.concatenate(scores_all)


def _fused_predict(image: np.ndarray, boxes: list):
    """Trộn HEATMAP (trung bình có trọng số, mỗi model đã flip-TTA) rồi giải mã DARK 1 lần. Trả (coords (N,17,2), joint_scores (N,17))."""
    m0 = _MEMBERS[0]
    coords_all, scores_all = [], []
    for start in range(0, len(boxes), BATCH_SIZE):
        chunk = boxes[start:start + BATCH_SIZE]
        tens, cs, ss = [], [], []
        for b in chunk:
            t_, c_, s_ = _prep(image, b, m0)
            tens.append(t_); cs.append(c_); ss.append(s_)
        batch = torch.from_numpy(np.stack(tens)).to(_DEVICE)
        H = 0.0
        for mb in _MEMBERS:
            hm = _fwd(mb, batch).float().cpu().numpy()
            if mb["tta"]:
                hf = _fwd(mb, torch.flip(batch, dims=[-1])).float().cpu().numpy()
                hm = (hm + S.flip_heatmaps(hf, mb["flip_shift"])) * 0.5
            H = H + mb["w"] * hm
        co, sc = S.decode_heatmaps(H, np.stack(cs), np.stack(ss), m0["decode_wh"])
        coords_all.append(co); scores_all.append(sc)
    return np.concatenate(coords_all), np.concatenate(scores_all)


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    if _MEMBERS is None:
        load_model()
    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}
    orig_boxes = boxes
    image, boxes, inv_scale = S._load_image_matching_boxes(sample, boxes)
    keypoints: list[Any] = [None] * len(boxes)
    scores: list[float] = [0.0] * len(boxes)
    usable = [i for i, b in enumerate(boxes) if float(b[2]) > 0 and float(b[3]) > 0]
    for i in set(range(len(boxes))) - set(usable):
        keypoints[i] = S._prior_pose(orig_boxes[i])
        scores[i] = 0.01
    if usable:
        ubox = [boxes[i] for i in usable]
        if _FUSION == "heatmap":
            coords, jscore = _fused_predict(image, ubox)
        else:
            coords = 0.0
            jscore = 0.0
            for mb in _MEMBERS:
                c, s = _member_predict(image, ubox, mb)
                coords = coords + mb["w"] * c
                jscore = jscore + mb["w"] * s
        for j, i in enumerate(usable):
            keypoints[i] = [[float(x) / inv_scale[0], float(y) / inv_scale[1], 1.0] for x, y in coords[j]]
            if _SCORE is None:
                scores[i] = float(np.mean(jscore[j]))
            else:   # TUỲ CHỌN: trung bình bậc p trên tập khớp chọn trước (ensemble.json "score": {"joints": [...], "power": p})
                sj = np.clip(np.asarray(jscore[j], dtype=np.float64)[_SCORE["joints"]], 1e-6, None)
                scores[i] = float(np.mean(sj ** _SCORE["power"]) ** (1.0 / _SCORE["power"]))
    return {"keypoints": keypoints, "scores": scores}
