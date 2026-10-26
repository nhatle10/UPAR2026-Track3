"""[TRACK 3] Đóng gói checkpoint đã fine-tune thành submission.zip.

KHÁC Track 1/2 (phải vendor `timm`/`transformers` vì submission import trực
tiếp thư viện đó lúc inference): ở đây export bằng `torch.jit.trace` + freeze
(giống hệt cách `export_model.py` mẫu của BTC làm) — submission chỉ cần
`torch`/`numpy`/`Pillow` (đã có sẵn trong runtime image theo docs_3/overview.md),
KHÔNG cần vendor `transformers` (nặng, nhiều phụ thuộc hơn hẳn `timm`). Đơn
giản hơn đáng kể so với `code/export_submission.py`/`code_2/export_submission.py`.

Cách dùng:
    python export_submission.py --checkpoint runs/final.pt \
        --backbone usyd-community/vitpose-plus-small --out submission_track3.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# mean_pose.json gốc của BTC (fallback pose khi box degenerate, w/h<=0) — KHÔNG
# phụ thuộc checkpoint nào (chỉ là thống kê vị trí trung bình trên COCO), dùng
# lại nguyên văn thay vì tự tính lại, xem docs_3/strategy.md + README.
_DEFAULT_MEAN_POSE = HERE / "submission_template" / "assets" / "mean_pose.json"   # copied from the official example submission


class _DropPresence(torch.nn.Module):
    """Wrapper CHỈ dùng lúc export — bỏ `presence_logits` khỏi output trước
    khi trace, để submission.zip hành xử Y HỆT model không có PresenceHead
    (không cần sửa gì `submission_template/run.py`). Checkpoint vẫn load đủ
    `presence_head.*` (state_dict khớp đúng kiến trúc lúc train), presence
    chỉ là tín hiệu huấn luyện phụ trợ — KHÔNG dùng lúc nộp bài, xem
    docs_3/strategy.md mục 7.7 / models/build.py::PresenceHead."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, pixel_values: torch.Tensor):
        return self.inner(pixel_values)[0]


def build_submission(checkpoint: Path, backbone: str, out_zip: Path, mean_pose_json: Path | None = None,
                      tta_flip: bool = False, head_type: str = "default", simcc_split_ratio: int = 2,
                      presence_head: bool = False, input_size=None, padding_factor: float = 1.25,
                      decode_fix: bool = True):
    from models.build import build_model, get_preprocess_config

    if tta_flip and head_type == "simcc":
        raise NotImplementedError("tta_flip chưa hỗ trợ cho head_type='simcc' — xem eval/coco_eval.py")

    # head_type/presence_head PHẢI khớp CHÍNH XÁC lúc train checkpoint này,
    # nếu không load_state_dict lỗi (tham số mới không có trong cấu hình khác).
    model = build_model(backbone, pretrained=False, head_type=head_type, simcc_split_ratio=simcc_split_ratio,
                         use_presence=presence_head, input_size=input_size)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    pre = get_preprocess_config(backbone, input_size=input_size, padding_factor=padding_factor)
    pre["decode_fix"] = bool(decode_fix)  # giải mã khớp target huấn luyện (đã đo +0.003..+0.010 AP); model train bằng encode mới thì đặt False
    pre["flip_shift"] = 1 if decode_fix else 0
    pre["tta_flip"] = tta_flip  # đọc lại trong submission_template/run.py::load_model()
    pre["head_type"] = head_type  # run.py cần biết để chọn đúng nhánh decode (heatmap vs SimCC)
    pre["simcc_split_ratio"] = simcc_split_ratio
    example = torch.zeros(1, 3, pre["input_height"], pre["input_width"])
    # presence_head=True: trace qua wrapper bỏ presence_logits -> submission
    # luôn chỉ có main_output, run.py không cần biết presence_head tồn tại.
    # .eval() trên wrapper: BẮT BUỘC, torch.jit.freeze() đòi module (kể cả
    # wrapper mới dựng, mặc định ở train mode) phải ở eval mode — đã bắt lỗi
    # thật lúc test (RuntimeError nếu thiếu dòng này).
    trace_target = _DropPresence(model).eval() if presence_head else model
    with torch.no_grad():
        traced = torch.jit.trace(trace_target, example, strict=False)
    traced = torch.jit.freeze(traced)
    with torch.no_grad():
        check = traced(example)
    if head_type == "simcc":
        print(f"[*] reload check: x_logits {tuple(check[0].shape)}, y_logits {tuple(check[1].shape)}")
    else:
        print(f"[*] reload check: heatmaps {tuple(check.shape)}")

    mean_pose_src = mean_pose_json or _DEFAULT_MEAN_POSE

    with tempfile.TemporaryDirectory(prefix="upar_track3_submission_") as tmp:
        staging = Path(tmp)
        shutil.copy(HERE / "submission_template" / "run.py", staging / "run.py")
        shutil.copy(HERE / "submission_template" / "metadata.yaml", staging / "metadata.yaml")

        assets = staging / "assets"
        assets.mkdir()
        traced.save(str(assets / "vitpose.torchscript"))
        with open(assets / "preprocess.json", "w") as f:
            json.dump(pre, f, indent=2)

        if mean_pose_src.exists():
            shutil.copy(mean_pose_src, assets / "mean_pose.json")
        else:
            print(f"CẢNH BÁO: không tìm thấy {mean_pose_src} — submission_template/run.py sẽ lỗi "
                  f"nếu gặp box degenerate (w<=0 hoặc h<=0) lúc chấm, vì _prior_pose() cần file này.")

        if out_zip.exists():
            out_zip.unlink()
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in staging.rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    zf.write(f, f.relative_to(staging))

    total_mb = out_zip.stat().st_size / 1e6
    print(f"Đã tạo {out_zip} ({total_mb:.0f} MB)")
    print("Kiểm tra: unzip -l", out_zip, "| head   # run.py phải ở gốc archive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--backbone", default="usyd-community/vitpose-plus-small")
    ap.add_argument("--out", type=Path, default=Path("submission_track3.zip"))
    ap.add_argument("--mean_pose", type=Path, default=None,
                     help="Mặc định dùng assets/mean_pose.json gốc của BTC (fallback box degenerate)")
    ap.add_argument("--tta_flip", action="store_true",
                     help="Bật flip-test TTA lúc nộp bài (gấp đôi thời gian infer, đo LOO trước khi bật — "
                          "xem docs_3/strategy.md mục 7.4)")
    ap.add_argument("--head_type", default="default", choices=["default", "fpn", "simcc"],
                     help="PHẢI khớp model.head_type đã dùng lúc train checkpoint này (mục 7.1/7.5)")
    ap.add_argument("--simcc_split_ratio", type=int, default=2,
                     help="Chỉ dùng khi --head_type simcc, PHẢI khớp lúc train")
    ap.add_argument("--presence_head", action="store_true",
                     help="PHẢI khớp model.presence_head đã dùng lúc train checkpoint này (mục 7.7) — "
                          "presence KHÔNG được đóng gói vào submission, chỉ cần khớp để load_state_dict đúng")
    ap.add_argument("--input_size", type=int, nargs=2, default=None, metavar=("H", "W"),
                     help="PHẢI khớp model.input_size lúc train (vd 384 288); mặc định = độ phân giải pretrain")
    ap.add_argument("--padding_factor", type=float, default=1.25, help="PHẢI khớp model.padding_factor lúc train")
    ap.add_argument("--no_decode_fix", action="store_true", help="Tắt sửa giải mã (chỉ để tái hiện hành vi cũ)")
    args = ap.parse_args()
    build_submission(checkpoint=args.checkpoint, backbone=args.backbone, out_zip=args.out,
                      mean_pose_json=args.mean_pose, tta_flip=args.tta_flip, head_type=args.head_type,
                      simcc_split_ratio=args.simcc_split_ratio, presence_head=args.presence_head,
                      input_size=tuple(args.input_size) if args.input_size else None, padding_factor=args.padding_factor, decode_fix=not args.no_decode_fix)


if __name__ == "__main__":
    main()
