"""Hiệu chỉnh lại thống kê BatchNorm của một checkpoint bằng ảnh TRAIN (KHÔNG dùng ảnh test) — sửa lỗi EMA bỏ sót buffer.
Dùng:  python recalibrate_bn.py --checkpoint runs/final/final.pt --out runs/final/final_bn.pt --config configs/final.yaml
(đã đo trên LOO: +0.0012..+0.0037 AP không TTA, +0.0005..+0.0038 có TTA, head mặc định; tác động lớn hơn cho head train từ đầu)"""
import argparse, os, sys, yaml, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.pose_dataset import load_datasets
from data.splits import build_loo_folds, build_final_split
from data.transforms import box_to_center_scale, warp_affine, warp_matrix
from models.build import build_model, get_preprocess_config
from PIL import Image


def recalibrate(model, samples, pre, device="cuda", n=6000, batch=64, workers=8):
    ih, iw = pre["input_height"], pre["input_width"]
    mean = np.asarray(pre["image_mean"], np.float32).reshape(3, 1, 1)
    std = np.asarray(pre["image_std"], np.float32).reshape(3, 1, 1)

    class DS(torch.utils.data.Dataset):
        def __len__(s):
            return min(n, len(samples))

        def __getitem__(s, i):
            x = samples[i]
            im = np.asarray(Image.open(x.image_path).convert("RGB"), np.float32)
            c, sc = box_to_center_scale(x.bbox, iw, ih, pre["normalize_factor"], pre["padding_factor"])
            cr = warp_affine(im, warp_matrix(c, sc, iw, ih), ih, iw)
            return torch.from_numpy(((cr.transpose(2, 0, 1) * pre["rescale_factor"] - mean) / std).astype(np.float32))

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(DS(), batch_size=batch, num_workers=workers):
            model(x.to(device))
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--datasets", nargs="*", default=["Market1501", "PA100k", "PETA"], help="dataset TRAIN dùng để hiệu chỉnh")
    ap.add_argument("--input_size", type=int, nargs=2, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--padding_factor", type=float, default=1.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=6000, help="số ảnh train dùng để hiệu chỉnh")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    bb = a.backbone or cfg["model"]["backbone"]
    size = tuple(a.input_size) if a.input_size else None
    pre = get_preprocess_config(bb, input_size=size, padding_factor=a.padding_factor)
    if a.device == "cpu":
        torch.set_num_threads(16)
    model = build_model(bb, pretrained=False, input_size=size).to(a.device).eval()
    model.load_state_dict(torch.load(a.checkpoint, map_location=a.device))
    samples = load_datasets(cfg["data"]["root"], a.datasets)
    rng = np.random.RandomState(0)
    samples = [samples[i] for i in rng.permutation(len(samples))[:a.n]]
    recalibrate(model, samples, pre, device=a.device, n=a.n, batch=a.batch, workers=4)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, a.out)
    print("đã lưu", a.out)


if __name__ == "__main__":
    main()
