"""Eval đầy đủ checkpoint LOO với/không flip-TTA trên toàn bộ dataset giữ lại.
Dùng: python eval_ckpt.py --tag crop_shrink [--head_type default]"""
import argparse, os, sys, yaml, torch, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))
from data.pose_dataset import load_pose_annotations
from eval.coco_eval import evaluate_oks_ap, predict_and_decode
from models.build import build_model, get_preprocess_config

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--head_type", default="default")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config)); pre = get_preprocess_config(cfg["model"]["backbone"])
rows = []
for ds in ("Market1501", "PA100k", "PETA"):
    S = load_pose_annotations(cfg["data"]["root"], ds)
    m = build_model(cfg["model"]["backbone"], pretrained=False, head_type=args.head_type).cuda().eval()
    m.load_state_dict(torch.load(f"runs/{args.tag}/{args.tag}_loo_{ds}.pt", map_location="cuda"))
    r = {}
    for tta in (False, True):
        with contextlib.redirect_stdout(io.StringIO()):
            met = evaluate_oks_ap(S, predict_and_decode(m, S, pre, "cuda", batch_size=64, tta_flip=tta))
        r[tta] = met["AP"]
    rows.append((ds, r[False], r[True])); print(f"{ds:11s} noTTA={r[False]:.4f}  flipTTA={r[True]:.4f}  delta={r[True]-r[False]:+.4f}", flush=True)
print(f"MEAN        noTTA={sum(x[1] for x in rows)/3:.4f}  flipTTA={sum(x[2] for x in rows)/3:.4f}")
