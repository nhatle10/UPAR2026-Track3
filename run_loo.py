"""[TRACK 3] Train + eval đủ 3 fold LOO, gộp kết quả — tương đương
code/run_loo.py (Track 1) / code_2/run_loo_retrieval.py (Track 2).

Cách dùng:
    python run_loo.py --config configs/base.yaml --tag baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from data.splits import build_loo_folds
from train import train_one_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_batches", type=int, default=None)
    ap.add_argument("--folds", nargs="*", default=None, help="chỉ chạy các fold này (vd loo_PETA loo_PA100k)")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b.c=value",
                    help="ghi đè config, vd train.augment.rotation_prob=0.5 (giá trị parse bằng YAML)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p_ in parts[:-1]:
            node = node[p_]
        node[parts[-1]] = yaml.safe_load(val)
    cfg["output_dir"] = os.path.join(cfg["output_dir"], args.tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    per_fold = []
    from data.splits import LOOFold, DATASETS
    _all = build_loo_folds() + [LOOFold(name="iid", train_datasets=list(DATASETS), val_dataset="IID")]
    for fold in [f for f in _all if (args.folds is None and f.name != "iid") or (args.folds is not None and f.name in args.folds)]:
        result = train_one_fold(cfg, fold, device=device, epochs=args.epochs, limit_batches=args.limit_batches)
        ckpt_path = os.path.join(cfg["output_dir"], f"{args.tag}_{fold.name}.pt")
        torch.save(result["state_dict"], ckpt_path)
        per_fold.append({"fold": fold.name, "best_ap": result["best_ap"], "history": result["history"]})
        print(f"[{fold.name}] best_ap={result['best_ap']:.4f} -> {ckpt_path}")

    aps = [r["best_ap"] for r in per_fold if not np.isnan(r["best_ap"])]
    agg = {"AP_mean": float(np.mean(aps)) if aps else float("nan"),
           "AP_std": float(np.std(aps)) if aps else float("nan")}
    print("\n=== LOO aggregate (trung bình qua 3 dataset) ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")

    log_path = os.path.join(os.path.dirname(cfg["output_dir"]), "loo_results.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps({"tag": args.tag, "config": args.config, "overrides": args.set, "per_fold": per_fold, "aggregate": agg}) + "\n")
    print(f"\nĐã ghi vào {log_path}")


if __name__ == "__main__":
    main()
