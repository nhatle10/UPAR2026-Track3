"""Sinh 3 fold leave-one-dataset-out (LOO), giống hệt tinh thần
code/data/splits.py (Track 1) — Track 3 dùng CHUNG 3 dataset nguồn
(Market1501/PA100k/PETA, đã xác nhận qua attribute_labels khớp Track 1), chỉ
khác là không có train/val tách sẵn (chỉ có train_<dataset>.json), nên LOO là
cách DUY NHẤT có validation nội bộ đáng tin cho Track 3.
"""
from __future__ import annotations

from dataclasses import dataclass

DATASETS = ["Market1501", "PA100k", "PETA"]

# Tên dataset trong code_3 (PascalCase, khớp Track 1) -> tên file
# train_<x>.json thật trong starter kit Track 3 (chữ thường, không dấu).
DATASET_TO_JSON = {
    "Market1501": "train_market.json",
    "PA100k": "train_pa100k.json",
    "PETA": "train_peta.json",
}


@dataclass
class LOOFold:
    name: str
    train_datasets: list[str]
    val_dataset: str


def build_loo_folds(datasets: list[str] = DATASETS) -> list[LOOFold]:
    folds = []
    for held_out in datasets:
        train = [d for d in datasets if d != held_out]
        folds.append(LOOFold(name=f"loo_{held_out}", train_datasets=train, val_dataset=held_out))
    return folds


def build_final_split(datasets: list[str] = DATASETS) -> LOOFold:
    """Train bản nộp cuối trên CẢ 3 dataset — dùng train_upar_all.json thật
    (đã verify = hợp của 3 file train_<dataset>.json, không phải tự gộp)."""
    return LOOFold(name="final", train_datasets=list(datasets), val_dataset=None)
