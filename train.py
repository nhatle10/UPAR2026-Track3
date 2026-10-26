"""[TRACK 3] Fine-tune ViTPose+ trên 1 fold LOO (train 2 dataset, eval OKS-AP
trên dataset thứ 3 hoàn toàn chưa thấy — xem docs_3/strategy.md mục 7.1).

`transformers.VitPoseForPoseEstimation` KHÔNG hỗ trợ train built-in (đã verify
qua source code — forward(labels=...) ném NotImplementedError), nên loss/
target heatmap ở đây TỰ VIẾT, không gọi API loss của thư viện.

Cách dùng:
    python train.py --config configs/base.yaml --fold loo_PETA
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")  # xem code/README.md — lỗi "Too many open files" đã gặp thật ở Track 1

sys.path.insert(0, os.path.dirname(__file__))
from data.pose_dataset import PoseDataset, load_datasets, load_pose_annotations
from data.splits import LOOFold, build_loo_folds
from eval.coco_eval import evaluate_oks_ap, predict_and_decode
from models.build import build_model, get_preprocess_config, soup_experts


def get_amp_context(device: str, enable: bool):
    if enable and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def state_dict_to_cpu(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor,
                  ohkm_topk: int | None = None) -> torch.Tensor:
    """MSE có trọng số theo visibility — target_weight=0 (keypoint không có
    nhãn, vd ear) loại khỏi loss, giống valid_mask bên Track 1.

    `ohkm_topk` (TUỲ CHỌN, mặc định None = tắt — xem docs_3/strategy.md mục
    7.4, Online Hard Keypoint Mining, Chen et al. CVPR 2018): nếu đặt, mỗi
    mẫu trong batch chỉ lan truyền gradient qua `ohkm_topk` keypoint có loss
    cao nhất (trong số keypoint còn hiệu lực, target_weight>0), ép model tập
    trung vào điểm khó thay vì bị các điểm dễ "trung bình hoá" loss."""
    w = target_weight[:, :, None, None]
    diff2 = (pred - target) ** 2 * w  # (B, K, H, W)

    if ohkm_topk is not None:
        # OHKM chuẩn (Chen et al. CPN 2018): chọn K keypoint loss cao nhất MỖI
        # MẪU, chỉ lan truyền gradient qua đó — chuẩn hoá theo mẫu (khác nhánh
        # mặc định bên dưới chuẩn hoá theo TOÀN BATCH), đúng công thức gốc OHKM.
        per_kpt = diff2.mean(dim=(2, 3))  # (B, K)
        k = min(ohkm_topk, per_kpt.shape[1])
        topk_vals, _ = per_kpt.topk(k, dim=1)
        return topk_vals.mean()

    denom = w.sum().clamp_min(1.0)
    return diff2.sum() / denom / pred.shape[-1] / pred.shape[-2] * pred.shape[1]


def simcc_loss(x_logits: torch.Tensor, y_logits: torch.Tensor, keypoints_crop: torch.Tensor,
                target_weight: torch.Tensor, crop_w: int, crop_h: int, split_ratio: int,
                sigma: float = 4.0) -> torch.Tensor:
    """Loss cho SimCCHead (docs_3/strategy.md mục 7.1/7.6/7.7, lấy cảm hứng
    SimCC gốc — Li et al. ECCV 2022) — cross-entropy với soft-label Gaussian
    (tương đương KL-divergence tới hằng số cộng không phụ thuộc dự đoán) trên
    2 trục x/y ĐỘC LẬP, KHÁC `heatmap_loss` (MSE trên heatmap 2D).

    keypoints_crop: (B, K, 2) toạ độ GỐC trong không gian crop (pixel, từ
        `PoseDataset.__getitem__`, KHÔNG rasterize) — SimCC cần toạ độ liên
        tục trực tiếp, không cần heatmap target.
    target_weight: (B, K) — TÁI DÙNG target_weight đã tính sẵn bởi
        generate_target_heatmap/generate_oks_target_heatmap (đã xử lý đúng
        visibility=0 VÀ keypoint ngoài vùng crop), không tính lại.
    sigma: độ rộng Gaussian soft-label THEO ĐƠN VỊ BIN (đã mịn hơn pixel
        `split_ratio` lần) — mặc định 4.0 (≈ sigma=2.0 pixel × split_ratio=2
        mặc định, tương đương độ rộng Gaussian của `heatmap_loss`), CHƯA
        tune riêng, so LOO nếu đổi.
    """
    num_x_bins, num_y_bins = x_logits.shape[-1], y_logits.shape[-1]
    bin_stride_x = crop_w / num_x_bins
    bin_stride_y = crop_h / num_y_bins
    mu_x = keypoints_crop[..., 0] / bin_stride_x  # (B, K)
    mu_y = keypoints_crop[..., 1] / bin_stride_y

    x_range = torch.arange(num_x_bins, device=x_logits.device, dtype=torch.float32)
    y_range = torch.arange(num_y_bins, device=y_logits.device, dtype=torch.float32)
    target_x = torch.exp(-((x_range[None, None, :] - mu_x[..., None]) ** 2) / (2 * sigma ** 2))
    target_y = torch.exp(-((y_range[None, None, :] - mu_y[..., None]) ** 2) / (2 * sigma ** 2))
    target_x = target_x / target_x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_y = target_y / target_y.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    log_px = torch.log_softmax(x_logits, dim=-1)
    log_py = torch.log_softmax(y_logits, dim=-1)
    loss_x = -(target_x * log_px).sum(dim=-1)  # (B, K)
    loss_y = -(target_y * log_py).sum(dim=-1)

    denom = target_weight.sum().clamp_min(1.0)
    return ((loss_x + loss_y) * target_weight).sum() / denom


def presence_loss(presence_logits: torch.Tensor, target_weight: torch.Tensor) -> torch.Tensor:
    """BCE cho PresenceHead (docs_3/strategy.md mục 7.7, lấy cảm hứng
    ProbPose) — TÁI DÙNG TRỰC TIẾP `target_weight` đã có sẵn làm nhãn: nó đã
    đúng nghĩa "keypoint có nhãn thật VÀ nằm trong vùng crop" (0/1), chính
    xác là định nghĩa presence — không cần sinh nhãn riêng."""
    return torch.nn.functional.binary_cross_entropy_with_logits(presence_logits, target_weight)


def llrd_groups(model, lr_backbone: float, decay: float, weight_decay: float):
    """Layer-wise lr decay (ViTPose gốc: 0.75 cho B, 0.8 cho L): lớp gần input nhận lr nhỏ hơn
    lr_backbone * decay^(L+1-layer_id), giữ đặc trưng tầng thấp gần pretrain."""
    import re
    n_layers = model.model.backbone.config.num_hidden_layers
    groups: dict[int, list] = {}
    for name, p in model.model.backbone.named_parameters():
        m = re.search(r"layer\.(\d+)\.", name)
        lid = int(m.group(1)) + 1 if m else (0 if "embeddings" in name else n_layers + 1)
        groups.setdefault(lid, []).append(p)
    return [{"params": ps, "lr": lr_backbone * decay ** (n_layers + 1 - lid), "weight_decay": weight_decay}
            for lid, ps in sorted(groups.items())]


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float, llrd: float = 0.0):
    backbone_params = [p for p in model.model.backbone.parameters() if p.requires_grad]
    # head_type="fpn"/"simcc": self.model.head (pretrain gốc) KHÔNG được
    # forward() gọi tới nữa (xem models/build.py) — loại khỏi optimizer.
    if model.head_type == "fpn":
        head_params = list(model.fpn_head.parameters())
    elif model.head_type == "simcc":
        head_params = list(model.simcc_head.parameters())
    else:
        head_params = list(model.model.head.parameters())
    if getattr(model, "use_presence", False):
        head_params = head_params + list(model.presence_head.parameters())
    if getattr(model, "scale_cond", False):
        head_params = head_params + list(model.film.parameters())
    head_group = {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
    if llrd and llrd > 0:
        return llrd_groups(model, lr_backbone, llrd, weight_decay) + [head_group]
    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        head_group,
    ]


def apply_freeze(model, spec: dict) -> None:
    """spec: {"layers": [i,...] đóng băng toàn bộ các khối ViT i, "attention": true -> đóng băng attention mọi khối,
    "ffn": true -> đóng băng FFN (kể cả expert), "embeddings": true -> đóng băng patch/pos embedding}."""
    import re
    bb = model.model.backbone
    n_frozen = 0
    for name, p in bb.named_parameters():
        m = re.search(r"layer\.(\d+)\.", name)
        li = int(m.group(1)) if m else None
        hit = False
        if li is not None and li in set(spec.get("layers", [])):
            hit = True
        if spec.get("attention") and li is not None and ".attention." in name:
            hit = True
        if spec.get("ffn") and li is not None and ".mlp." in name:
            hit = True
        if spec.get("embeddings") and li is None and "embeddings" in name:
            hit = True
        if hit:
            p.requires_grad_(False)
            n_frozen += p.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] đóng băng {n_frozen/1e6:.1f}M / {total/1e6:.1f}M tham số ({n_frozen/total*100:.0f}%)", flush=True)


def train_one_fold(cfg: dict, fold: LOOFold, device: str, epochs: int | None = None,
                    limit_batches: int | None = None) -> dict:
    root = cfg["data"]["root"]
    backbone_name = cfg["model"]["backbone"]
    in_size = cfg["model"].get("input_size")  # TUỲ CHỌN [H, W], mặc định None = độ phân giải pretrain (256x192)
    in_size = tuple(in_size) if in_size else None
    pre = get_preprocess_config(backbone_name, input_size=in_size,
                                padding_factor=cfg["model"].get("padding_factor", 1.25))

    # Đặt seed (trước đây cfg["seed"] không được dùng -> mỗi lần chạy khác nhau, không tái lập được).
    # Worker của DataLoader được PyTorch tự seed random/numpy từ RNG của torch.
    import random
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if fold.name == "iid":   # THƯỚC TRONG-MIỀN: giữ lại 1 khối liên tiếp 10% (theo tên file) của MỖI dataset (Market1501 sắp theo identity => khối rời identity)
        train_samples, val_samples = [], []
        for _d in fold.train_datasets:
            _s = sorted(load_pose_annotations(root, _d), key=lambda x: x.image_path)
            _a, _b = int(0.45 * len(_s)), int(0.55 * len(_s))
            val_samples += _s[_a:_b]
            train_samples += _s[:_a] + _s[_b:]
    else:
        train_samples = load_datasets(root, fold.train_datasets)
        val_samples = load_pose_annotations(root, fold.val_dataset) if fold.val_dataset else []
    # Eval mỗi epoch trên tập con cố định (nhanh hơn nhiều, dùng cho early-stop/scheduler);
    # eval ĐẦY ĐỦ chỉ chạy 1 lần ở cuối bằng checkpoint tốt nhất.
    n_sub = cfg["train"].get("eval_subset", 0)
    if val_samples and n_sub and len(val_samples) > n_sub:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(val_samples), n_sub, replace=False))
        epoch_val_samples = [val_samples[i] for i in idx]
    else:
        epoch_val_samples = val_samples
    eval_workers = cfg["train"].get("eval_workers", 16)

    aug_cfg = cfg["train"]["augment"]
    _jw = cfg["train"].get("joint_weights")
    joint_w = None
    if _jw:
        joint_w = torch.ones(17, device=device)
        for _k, _v in dict(_jw).items():
            joint_w[int(_k)] = float(_v)
    train_ds = PoseDataset(
        train_samples, pre["image_mean"], pre["image_std"], pre["rescale_factor"],
        pre["input_height"], pre["input_width"], pre["heatmap_height"], pre["heatmap_width"],
        augment=True, padding_factor=pre["padding_factor"], normalize_factor=pre["normalize_factor"],
        sigma=cfg["train"]["sigma"], flip_prob=aug_cfg["flip_prob"], scale_jitter=aug_cfg["scale_jitter"],
        rotation_prob=aug_cfg.get("rotation_prob", 0.0), rotation_degrees=aug_cfg.get("rotation_degrees", 30.0),
        crop_shrink_prob=aug_cfg.get("crop_shrink_prob", 0.0),
        crop_shrink_range=tuple(aug_cfg.get("crop_shrink_range", (0.5, 0.85))),
        center_jitter=aug_cfg.get("center_jitter", 0.0),
        target_type=cfg["train"].get("target_type", "gaussian"),
        quality_prob=aug_cfg.get("quality_prob", 0.0), color_jitter=aug_cfg.get("color_jitter", 0.0),
        illum_prob=aug_cfg.get("illum_prob", 0.0), motion_prob=aug_cfg.get("motion_prob", 0.0), lowres_prob=aug_cfg.get("lowres_prob", 0.0),
        occ_prob=aug_cfg.get("occ_prob", 0.0), paste_prob=aug_cfg.get("paste_prob", 0.0), lowocc_prob=aug_cfg.get("lowocc_prob", 0.0),
        aspect_jitter=aug_cfg.get("aspect_jitter", 0.0),
        target_encoding=cfg["train"].get("target_encoding", "legacy"),
        sigma_occluded=cfg["train"].get("sigma_occluded"),
        noisy_label_weight=cfg["train"].get("noisy_label_weight", 1.0),
        stock_affine=aug_cfg.get("stock_affine", False), half_body_prob=aug_cfg.get("half_body_prob", 0.0),
        blackout_prob=aug_cfg.get("blackout_prob", 0.0), blackout_zero_weight=aug_cfg.get("blackout_zero_weight", False),
        shear_prob=aug_cfg.get("shear_prob", 0.0), shear_degrees=aug_cfg.get("shear_degrees", 15.0),
        fda_prob=aug_cfg.get("fda_prob", 0.0), fda_beta=aug_cfg.get("fda_beta", 0.1), randconv_prob=aug_cfg.get("randconv_prob", 0.0),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["data"]["num_workers"], drop_last=True)

    model = build_model(backbone_name, pretrained=cfg["model"]["pretrained"],
                        head_type=cfg["model"].get("head_type", "default"),
                        simcc_split_ratio=cfg["model"].get("simcc_split_ratio", 2),
                        use_presence=cfg["model"].get("presence_head", False),
                        input_size=in_size, scale_cond=cfg["model"].get("scale_cond", False),
                        dataset_experts=cfg["model"].get("dataset_experts", False),
                        expert_index=cfg["model"].get("expert_index", 0)).to(device)
    if cfg["model"].get("drop_path", 0.0) > 0:   # TUỲ CHỌN: stochastic depth (ViTPose gốc 0.3-0.5); mặc định tắt
        from models.build import enable_drop_path
        enable_drop_path(model, float(cfg["model"]["drop_path"]))
        print(f"[drop_path] bật stochastic depth tối đa {cfg['model']['drop_path']}", flush=True)
    freeze_spec = cfg["train"].get("freeze", {})    # TUỲ CHỌN: fine-tune "phẫu thuật" (Lee+ ICLR 2023) — xem apply_freeze()
    if freeze_spec:
        apply_freeze(model, freeze_spec)
    if cfg["model"].get("grad_checkpointing") and hasattr(model.model.backbone, "gradient_checkpointing_enable"):
        # use_reentrant=False: khi lớp đầu bị đóng băng, đầu vào của khối checkpoint không cần grad -> bản reentrant sẽ KHÔNG tính grad cho tham số khối đó
        try:
            model.model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.model.backbone.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(param_groups(model, cfg["train"]["lr_backbone"], cfg["train"]["lr_head"],
                                          cfg["train"]["weight_decay"], llrd=cfg["train"].get("llrd", 0.0)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2) \
        if cfg["train"]["scheduler"] == "plateau" else None
    step_sched = None
    if cfg["train"]["scheduler"] == "cosine":   # TUỲ CHỌN: warmup tuyến tính rồi cosine về min_lr_factor, bước theo iteration
        import math
        total_steps = (epochs if epochs is not None else cfg["train"]["epochs"]) * len(train_loader)
        warm = int(cfg["train"].get("warmup_steps", 500)); minf = float(cfg["train"].get("min_lr_factor", 0.01))
        lam = lambda st: (st + 1) / warm if st < warm else minf + (1 - minf) * 0.5 * (1 + math.cos(math.pi * (st - warm) / max(total_steps - warm, 1)))
        step_sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    ema_model = None
    ema_cpu_every = int(cfg["train"]["ema"].get("cpu_every", 0))  # >0: giữ EMA trên CPU, cập nhật mỗi N bước (tiết kiệm VRAM cho model lớn)
    if cfg["train"]["ema"]["enable"]:
        ema_model = copy.deepcopy(model).eval()
        if ema_cpu_every:
            ema_model = ema_model.cpu()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    from data.pose_dataset import DS_INDEX
    used_experts = [DS_INDEX[d] for d in fold.train_datasets]
    def eval_view(m):   # dataset_experts: trộn trọng số các expert đã train để eval trên dataset CHƯA THẤY (1 lượt suy luận)
        return soup_experts(m, used_experts) if cfg["model"].get("dataset_experts", False) else m
    legacy_enc = cfg["train"].get("target_encoding", "legacy") == "legacy"  # legacy -> giải mã phải dùng decode_fix=True
    amp_ctx = get_amp_context(device, cfg["train"]["amp"])
    n_epochs = epochs if epochs is not None else cfg["train"]["epochs"]
    ohkm_cfg = cfg["train"].get("ohkm", {})
    ohkm_topk = ohkm_cfg.get("topk") if ohkm_cfg.get("enable") else None  # TUỲ CHỌN, mặc định tắt
    best_ap, best_state, no_improve = -1.0, None, 0
    history = []

    import time
    global_step = 0
    for epoch in range(n_epochs):
        t_ep = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for bi, batch in enumerate(train_loader):
            if limit_batches is not None and bi >= limit_batches:
                break
            images = batch["image"].to(device)
            target_weight = batch["target_weight"].to(device)
            if joint_w is not None:   # TUỲ CHỌN: trọng số loss theo khớp (vd tăng khớp đầu có sigma OKS nhỏ); mặc định tắt
                target_weight = target_weight * joint_w

            opt.zero_grad(set_to_none=True)
            with amp_ctx:
                if model.scale_cond:
                    pred = model(images, batch["scale_feat"].to(device))
                elif model.dataset_experts:
                    pred = model(images, dataset_index=batch["ds_index"].to(device))
                else:
                    pred = model(images)
                presence_logits = None
                if model.use_presence:  # forward() trả (main_output, presence_logits) khi bật
                    pred, presence_logits = pred

                if model.head_type == "simcc":
                    keypoints_crop = batch["keypoints_crop"].to(device)
                    loss = simcc_loss(pred[0], pred[1], keypoints_crop, target_weight,
                                       crop_w=pre["input_width"], crop_h=pre["input_height"],
                                       split_ratio=model.simcc_split_ratio,
                                       sigma=cfg["train"].get("simcc_sigma", 4.0))
                else:
                    target = batch["target"].to(device)
                    loss = heatmap_loss(pred, target, target_weight, ohkm_topk=ohkm_topk)

                if presence_logits is not None:
                    loss = loss + cfg["train"].get("presence_loss_weight", 1.0) * \
                        presence_loss(presence_logits, target_weight)
            loss.backward()
            opt.step()
            if step_sched is not None:
                step_sched.step()

            if ema_model is not None and (not ema_cpu_every or (bi + 1) % ema_cpu_every == 0):
                # EMA trên CPU cập nhật mỗi N bước với decay^N (xấp xỉ EMA mỗi bước)
                decay = cfg["train"]["ema"]["decay"] ** (ema_cpu_every or 1)
                if cfg["train"]["ema"].get("ramp", False):   # TUỲ CHỌN: decay tăng dần (1+n)/(10+n) -> không giữ trọng số khởi tạo
                    n_upd = global_step // (ema_cpu_every or 1)
                    decay = min(decay, (1.0 + n_upd) / (10.0 + n_upd))
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.mul_(decay).add_(p.detach().to(ema_p.device), alpha=1 - decay)
                    # SỬA LỖI (đã đo): EMA trước đây chỉ cập nhật parameters(), BỎ SÓT buffer => running_mean/var của BatchNorm
                    # (head ViTPose + FPN/SimCC/Presence) đứng yên ở giá trị khởi tạo. Sao chép buffer từ model đang train.
                    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
                        ema_b.copy_(b.detach().to(ema_b.device))

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)

        metrics = {"AP": float("nan")}
        if epoch_val_samples and (epoch % cfg["train"]["eval_every"] == 0 or epoch == n_epochs - 1):
            eval_model = ema_model if ema_model is not None else model
            if ema_model is not None and ema_cpu_every:
                eval_model.to(device)   # EMA giữ trên CPU để tiết kiệm VRAM -> đưa lên GPU khi eval
            dets = predict_and_decode(eval_view(eval_model), epoch_val_samples, pre, device,
                                       batch_size=cfg["train"]["eval_batch_size"],
                                       tta_flip=cfg["train"].get("tta_flip", False),
                                       num_workers=eval_workers, decode_fix=legacy_enc)
            if ema_model is not None and ema_cpu_every:
                eval_model.cpu();torch.cuda.empty_cache()
            metrics = evaluate_oks_ap(epoch_val_samples, dets)
            print(f"[{fold.name}] epoch {epoch} train_loss={avg_loss:.4f} "
                  f"AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} "
                  f"AP_medium={metrics['AP_medium']:.4f} AP_large={metrics['AP_large']:.4f} "
                  f"({time.time() - t_ep:.0f}s)", flush=True)
            history.append({"epoch": epoch, "train_loss": avg_loss, **metrics})
            _ct = cfg["train"].get("curve_tag")   # TUỲ CHỌN: lưu EMA mỗi epoch của run có val -> đường cong học để CHỌN SỐ EPOCH (LR không đổi nên epoch k của run dài == run k epoch)
            if _ct:
                os.makedirs(cfg["output_dir"], exist_ok=True)
                torch.save(state_dict_to_cpu(ema_model if ema_model is not None else model),
                           os.path.join(cfg["output_dir"], f"curve_{_ct}_ep{epoch + 1}.pt"))

            if scheduler is not None and not np.isnan(metrics["AP"]):
                scheduler.step(metrics["AP"])

            if cfg["train"].get("select", "best") == "last":   # TUỲ CHỌN: luôn lấy epoch cuối (khớp quy trình `final` không có validation)
                best_ap = metrics["AP"]
                best_state = state_dict_to_cpu(eval_model)
            elif not np.isnan(metrics["AP"]) and metrics["AP"] > best_ap:
                best_ap = metrics["AP"]
                best_state = state_dict_to_cpu(eval_model)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["train"]["early_stop_patience"] and cfg["train"].get("select", "best") != "last":
                    print(f"[{fold.name}] early stop tại epoch {epoch} (không cải thiện "
                          f"{cfg['train']['early_stop_patience']} epoch liên tiếp)")
                    break
        else:
            print(f"[{fold.name}] epoch {epoch} train_loss={avg_loss:.4f} (không eval fold 'final', val_dataset=None)")
            if cfg["train"].get("save_every_epoch", False):   # TUỲ CHỌN: lưu EMA mỗi epoch để chọn số epoch sau (fold 'final' không có validation)
                os.makedirs(cfg["output_dir"], exist_ok=True)
                torch.save(state_dict_to_cpu(ema_model if ema_model is not None else model),
                           os.path.join(cfg["output_dir"], f"epoch_{epoch + 1}.pt"))

    if best_state is None:  # fold "final" (val_dataset=None) hoặc chưa eval lần nào
        best_state = state_dict_to_cpu(ema_model if ema_model is not None else model)

    full_metrics = None
    if val_samples and best_ap >= 0:
        # Eval đầy đủ 1 lần với checkpoint tốt nhất (số báo cáo chính thức của fold)
        eval_model = ema_model if ema_model is not None else model
        eval_model.load_state_dict(best_state)
        if ema_model is not None and ema_cpu_every:
            eval_model.to(device)
        dets = predict_and_decode(eval_view(eval_model), val_samples, pre, device,
                                   batch_size=cfg["train"]["eval_batch_size"],
                                   tta_flip=cfg["train"].get("tta_flip", False), num_workers=eval_workers,
                                   decode_fix=legacy_enc)
        full_metrics = evaluate_oks_ap(val_samples, dets)
        print(f"[{fold.name}] FULL-EVAL best ckpt: AP={full_metrics['AP']:.4f} AP50={full_metrics['AP50']:.4f} "
              f"AP_medium={full_metrics['AP_medium']:.4f} AP_large={full_metrics['AP_large']:.4f}", flush=True)
        best_ap = full_metrics["AP"]
        tight = evaluate_oks_ap(val_samples, dets, area_scale=0.6)   # thước đo CHẶT (f=0.6, đã hiệu chuẩn so với điểm Codabench): chọn phương pháp theo cả hai
        print(f"[{fold.name}] FULL-EVAL tight(f=0.6): AP={tight['AP']:.4f}", flush=True)

    return {"model": model, "state_dict": best_state, "best_ap": best_ap, "history": history,
            "full_metrics": full_metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", required=True, help="vd loo_PETA, loo_Market1501, loo_PA100k, final")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_batches", type=int, default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    from data.splits import build_final_split
    fold = next((f for f in build_loo_folds() if f.name == args.fold), None)
    if fold is None and args.fold == "final":
        fold = build_final_split()
    if fold is None:
        raise ValueError(f"Không tìm thấy fold {args.fold}")

    result = train_one_fold(cfg, fold, device, epochs=args.epochs, limit_batches=args.limit_batches)
    out_path = os.path.join(cfg["output_dir"], f"{args.fold}.pt")
    torch.save(result["state_dict"], out_path)
    print(f"Đã lưu {out_path} (best_ap={result['best_ap']:.4f})")


if __name__ == "__main__":
    main()
