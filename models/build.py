"""Wrap transformers.VitPoseForPoseEstimation — xem docs_3/strategy.md mục 6
cho các fact đã verify thật (shape heatmap, dataset_index, labels= chưa hỗ trợ).

QUAN TRỌNG cho submission.zip: file này CHỈ được import torch + transformers
(không import từ code_3/data, code_3/train...) để export_submission.py có thể
copy riêng file này vào submission mà không kéo theo cả project — giống quy
ước model_def.py bên code_2/export_submission.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

COCO_DATASET_INDEX = 0  # đúng theo export_model.py mẫu của BTC — ta LUÔN dùng expert này


class FPNHead(nn.Module):
    """Head thay thế, LẤY CẢM HỨNG từ ViTDet Simple Feature Pyramid (Li et al.,
    ECCV 2022, arXiv 2203.16527 — xem docs_3/strategy.md mục 7.5). KHÔNG phải
    port nguyên văn (ViTDet làm pyramid đa cấp cho multi-level RPN detection,
    ở đây chỉ cần 1 heatmap output duy nhất) — chỉ mượn Ý TƯỞNG cốt lõi: dựng
    pyramid đa tỷ lệ TỪ MỘT feature map cuối cùng của ViT (không cần backbone
    phân cấp) qua nhiều nhánh conv/deconv khác nhau rồi fuse, thay vì chỉ
    upsample tuyến tính 1 đường như head gốc của ViTPose.

    2 nhánh, khác nhau về cách upsample (bù trừ điểm yếu cho nhau):
    - `deconv`: ConvTranspose2d học được (giống tinh thần head gốc ViTPose).
    - `bilinear`: upsample nội suy cố định + conv — không có artefact dạng
      bàn cờ (checkerboard) của deconv, bù thông tin khác nhánh deconv.
    Concat 2 nhánh rồi fuse bằng conv 1x1 ra heatmap.

    Head này train TỪ ĐẦU (không có trọng số pretrain, khác head gốc) — chỉ
    backbone giữ pretrain, đúng tinh thần "mọi điều chỉnh chỉ xảy ra lúc
    fine-tune" mà ViTDet đã verify là áp dụng được cho ViT đã pretrain."""

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        mid = in_channels // 2
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.bilinear = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(mid * 2, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.deconv(x)
        b = self.bilinear(x)
        return self.fuse(torch.cat([a, b], dim=1))


class SimCCHead(nn.Module):
    """Head thay thế, LẤY CẢM HỨNG SimCC (Li et al., ECCV 2022, arXiv
    2107.03332) và bằng chứng độc lập về lợi thế ở độ phân giải thấp
    (RTMPose, HRPVT — xem docs_3/strategy.md mục 7.1/7.6/7.7). Thay vì dự
    đoán heatmap 2D rồi argmax (có sai số lượng tử hoá), coi bài toán là 2
    bài CLASSIFICATION độc lập: phân loại toạ độ x và y riêng biệt thành các
    bin rời rạc, mịn hơn pixel gốc `split_ratio` lần (giảm sai số lượng tử).

    KHÔNG dùng heatmap — output là 2 tensor logits (x_logits, y_logits),
    KHÁC hẳn `FPNHead`/head gốc (luôn trả về 1 tensor heatmap `(B,K,H,W)`).
    Đây là thay đổi CONTRACT đầu ra của `PoseModel.forward()`, ảnh hưởng tới
    loss/decode/export — xem `train.py::simcc_loss`,
    `data/transforms.py::decode_simcc`, `submission_template/run.py`.

    Kiến trúc: conv nhỏ nén feature map backbone về K kênh (1 kênh/keypoint),
    flatten không gian (Hp*Wp), rồi 2 projection ĐỘC LẬP THEO TỪNG KEYPOINT
    (không chia sẻ trọng số giữa các keypoint — cài qua `nn.Conv1d` với
    `groups=num_keypoints`, mỗi group chỉ thấy đúng 1 kênh/keypoint của
    chính nó) ra num_x_bins/num_y_bins logits."""

    def __init__(self, in_channels: int, num_keypoints: int, patch_h: int, patch_w: int,
                 heatmap_h: int, heatmap_w: int, split_ratio: int = 2):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.split_ratio = split_ratio
        self.num_x_bins = heatmap_w * split_ratio
        self.num_y_bins = heatmap_h * split_ratio

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_keypoints, kernel_size=1),
        )
        flat_dim = patch_h * patch_w
        self.fc_x = nn.Conv1d(num_keypoints, num_keypoints * self.num_x_bins,
                               kernel_size=flat_dim, groups=num_keypoints)
        self.fc_y = nn.Conv1d(num_keypoints, num_keypoints * self.num_y_bins,
                               kernel_size=flat_dim, groups=num_keypoints)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.reduce(x)  # (B, K, Hp, Wp)
        b = feat.shape[0]
        flat = feat.reshape(b, self.num_keypoints, -1)  # (B, K, Hp*Wp)
        x_logits = self.fc_x(flat).reshape(b, self.num_keypoints, self.num_x_bins)
        y_logits = self.fc_y(flat).reshape(b, self.num_keypoints, self.num_y_bins)
        return x_logits, y_logits


class PresenceHead(nn.Module):
    """Head PHỤ (tuỳ chọn, cộng thêm vào BẤT KỲ `head_type` nào), dự đoán
    PRESENCE PROBABILITY — xác suất keypoint có NẰM TRONG vùng crop
    (activation window) hay không, TÁCH BIỆT với độ tin cậy vị trí của head
    chính. LẤY CẢM HỨNG ProbPose (Purkrabek & Matas, CVPR 2025, arXiv
    2412.02254 — xem docs_3/strategy.md mục 7.7): giải quyết lỗ hổng
    heatmap/SimCC không có cách biểu diễn "không nhìn thấy", cứ đoán bừa 1
    vị trí khi keypoint bị cắt khỏi crop.

    Thiết kế TỐI GIẢN (không port đầy đủ kiến trúc ProbPose — họ dùng
    probability map Sparsemax đầy đủ, ở đây chỉ cần 1 scalar presence/
    keypoint): nén feature map backbone về K kênh (1/keypoint, giống
    SimCCHead), MAX-pool không gian mỗi kênh (max nhạy với 1 đỉnh hoạt động
    mạnh dù nhỏ, hợp hơn average cho tín hiệu "có tồn tại hay không"), rồi 1
    phép affine học được (scale+bias riêng từng keypoint) ra logit.

    CHƯA tích hợp presence vào decode/score cuối cùng (chỉ dùng làm tín hiệu
    huấn luyện phụ trợ cho backbone/feature chia sẻ) — xem "CHƯA làm" trong
    code_3/README.md."""

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(num_keypoints))
        self.bias = nn.Parameter(torch.zeros(num_keypoints))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.reduce(x)  # (B, K, Hp, Wp)
        pooled = feat.amax(dim=(2, 3))  # (B, K) — max-pool theo không gian, riêng từng keypoint
        return pooled * self.scale + self.bias  # (B, K) logits


class PoseModel(nn.Module):
    """Bọc VitPoseForPoseEstimation để forward() trả thẳng heatmaps tensor
    (không phải dataclass output), tiện cho torch.jit.trace/export + train loop.

    `head_type` (TUỲ CHỌN, mặc định "default" — docs_3/strategy.md mục 7.5):
    - "default": head pretrain gốc của ViTPose+, hành vi GIỐNG HỆT bản trước
      khi thêm tham số này (đã verify, xem code_3/README.md).
    - "fpn": thay bằng `FPNHead` train từ đầu (backbone vẫn giữ pretrain) —
      lấy feature map cuối của backbone trực tiếp (bỏ qua head pretrain gốc),
      dùng CÙNG lệnh `forward_with_filtered_kwargs` + reshape mà
      `VitPoseForPoseEstimation.forward()` thật dùng nội bộ (đã đọc source
      code xác nhận, không suy đoán).
    - "simcc": thay bằng `SimCCHead` — QUAN TRỌNG: `forward()` trả về TUPLE
      `(x_logits, y_logits)` thay vì 1 tensor heatmap, khác 2 head còn lại.
      Code gọi `PoseModel(...)` phải tự kiểm tra `head_type` để biết cách xử
      lý output (xem `train.py`, `eval/coco_eval.py`).

    `use_presence` (TUỲ CHỌN, mặc định False — docs_3/strategy.md mục 7.7):
    cộng thêm `PresenceHead` — khi bật, `forward()` trả về
    `(main_output, presence_logits)` thay vì chỉ `main_output` (main_output
    vẫn đúng y hệt dạng của `head_type` đã chọn — tensor hoặc tuple cho
    "simcc"). Khi tắt (mặc định), `forward()` GIỐNG HỆT TUYỆT ĐỐI hành vi
    trước khi thêm tham số này (đã verify)."""

    def __init__(self, backbone_name: str = "usyd-community/vitpose-plus-small", pretrained: bool = True,
                 head_type: str = "default", simcc_split_ratio: int = 2, use_presence: bool = False,
                 input_size: tuple[int, int] | None = None, scale_cond: bool = False,
                 dataset_experts: bool = False, expert_index: int = COCO_DATASET_INDEX):
        super().__init__()
        from transformers import VitPoseConfig, VitPoseForPoseEstimation
        native = VitPoseConfig.from_pretrained(backbone_name)
        if input_size is not None and tuple(input_size) != tuple(native.backbone_config.image_size):
            # Độ phân giải khác bản pretrain (vd 384x288): dựng model với image_size mới, nạp weight pretrain,
            # nội suy bicubic position embedding (lưới patch 16x12 -> 24x18). ViTPose gốc cũng làm vậy khi tăng độ phân giải.
            import copy
            cfg = copy.deepcopy(native)
            cfg.backbone_config.image_size = list(input_size)
            self.model = VitPoseForPoseEstimation(cfg)
            if pretrained:
                src = VitPoseForPoseEstimation.from_pretrained(backbone_name).state_dict()
                key = [k for k in src if k.endswith("position_embeddings")][0]
                pe = src[key]                                  # (1, 1+Hp*Wp, C)
                ph, pw = (native.backbone_config.image_size[0] // native.backbone_config.patch_size[0],
                          native.backbone_config.image_size[1] // native.backbone_config.patch_size[1])
                nh, nw = input_size[0] // cfg.backbone_config.patch_size[0], input_size[1] // cfg.backbone_config.patch_size[1]
                grid = pe[:, 1:].reshape(1, ph, pw, -1).permute(0, 3, 1, 2)
                grid = torch.nn.functional.interpolate(grid, size=(nh, nw), mode="bicubic", align_corners=False)
                src[key] = torch.cat([pe[:, :1], grid.permute(0, 2, 3, 1).reshape(1, nh * nw, -1)], dim=1)
                missing = self.model.load_state_dict(src, strict=True)
        elif pretrained:
            self.model = VitPoseForPoseEstimation.from_pretrained(backbone_name)
        else:
            self.model = VitPoseForPoseEstimation(native)
        if expert_index != COCO_DATASET_INDEX:
            # Luật Track 3: chỉ được pretrain bằng COCO/PoseTrack18. Expert 1.. của ViTPose+ là trọng số huấn luyện trên AIC/MPII/... => KHÔNG dùng.
            raise ValueError("expert_index != 0 (COCO) bị chặn vì vi phạm luật dữ liệu pretrain (docs_3/terms.md mục 3)")
        self.register_buffer("dataset_index", torch.tensor([expert_index], dtype=torch.int64))

        assert head_type in ("default", "fpn", "simcc"), head_type
        self.head_type = head_type
        self.simcc_split_ratio = simcc_split_ratio
        self.use_presence = use_presence
        bcfg = self.model.config.backbone_config
        self._patch_h = bcfg.image_size[0] // bcfg.patch_size[0]
        self._patch_w = bcfg.image_size[1] // bcfg.patch_size[1]
        if head_type == "fpn":
            self.fpn_head = FPNHead(in_channels=bcfg.hidden_size, num_keypoints=self.model.config.num_labels)
        elif head_type == "simcc":
            scale_factor = int(self.model.config.scale_factor)
            heatmap_h = bcfg.image_size[0] // scale_factor
            heatmap_w = bcfg.image_size[1] // scale_factor
            self.simcc_head = SimCCHead(in_channels=bcfg.hidden_size, num_keypoints=self.model.config.num_labels,
                                         patch_h=self._patch_h, patch_w=self._patch_w,
                                         heatmap_h=heatmap_h, heatmap_w=heatmap_w, split_ratio=simcc_split_ratio)
        self.dataset_experts = dataset_experts
        if dataset_experts:   # nhân bản expert 0 (COCO) sang expert 1,2: bắt đầu giống hệt, rồi phân hoá theo từng dataset nguồn
            import copy as _copy
            for layer in self.model.backbone.encoder.layer:
                for k in (1, 2):
                    layer.mlp.experts[k].load_state_dict(layer.mlp.experts[0].state_dict())
        self.scale_cond = scale_cond
        if scale_cond:
            assert head_type == "default" and not use_presence, "scale_cond chỉ hỗ trợ head mặc định"
            c = bcfg.hidden_size
            # FiLM theo log-kích-thước box gốc: feature' = feature * (1 + gamma(s)) + beta(s); lớp cuối khởi tạo 0 => bắt đầu = identity
            self.film = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2 * c))
            nn.init.zeros_(self.film[2].weight)
            nn.init.zeros_(self.film[2].bias)
        if use_presence:
            self.presence_head = PresenceHead(in_channels=bcfg.hidden_size, num_keypoints=self.model.config.num_labels)

    def _backbone_seq(self, pixel_values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Feature map cuối backbone, reshape (B,C,Hp,Wp) — dùng chung cho
        fpn/simcc/presence. Giống hệt cách VitPoseForPoseEstimation.forward()
        thật tính nội bộ (đã đọc source xác nhận)."""
        outputs = self.model.backbone.forward_with_filtered_kwargs(pixel_values, dataset_index=index)
        seq = outputs.feature_maps[-1]
        b = seq.shape[0]
        return seq.permute(0, 2, 1).reshape(b, -1, self._patch_h, self._patch_w).contiguous()

    def forward(self, pixel_values: torch.Tensor, scale_feat: torch.Tensor | None = None,
                dataset_index: torch.Tensor | None = None):
        index = self.dataset_index.expand(pixel_values.shape[0]) if dataset_index is None else dataset_index.to(torch.int64)
        if self.scale_cond:
            seq = self._backbone_seq(pixel_values, index)
            g, be = self.film(scale_feat.reshape(-1, 1).to(seq.dtype)).chunk(2, dim=-1)
            seq = seq * (1.0 + g[:, :, None, None]) + be[:, :, None, None]
            return self.model.head(seq)
        if self.head_type == "default":
            main_output = self.model(pixel_values=pixel_values, dataset_index=index).heatmaps
            if not self.use_presence:
                return main_output
            # use_presence=True + head_type=default: self.model(...) ở trên đã
            # tự tính backbone nội bộ (black-box), KHÔNG lấy lại được seq từ đó
            # — đành forward backbone THÊM 1 lần riêng cho presence (tốn thêm
            # compute, chấp nhận được vì đây là tổ hợp tuỳ chọn/thử nghiệm,
            # KHÔNG đụng tới path "default" đơn thuần (không presence) vốn đã
            # test kỹ — xem code_3/README.md).
            seq = self._backbone_seq(pixel_values, index)
            return main_output, self.presence_head(seq)

        # "fpn"/"simcc": đã cần seq cho head chính, presence tái dùng LUÔN,
        # không tốn thêm backbone forward.
        seq = self._backbone_seq(pixel_values, index)
        main_output = self.fpn_head(seq) if self.head_type == "fpn" else self.simcc_head(seq)
        if not self.use_presence:
            return main_output
        return main_output, self.presence_head(seq)


def get_preprocess_config(backbone_name: str = "usyd-community/vitpose-plus-small",
                          input_size: tuple[int, int] | None = None, padding_factor: float = 1.25) -> dict:
    """Mean/std/input size/scale_factor — lấy từ VitPoseImageProcessor + config
    thật (đã verify: image_size=[256,192], scale_factor=4 cho -small), KHÔNG
    hardcode để tự động đúng nếu đổi sang -base/-large."""
    if str(backbone_name).startswith("hrnet"):
        h, w = (int(input_size[0]), int(input_size[1])) if input_size is not None else (384, 288)
        return {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225], "rescale_factor": 1.0 / 255.0,
                "normalize_factor": 200.0, "padding_factor": float(padding_factor), "input_height": h, "input_width": w,
                "scale_factor": 4, "heatmap_height": h // 4, "heatmap_width": w // 4, "num_keypoints": 17}
    from transformers import VitPoseConfig, VitPoseImageProcessor
    processor = VitPoseImageProcessor.from_pretrained(backbone_name)
    cfg = VitPoseConfig.from_pretrained(backbone_name)
    h, w = int(processor.size["height"]), int(processor.size["width"])
    if input_size is not None:
        h, w = int(input_size[0]), int(input_size[1])
    return {
        "image_mean": list(processor.image_mean),
        "image_std": list(processor.image_std),
        "rescale_factor": float(processor.rescale_factor),
        "normalize_factor": float(getattr(processor, "normalize_factor", 200.0)),
        "padding_factor": float(padding_factor),
        "input_height": h,
        "input_width": w,
        "scale_factor": int(cfg.scale_factor),
        "heatmap_height": h // int(cfg.scale_factor),
        "heatmap_width": w // int(cfg.scale_factor),
        "num_keypoints": int(cfg.num_labels),
    }


HRNET_W48_PT18_CKPT = "pretrained/hrnet_w48_posetrack18_clean.pth"   # ImageNet-1K -> COCO -> PoseTrack18 (MMPose, load_from=COCO ckpt)
HRNET_W48_CKPT = "pretrained/hrnet_w48_coco_384x288.pth"
HRNET_W32_PT18_CKPT = "pretrained/hrnet_w32_posetrack18_clean.pth"   # ImageNet-1K -> COCO(256x192) -> PoseTrack18 (MMPose hrnet_w32_posetrack18_256x192-1ee951c4, load_from=COCO ckpt)


def build_model(backbone_name: str = "usyd-community/vitpose-plus-small", pretrained: bool = True,
                 head_type: str = "default", simcc_split_ratio: int = 2, use_presence: bool = False,
                 input_size: tuple[int, int] | None = None, scale_cond: bool = False,
                 dataset_experts: bool = False, expert_index: int = COCO_DATASET_INDEX) -> PoseModel:
    if str(backbone_name).startswith("hrnet"):   # HRNet-W48 (MMPose, COCO-only) — xem models/hrnet.py
        from models.hrnet import HRNetPoseModel
        if str(backbone_name) == "hrnet_w32_pt18":
            return HRNetPoseModel(HRNET_W32_PT18_CKPT if pretrained else None, width=32)
        return HRNetPoseModel((HRNET_W48_PT18_CKPT if str(backbone_name) == "hrnet_w48_pt18" else HRNET_W48_CKPT) if pretrained else None)
    return PoseModel(backbone_name=backbone_name, pretrained=pretrained, head_type=head_type,
                      simcc_split_ratio=simcc_split_ratio, use_presence=use_presence, input_size=input_size,
                      scale_cond=scale_cond, dataset_experts=dataset_experts, expert_index=expert_index)


def enable_drop_path(model, max_rate: float) -> None:
    """Stochastic depth cho backbone ViT của HF (HF VitPose KHÔNG cài đặt; ViTPose gốc dùng drop_path_rate 0.3-0.5 chống quá khớp).
    Tỷ lệ tăng tuyến tính 0 -> max_rate theo tầng; chỉ có tác dụng khi model.train(). Mặc định KHÔNG gọi (hành vi cũ)."""
    import types
    layers = model.model.backbone.encoder.layer
    n = len(layers)

    def _dp(x, p, training):
        if not training or p <= 0.0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= p).to(x.dtype) / (1.0 - p)
        return x * keep

    def fwd(self, hidden_states, dataset_index=None, head_mask=None):
        a = self.attention(self.layernorm_before(hidden_states), head_mask)
        hidden_states = hidden_states + _dp(a, self.drop_prob, self.training)
        o = self.layernorm_after(hidden_states)
        o = self.mlp(o) if self.num_experts == 1 else self.mlp(o, indices=dataset_index)
        return hidden_states + _dp(o, self.drop_prob, self.training)

    for i, layer in enumerate(layers):
        layer.drop_prob = float(max_rate) * i / max(n - 1, 1)
        layer.forward = types.MethodType(fwd, layer)


def soup_experts(model: "PoseModel", used: list[int]) -> "PoseModel":
    """Bản sao model với expert 0 = TRUNG BÌNH trọng số các expert đã huấn luyện (`used`) ở mọi tầng FFN.
    Dùng lúc eval trên dataset CHƯA THẤY (không có expert riêng) — suy luận 1 lượt, không tốn thêm chi phí."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        for layer in m.model.backbone.encoder.layer:
            ex = layer.mlp.experts
            for pname in ("weight", "bias"):
                avg = sum(getattr(ex[i], pname).detach() for i in used) / len(used)
                getattr(ex[0], pname).copy_(avg)
    return m
