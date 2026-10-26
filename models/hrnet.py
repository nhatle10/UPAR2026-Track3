"""HRNet-W48 top-down (Sun et al., CVPR 2019), cài lại với ĐÚNG tên tham số của checkpoint MMPose
(hrnet_w48_coco_384x288, huấn luyện CHỈ trên COCO, khởi tạo ImageNet-1K — ghi trong meta.config của file) để nạp strict.
Đối tác ghép đa dạng (CNN) cho ViTPose-L; hợp lệ theo email BTC: ImageNet-1K + COCO."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True); self.downsample = downsample

    def forward(self, x):
        r = x if self.downsample is None else self.downsample(x)
        o = self.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return self.relu(o + r)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False); self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True); self.downsample = downsample

    def forward(self, x):
        r = x if self.downsample is None else self.downsample(x)
        o = self.relu(self.bn1(self.conv1(x)))
        o = self.relu(self.bn2(self.conv2(o)))
        o = self.bn3(self.conv3(o))
        return self.relu(o + r)


class HRModule(nn.Module):
    def __init__(self, nb, num_blocks, in_ch, ch, multi_scale_output=True):
        super().__init__()
        self.nb, self.in_ch, self.multi = nb, list(in_ch), multi_scale_output
        self.branches = nn.ModuleList([nn.Sequential(*[BasicBlock(self.in_ch[b] if k == 0 else ch[b], ch[b]) for k in range(num_blocks[b])]) for b in range(nb)])
        fl = []
        for i in range(nb if multi_scale_output else 1):
            row = []
            for j in range(nb):
                if j > i:
                    row.append(nn.Sequential(nn.Conv2d(self.in_ch[j], self.in_ch[i], 1, bias=False), nn.BatchNorm2d(self.in_ch[i])))
                elif j == i:
                    row.append(None)
                else:
                    ds = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            ds.append(nn.Sequential(nn.Conv2d(self.in_ch[j], self.in_ch[i], 3, 2, 1, bias=False), nn.BatchNorm2d(self.in_ch[i])))
                        else:
                            ds.append(nn.Sequential(nn.Conv2d(self.in_ch[j], self.in_ch[j], 3, 2, 1, bias=False), nn.BatchNorm2d(self.in_ch[j]), nn.ReLU(inplace=True)))
                    row.append(nn.Sequential(*ds))
            fl.append(nn.ModuleList(row))
        self.fuse_layers = nn.ModuleList(fl)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = [self.branches[i](x[i]) for i in range(self.nb)]
        out = []
        for i, fl in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fl[0](x[0])
            for j in range(1, self.nb):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(fl[j](x[j]), size=x[i].shape[-2:], mode="nearest")
                else:
                    y = y + fl[j](x[j])
            out.append(self.relu(y))
        return out


class HRNetBackbone(nn.Module):
    CH = (48, 96, 192, 384)

    def __init__(self, ch=None):
        super().__init__()
        c = tuple(ch) if ch else self.CH
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False); self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        ds = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.layer1 = nn.Sequential(Bottleneck(64, 64, 1, ds), *[Bottleneck(256, 64) for _ in range(3)])
        self.transition1 = self._transition([256], list(c[:2]))
        self.stage2 = nn.Sequential(HRModule(2, (4, 4), c[:2], c[:2]))
        self.transition2 = self._transition(list(c[:2]), list(c[:3]))
        self.stage3 = nn.Sequential(*[HRModule(3, (4,) * 3, c[:3], c[:3]) for _ in range(4)])
        self.transition3 = self._transition(list(c[:3]), list(c[:4]))
        self.stage4 = nn.Sequential(*[HRModule(4, (4,) * 4, c[:4], c[:4], multi_scale_output=(m < 2)) for m in range(3)])

    @staticmethod
    def _transition(pre, cur):
        out = []
        for i in range(len(cur)):
            if i < len(pre):
                out.append(nn.Sequential(nn.Conv2d(pre[i], cur[i], 3, 1, 1, bias=False), nn.BatchNorm2d(cur[i]), nn.ReLU(inplace=True)) if cur[i] != pre[i] else None)
            else:
                cs = []
                for j in range(i + 1 - len(pre)):
                    o = cur[i] if j == i - len(pre) else pre[-1]
                    cs.append(nn.Sequential(nn.Conv2d(pre[-1], o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)))
                out.append(nn.Sequential(*cs))
        return nn.ModuleList(out)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))); x = self.relu(self.bn2(self.conv2(x))); x = self.layer1(x)
        xl = [t(x) if t is not None else x for t in self.transition1]
        y = self.stage2(xl)
        xl = [self.transition2[i](y[-1]) if self.transition2[i] is not None else y[i] for i in range(3)]
        y = self.stage3(xl)
        xl = [self.transition3[i](y[-1]) if self.transition3[i] is not None else y[i] for i in range(4)]
        y = self.stage4(xl)
        return y[0]


class _Inner(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.backbone = HRNetBackbone((width, 2 * width, 4 * width, 8 * width))
        self.head = nn.Conv2d(width, 17, 1)


class HRNetPoseModel(nn.Module):
    """Giao diện tương thích PoseModel của train.py/eval/export: forward(x)->heatmap (B,17,H/4,W/4); model.backbone/model.head cho param_groups."""
    head_type = "default"; scale_cond = False; dataset_experts = False; use_presence = False; simcc_split_ratio = 2

    def __init__(self, ckpt: str | None = None, width: int = 48):
        super().__init__()
        self.model = _Inner(width)
        if ckpt:
            self.load_mmpose(ckpt)

    def load_mmpose(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        new = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                new["model." + k] = v
            elif k == "keypoint_head.final_layer.weight":
                new["model.head.weight"] = v
            elif k == "keypoint_head.final_layer.bias":
                new["model.head.bias"] = v
            else:
                raise KeyError(k)
        self.load_state_dict(new, strict=True)

    def forward(self, x, scale_feat=None, dataset_index=None):
        return self.model.head(self.model.backbone(x))
