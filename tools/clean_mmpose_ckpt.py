"""Strip the MMPose metadata (numpy objects) from an official MMPose checkpoint so that it loads with
torch.load(weights_only=True). Only the state_dict is kept.
usage: python tools/clean_mmpose_ckpt.py IN.pth OUT.pth"""
import sys, torch

src, dst = sys.argv[1:3]
d = torch.load(src, map_location="cpu", weights_only=False)   # official file from download.openmmlab.com
meta = d.get("meta", {})
torch.save({"state_dict": d["state_dict"], "meta_config": str(meta.get("config", "")),
            "epoch": meta.get("epoch"), "time": meta.get("time"), "mmpose_version": meta.get("mmpose_version")}, dst)
print("saved", dst)
