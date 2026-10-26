"""Uniform weight soup of several fine-tuned checkpoints.
usage: python tools/build_soup.py OUT.pt CKPT1.pt CKPT2.pt ...
Floating-point parameters are averaged; BatchNorm running statistics are copied from the first
checkpoint and must be re-estimated afterwards with recalibrate_bn.py."""
import sys, os, torch

out, ckpts = sys.argv[1], sys.argv[2:]
A = torch.load(ckpts[0], map_location="cpu")
acc = {k: A[k].float().clone() for k in A if A[k].is_floating_point() and "running_" not in k}
for p in ckpts[1:]:
    S = torch.load(p, map_location="cpu")
    for k in acc:
        acc[k] += S[k].float()
soup = {k: (acc[k] / len(ckpts)).to(A[k].dtype) if k in acc else A[k].clone() for k in A}
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
torch.save(soup, out)
print(f"saved {out} (uniform soup of {len(ckpts)} checkpoints)")
