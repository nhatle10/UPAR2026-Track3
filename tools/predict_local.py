"""Run a submission package locally exactly as the evaluation server does (import run.py, call
predict_image once per image) on a COCO-style annotation file with person boxes, then report OKS AP.
usage: python tools/predict_local.py SUBMISSION_DIR ANNOTATIONS.json IMAGE_ROOT [--limit N] [--out preds.json]
ANNOTATIONS.json: COCO keypoint format (images, annotations with bbox [and keypoints for scoring])."""
import argparse, importlib.util, json, os, sys, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("submission_dir"); ap.add_argument("annotations"); ap.add_argument("image_root")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="predictions.json")
a = ap.parse_args()
a.submission_dir, a.annotations, a.image_root, a.out = (os.path.abspath(x) for x in (a.submission_dir, a.annotations, a.image_root, a.out))

spec = importlib.util.spec_from_file_location("run", os.path.join(a.submission_dir, "run.py"))
run = importlib.util.module_from_spec(spec); sys.path.insert(0, a.submission_dir); os.chdir(a.submission_dir)
spec.loader.exec_module(run)

gt = json.load(open(a.annotations))
anns = defaultdict(list)
for x in gt["annotations"]:
    anns[x["image_id"]].append(x)
images = [im for im in gt["images"] if im["id"] in anns]
if a.limit:
    images = images[: a.limit]
dets, t0 = [], time.time()
for im in images:
    boxes = [x["bbox"] for x in anns[im["id"]]]
    sample = {"image_id": im["id"], "file_name": im["file_name"], "image_path": os.path.join(a.image_root, im["file_name"]),
              "width": im.get("width"), "height": im.get("height"), "boxes": boxes, "box_ids": [x["id"] for x in anns[im["id"]]]}
    out = run.predict_image(sample)
    kps, scs = (out["keypoints"], out["scores"]) if isinstance(out, dict) else (out, [1.0] * len(out))
    for k, s in zip(kps, scs):
        flat = [float(v) for p in k for v in (p if isinstance(p, (list, tuple)) else [p])]
        dets.append({"image_id": im["id"], "category_id": 1, "keypoints": flat, "score": float(s)})
print(f"{len(images)} images in {time.time() - t0:.1f}s")
json.dump(dets, open(a.out, "w"))
if any("keypoints" in x for x in gt["annotations"]):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    sub = dict(gt); sub["images"] = images; ids = {im["id"] for im in images}
    sub["annotations"] = [x for x in gt["annotations"] if x["image_id"] in ids]
    sub.setdefault("categories", [{"id": 1, "name": "person"}])
    G = COCO(); G.dataset = sub; G.createIndex(); D = G.loadRes(dets)
    E = COCOeval(G, D, "keypoints")
    E.params.kpt_oks_sigmas = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
    E.params.maxDets = [20]; E.evaluate(); E.accumulate(); E.summarize()
