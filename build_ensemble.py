"""Đóng gói zip GHÉP từ các zip thành viên đã có (dùng lại .torchscript, không export lại).
Dùng:  python build_ensemble.py --out ens.zip --member a.zip:1.0:1.1 --member b.zip:1.0:1.1
       (mỗi --member = <zip>:<trọng số>[:<padding_factor ghi đè>])"""
import argparse, json, zipfile
from pathlib import Path

HERE = Path(__file__).parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--fusion", default="coords", choices=["coords", "heatmap"], help="coords: trộn toạ độ đã giải mã; heatmap: trộn heatmap rồi giải mã 1 lần (cần cùng padding/kích thước)")
    ap.add_argument("--score_power", type=float, default=None, help="TUỲ CHỌN: điểm = trung bình bậc p của đỉnh heatmap trên --score_joints")
    ap.add_argument("--score_joints", default="0,5,6,7,8,9,10,11,12,13,14,15,16")
    a = ap.parse_args()
    members, files = [], {}
    for i, spec in enumerate(a.member):
        parts = spec.split(":")
        zpath, w = parts[0], float(parts[1]) if len(parts) > 1 else 1.0
        pad = float(parts[2]) if len(parts) > 2 and parts[2] else None
        fp16 = len(parts) > 3 and parts[3] == "fp16"   # TUỲ CHỌN: <zip>:<w>:<pad>:fp16
        with zipfile.ZipFile(zpath) as z:
            meta = json.loads(z.read("assets/preprocess.json"))
            if pad is not None:
                meta["padding_factor"] = pad
            assert meta.get("decode_fix") in (True, False)
            files[f"assets/m{i}_preprocess.json"] = json.dumps(meta, indent=2).encode()
            files[f"assets/m{i}.torchscript"] = z.read("assets/vitpose.torchscript")
            if i == 0:
                files["assets/mean_pose.json"] = z.read("assets/mean_pose.json")
                files["metadata.yaml"] = z.read("metadata.yaml")
        members.append({"model": f"assets/m{i}.torchscript", "meta": f"assets/m{i}_preprocess.json", "weight": w, **({"fp16": True} if fp16 else {})})
        print(f"member {i}: {zpath} w={w} input={meta['input_height']}x{meta['input_width']} pad={meta['padding_factor']} "
              f"tta={meta['tta_flip']} decode_fix={meta['decode_fix']} fp16={fp16}")
    spec = {"members": members, "fusion": a.fusion}
    if a.score_power:
        spec["score"] = {"joints": [int(x) for x in a.score_joints.split(",")], "power": a.score_power}
    files["assets/ensemble.json"] = json.dumps(spec, indent=2).encode()
    files["run.py"] = (HERE / "submission_template" / "run_ensemble.py").read_bytes()
    files["single_impl.py"] = (HERE / "submission_template" / "run.py").read_bytes()
    if a.out.exists():
        a.out.unlink()
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in files.items():
            zo.writestr(name, data)
    print(f"Đã tạo {a.out} ({a.out.stat().st_size/1e6:.0f} MB)")

if __name__ == "__main__":
    main()
