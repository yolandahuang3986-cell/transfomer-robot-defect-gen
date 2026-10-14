"""Re-run shared QC on a worker output dir the way A's formal loader does
(src/inspection/formal_bundle.py: source RGB BILINEAR-resized to 256, foreground estimated ON that grid),
and compare with the foreground our placement used (estimated at native resolution, then nearest-resized).
Runs in 'defectgen' from the repo root on main (PYTHONPATH=$PWD). CPU, seconds.
Usage: python formal_qc_recheck.py <worker out_dir>"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))  # repo root
from src.generation.base import GeneratedBatch, validate_batch
from src.generation.qc import DEFAULT_QC_POLICY, estimate_object_foreground, filter_generated_batch

out = sys.argv[1]
req = json.load(open(os.path.join(out, "request.json")))
res = json.load(open(os.path.join(out, "result.json")))
by = {int(r["index"]): r for r in res["samples"]}
samples = [s for s in req["samples"] if int(s["index"]) in by]
root = req["mvtec_root"]

imgs, msks, meta, formal_fg = [], [], [], {}
print(f"{'#':>3} {'src':>8} | {'fg_formal%':>10} {'fg_place%':>9} {'place_in_formal%':>16} | {'ovl_formal':>10} {'ovl_place':>9}")
for s in samples:
    i, sid = int(s["index"]), s["source_image_id"]
    im = np.asarray(Image.open(os.path.join(out, "images", f"{i:05d}.png")).convert("RGB"), np.float32) / 255
    m = np.asarray(Image.open(os.path.join(out, "masks", f"{i:05d}.png")).convert("L")) > 127
    imgs.append(np.moveaxis(im, -1, 0)); msks.append(m.astype(np.float32)[None])
    meta.append({"method": "diffusion", "category": req["category"], "defect_type": req["defect_type"],
                 "source_image_id": sid, "seed": int(req["seed"]),
                 "generation_config": {"placement": req.get("placement")},
                 "generation_seconds": float(by[i]["generation_seconds"])})
    if sid not in formal_fg:                                    # exactly as formal_bundle.load_formal_pool
        with Image.open(os.path.join(root, sid)) as src:
            a = np.asarray(src.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), np.float32)
        formal_fg[sid] = estimate_object_foreground(a.transpose(2, 0, 1) / 255)
    ff = formal_fg[sid]
    pf = np.asarray(Image.open(s["foreground_png"]).convert("L")) > 127 if "foreground_png" in s else None
    row = f"{i:>3} {os.path.basename(sid):>8} | {ff.mean()*100:10.1f} "
    row += (f"{pf.mean()*100:9.1f} {(pf & ff).sum() / max(1, pf.sum())*100:16.1f} | " if pf is not None else f"{'-':>9} {'-':>16} | ")
    row += f"{(m & ff).sum() / m.sum():10.3f} " + (f"{(m & pf).sum() / m.sum():9.3f}" if pf is not None else "")
    print(row)
batch = validate_batch(GeneratedBatch(np.stack(imgs), np.stack(msks), meta), len(meta))
rep = filter_generated_batch(batch, formal_fg, policy=DEFAULT_QC_POLICY).report.to_dict()
print(f"\nshared QC with the FORMAL foreground convention: accepted {rep['accepted_count']}/{rep['attempted_count']} "
      f"| rejections {rep['rejection_counts']}")
for r in rep["rejected_samples"]:
    print("  rejected:", r)
json.dump(rep, open(os.path.join(out, "shared_qc_report_formal_convention.json"), "w"), indent=2)
