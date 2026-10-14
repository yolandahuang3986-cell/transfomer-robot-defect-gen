"""Run the SHARED project QC on one worker output dir and report it separately from the
Diffusion-internal stage-1 statistics. Runs in 'defectgen' from the repo root (PYTHONPATH=$PWD).
Usage: python qc_check_worker_output.py <worker out_dir>"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))  # repo root
from src.generation.base import GeneratedBatch, validate_batch
from src.generation.qc import DEFAULT_QC_POLICY, estimate_foregrounds, filter_generated_batch

out = sys.argv[1]
req = json.load(open(os.path.join(out, "request.json")))
res = json.load(open(os.path.join(out, "result.json")))
root = req["mvtec_root"]
by_index = {int(r["index"]): r for r in res["samples"]}
req["samples"] = [s for s in req["samples"] if int(s["index"]) in by_index]   # underfilled runs: produced only

imgs, msks, meta = [], [], []
for s in req["samples"]:
    i = int(s["index"])
    imgs.append(np.moveaxis(np.asarray(Image.open(os.path.join(out, "images", f"{i:05d}.png")).convert("RGB"),
                                       dtype=np.float32) / 255.0, -1, 0))
    msks.append((np.asarray(Image.open(os.path.join(out, "masks", f"{i:05d}.png")).convert("L")) > 127)
                .astype(np.float32)[None])
    meta.append({"method": "diffusion", "category": req["category"], "defect_type": req["defect_type"],
                 "source_image_id": s["source_image_id"], "seed": int(req["seed"]),
                 "generation_config": {"placement": req.get("placement", "v0"), **req["sampling"]},
                 "generation_seconds": float(by_index[i]["generation_seconds"])})
batch = validate_batch(GeneratedBatch(np.stack(imgs), np.stack(msks), meta), len(meta))

ids = sorted({m["source_image_id"] for m in meta})
normals = np.stack([np.moveaxis(np.asarray(Image.open(os.path.join(root, sid)).convert("RGB"),
                                           dtype=np.float32) / 255.0, -1, 0) for sid in ids])
fgs = estimate_foregrounds(normals, ids, DEFAULT_QC_POLICY)
report = filter_generated_batch(batch, fgs, policy=DEFAULT_QC_POLICY).report.to_dict()
json.dump(report, open(os.path.join(out, "shared_qc_report.json"), "w"), indent=2)

print("=== Diffusion-internal (stage 1, not part of shared QC) ===")
print(" status:", res.get("status", "ok"), f"{res.get('n')}/{res.get('requested', res.get('n'))}",
      "| placement:", req.get("placement", "v0"), "| mask attempts:", res["mask_attempts"],
      "| upstream check_mask rejected:", res.get("stage1_upstream_check_mask_rejected"),
      "| internal filter:", res.get("diffusion_internal_filter"))
print(f"=== Shared QC ({report['policy_version']}) ===")
print(f" accepted {report['accepted_count']}/{report['attempted_count']} "
      f"({report['acceptance_rate']:.0%}) | rejections {report['rejection_counts']}")
for r in report["rejected_samples"]:
    print("  rejected:", r)

# ---- analysis only (not a filter, not part of QC): is the defect visible? ----
# inside  = mean |generated - source| inside the mask (0-255)
# outside = same in a 6-px ring around the mask: the VAE/inpainting baseline change
from scipy.ndimage import binary_dilation
print("=== Analysis: defect visibility (reporting only) ===")
vis = []
for s, im, mk in zip(req["samples"], imgs, msks):
    src = np.asarray(Image.open(os.path.join(root, s["source_image_id"])).convert("RGB")
                     .resize(im.shape[1:][::-1], Image.BICUBIC), np.float32)
    gen = np.moveaxis(im, 0, -1) * 255.0
    d = np.abs(gen - src).mean(-1)
    m = mk[0] > 0.5
    ring = binary_dilation(m, iterations=6) & ~m
    inside, outside = float(d[m].mean()), float(d[ring].mean())
    vis.append({"index": int(s["index"]), "inside": round(inside, 1), "ring": round(outside, 1),
                "ratio": round(inside / max(outside, 1e-6), 2), "mask_px": int(m.sum())})
    print(" ", vis[-1])
json.dump(vis, open(os.path.join(out, "visibility.json"), "w"), indent=2)
