"""Where did the generator actually change pixels, relative to the saved mask?
The saved mask is pixel-level, but DDIM edits whole 32x32-latent cells (8x8 px each, Resize + >=0.5).
Regions per sample (256x256):
  mask      = saved pixel mask (the label downstream will see)
  footprint = latent cells that were edited, upsampled x8 (nearest)
  far ring  = pixels 8..24 px outside the footprint (should only carry VAE noise)
Prints mean |gen - src| per region and a footprint-based ratio, next to the human labels.
Runs in 'Anomalydiffusion' env (same torchvision Resize as generation). CPU, seconds.
Usage: python edit_region_analysis.py <worker out_dir> <labels.json>"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from torchvision import transforms

out, lab = sys.argv[1], json.load(open(sys.argv[2]))
req = json.load(open(os.path.join(out, "request.json")))
res = json.load(open(os.path.join(out, "result.json")))
done = {int(r["index"]) for r in res["samples"]}
vis = {v["index"]: v["ratio"] for v in json.load(open(os.path.join(out, "visibility.json")))}
A, B = lab["annotator_yuchen"], lab["annotator_claude_not_blind"]
r32 = transforms.Resize((32, 32))
rows = []
print(f"{'#':>3} {'Y':>2} {'C':>2} {'old':>5} | {'mask':>5} {'fp-mask':>7} {'far':>5} | {'new':>5} | fp_px mask_px")
for s in req["samples"]:
    i = int(s["index"])
    if i not in done:
        continue
    src = np.asarray(Image.open(os.path.join(req["mvtec_root"], s["source_image_id"])).convert("RGB")
                     .resize((256, 256), Image.BICUBIC), np.float32)
    gen = np.asarray(Image.open(os.path.join(out, "images", f"{i:05d}.png")).convert("RGB"), np.float32)
    d = np.abs(gen - src).mean(-1)
    m = np.asarray(Image.open(os.path.join(out, "masks", f"{i:05d}.png")).convert("L")) > 127
    lat = (r32(torch.from_numpy(m.astype(np.float32))[None, None]) >= 0.5)[0, 0].numpy()
    fp = np.kron(lat, np.ones((8, 8), bool)).astype(bool)
    far = binary_dilation(fp, iterations=24) & ~binary_dilation(fp, iterations=8)
    dm, dfm, dfar = d[m].mean(), d[fp & ~m].mean() if (fp & ~m).any() else float("nan"), d[far].mean()
    new = d[fp].mean() / max(dfar, 1e-6)
    rows.append((i, A[str(i)], B[str(i)], vis[i], dm, dfm, dfar, new, int(fp.sum()), int(m.sum())))
    print(f"{i:>3} {A[str(i)]:>2} {B[str(i)]:>2} {vis[i]:>5.2f} | {dm:5.1f} {dfm:7.1f} {dfar:5.1f} | {new:5.2f} | {fp.sum():5d} {m.sum():5d}")

both_v = [r for r in rows if r[1] == "V" and r[2] == "V"]
not_v = [r for r in rows if r[1] != "V" and r[2] != "V"]
print(f"\nboth say V: {len(both_v)}  | neither says V: {len(not_v)}  | disagree: {len(rows) - len(both_v) - len(not_v)}")
for name, k in (("old ratio", 3), ("new ratio", 7)):
    v = [r[k] for r in both_v]; n = [r[k] for r in not_v]
    pairs = [(a, b) for a in v for b in n]
    auc = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / max(1, len(pairs))
    print(f"{name}: V mean {np.mean(v):.2f} [{min(v):.2f}-{max(v):.2f}]  not-V mean {np.mean(n):.2f} "
          f"[{min(n):.2f}-{max(n):.2f}]  pairwise AUC {auc:.2f}")
