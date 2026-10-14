"""Hypothesis test: are 'invisible' defects the ones whose mask vanishes on the 32x32 latent grid?
Upstream log_images does: mask -> transforms.Resize((32,32)) -> >=0.5 -> latent inpainting mask.
If that latent mask is empty, DDIM keeps the original latent everywhere -> no defect is drawn.
Runs in the 'Anomalydiffusion' env (same torchvision as generation). CPU only, a few seconds.
Usage: python latent_mask_check.py <worker out_dir> [<worker out_dir> ...]"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

resize32 = transforms.Resize((32, 32))          # identical call to ddpm.log_images
for out in sys.argv[1:]:
    req = json.load(open(os.path.join(out, "request.json")))
    vis_path = os.path.join(out, "visibility.json")
    vis = {v["index"]: v for v in json.load(open(vis_path))} if os.path.exists(vis_path) else {}
    print(f"=== {out}  (placement {req.get('placement', 'v0')}) ===")
    rows = []
    for s in req["samples"]:
        i = int(s["index"])
        m = (np.asarray(Image.open(os.path.join(out, "masks", f"{i:05d}.png")).convert("L")) > 127).astype(np.float32)
        t = resize32(torch.from_numpy(m)[None, None])
        cells = int((t >= 0.5).sum())
        rows.append((i, int(m.sum()), cells, vis.get(i, {}).get("ratio")))
        print(f"  #{i}  mask_px={int(m.sum()):4d}  latent_cells={cells:3d}  visibility_ratio={rows[-1][3]}")
    empty = [r for r in rows if r[2] == 0]
    print(f"  -> {len(empty)}/{len(rows)} masks are EMPTY on the 32x32 latent grid")
