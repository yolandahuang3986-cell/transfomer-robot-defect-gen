"""How many 32x32 latent cells do the REAL support masks cover?  (train-split support only)
Same resize as generation: 256x256 (NEAREST, as the area window) -> transforms.Resize((32,32)) -> >=0.5.
Gives a data-derived floor for 'renderable' masks per defect type, and shows which types sit
near the latent-resolution limit. Runs in 'Anomalydiffusion' env, CPU, a few seconds.
Usage: python support_latent_cells.py [support_list.json]"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

lst = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/external/diffusion_support/support_core_n5_seed42.json")
root = os.path.expanduser("~/datasets/mvtec_ad")
resize32 = transforms.Resize((32, 32))
print(f"{'group':28s} {'px@256 per mask':32s} {'latent cells per mask':28s} min")
for group, recs in sorted(json.load(open(lst))["groups"].items()):
    px, cells = [], []
    for r in recs:
        m = (np.asarray(Image.open(os.path.join(root, r["mask"])).convert("L")
                        .resize((256, 256), Image.NEAREST)) > 127).astype(np.float32)
        px.append(int(m.sum()))
        cells.append(int((resize32(torch.from_numpy(m)[None, None]) >= 0.5).sum()))
    print(f"{group:28s} {str(px):32s} {str(cells):28s} {min(cells)}")
