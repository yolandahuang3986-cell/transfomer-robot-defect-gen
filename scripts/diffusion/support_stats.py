#!/usr/bin/env python3
"""Support-mask statistics for the diffusion-internal mask filter.

Runs in the AnomalyDiffusion environment on purpose: the latent-cell count must
use the same torchvision Resize as upstream inference (ddpm.log_images resizes
the 256 px mask to 32x32 and keeps cells >= 0.5).

Input  (--request JSON): {"root": "...", "masks": ["cat/ground_truth/<type>/<id>_mask.png", ...]}
Output (stdout, one JSON line): {"area_pct_256": [...], "latent_cells": [...], "torchvision": "..."}
Masks are first resized to 256 x 256 with NEAREST, as in mask_filter_v1/v2.
"""
import argparse
import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    req = json.load(open(parser.parse_args().request))
    resize32 = transforms.Resize((32, 32))
    areas, cells = [], []
    for rel in req["masks"]:
        m = np.asarray(Image.open(os.path.join(req["root"], rel)).convert("L")
                       .resize((256, 256), Image.NEAREST)) > 127
        areas.append(round(float(m.mean()) * 100, 4))
        cells.append(int((resize32(torch.from_numpy(m.astype(np.float32))[None, None]) >= 0.5).sum()))
    print(json.dumps({"area_pct_256": areas, "latent_cells": cells,
                      "torchvision": torchvision.__version__}), flush=True)


if __name__ == "__main__":
    main()
