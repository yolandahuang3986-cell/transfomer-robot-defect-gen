"""Diagnose stage 1 only: what does each mask embedding produce BEFORE thresholding, and how often is it usable?
Loads the mask model once; for every embedding samples PROBE_N masks (default 64) with the same seed.
Each sample is classified (rules below are descriptive, for the failure taxonomy - not a filter yet):
  empty     binary area < 0.005 %
  inverted  white touches the image border on > 1/4 of its length, or area > 20 %
            (the model drew a dark shape on a light background, so mean>0.8 selects the background)
  dirty_bg  raw median > -0.8: the background is not clean black (grey / gradient / textured)
  valid     none of the above
Writes probe.png (first 16 per embedding: raw row + binary row, category letter), probe.json.
Usage: PROBE_N=64 python mask_probe.py <out_dir> <support_list.json> <cat+defect> <label>=<emb.pt> ..."""
import json, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

UP = os.path.expanduser("~/external/anomalydiffusion")
os.chdir(UP); sys.path.insert(0, UP); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/diffusion
from ad_worker import load_model, setup_seed, check_mask, captions   # same code path as the worker
try:
    from scipy.ndimage import label as cc_label
except ImportError:
    cc_label = None

out, support_list, group = sys.argv[1], sys.argv[2], sys.argv[3]
items = [a.split("=", 1) for a in sys.argv[4:]]
os.makedirs(out, exist_ok=True)
N, B, S, SHOW = int(os.environ.get("PROBE_N", 64)), 8, 64, 16

# reference: support-mask areas at the same 256x256 resolution
root = os.path.expanduser("~/datasets/mvtec_ad")
ref = []
for r in json.load(open(support_list))["groups"][group]:
    m = np.array(Image.open(os.path.join(root, r["mask"])).convert("L").resize((256, 256), Image.NEAREST)) > 127
    ref.append(round(float(m.mean()) * 100, 3))
print("support mask area % at 256px:", ref)


def classify(g, b):
    area = float(b.mean())
    border = torch.cat([b[0], b[-1], b[:, 0], b[:, -1]]).mean().item()
    if area < 5e-5:
        return "empty"
    if border > 0.25 or area > 0.20:
        return "inverted"
    if float(g.median()) > -0.8:
        return "dirty_bg"
    return "valid"


model = load_model("configs/latent-diffusion/txt2img-1p4B-finetune.yaml",
                   f"{UP}/models/ldm/text2img-large/model.ckpt").cuda()
rows, report = [], {"group": group, "support_area_pct_256": ref, "n": N, "embeddings": {}}
for label, emb in items:
    model.embedding_manager.load(emb)
    setup_seed(42)
    gs = []
    for _ in range(0, N, B):
        batch = {"image": torch.zeros(B, 256, 256, 3), "caption": captions(B)}
        with torch.no_grad(), model.ema_scope():
            gs.append(model.log_images(batch, N=B, sample=True, inpaint=False, unconditional_only=True,
                                       ddim_steps=100)["samples_scaled"].cpu().mean(1))
    g = torch.cat(gs)[:N]
    b = (g > 0.8).float()
    per = []
    for k in range(N):
        cat = classify(g[k], b[k])
        ncomp = int(cc_label(b[k].numpy())[1]) if cc_label else -1
        per.append({"cat": cat, "area_pct": round(float(b[k].mean()) * 100, 3), "components": ncomp,
                    "check_mask": bool(check_mask(b[k][None]))})
    counts = {c: sum(p["cat"] == c for p in per) for c in ("valid", "empty", "inverted", "dirty_bg")}
    va = [p["area_pct"] for p in per if p["cat"] == "valid"]
    summary = {"counts": counts,
               "check_mask_pass": sum(p["check_mask"] for p in per),
               "check_mask_pass_but_not_valid": sum(p["check_mask"] and p["cat"] != "valid" for p in per),
               "valid_area_pct_median": round(float(np.median(va)), 3) if va else None,
               "valid_components_median": float(np.median([p["components"] for p in per if p["cat"] == "valid"])) if va else None}
    report["embeddings"][label] = {"path": emb, "summary": summary, "samples": per}
    print(label, summary)
    for t in (((g[:SHOW].clamp(-1, 1) + 1) / 2), b[:SHOW]):
        row = Image.new("L", (SHOW * S, S))
        for k in range(min(SHOW, N)):
            row.paste(Image.fromarray((t[k].numpy() * 255).astype(np.uint8)).resize((S, S)), (k * S, 0))
            ImageDraw.Draw(row).text((2, S - 11), per[k]["cat"][0].upper(), fill=128)
        ImageDraw.Draw(row).text((2, 1), label, fill=128)
        rows.append(row)
sheet = Image.new("L", (SHOW * S, S * len(rows)), 60)
for r, row in enumerate(rows):
    sheet.paste(row, (0, r * S))
sheet.save(os.path.join(out, "probe.png"))
json.dump(report, open(os.path.join(out, "probe.json"), "w"), indent=2)
print("->", out)
