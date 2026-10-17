"""One row per sample: source(train/good) | mask | generated | mask outline on generated.
Usage: python make_contact_sheet.py <worker out_dir>     (light: a few PNGs, OK on the head node)"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

out = sys.argv[1]
req = json.load(open(os.path.join(out, "request.json")))
_res = os.path.join(out, "result.json")
if os.path.exists(_res):                       # underfilled runs: show only the samples that were produced
    _done = {int(r["index"]) for r in json.load(open(_res))["samples"]}
    req["samples"] = [s for s in req["samples"] if int(s["index"]) in _done]
S = 256
rows = []
for s in req["samples"]:
    i = int(s["index"])
    src = Image.open(os.path.join(req["mvtec_root"], s["source_image_id"])).convert("RGB").resize((S, S), Image.BICUBIC)
    m = Image.open(os.path.join(out, "masks", f"{i:05d}.png")).convert("L")
    gen = Image.open(os.path.join(out, "images", f"{i:05d}.png")).convert("RGB")
    mk = np.array(m) > 127
    ov = np.array(gen).copy()
    er = np.zeros_like(mk)                     # 1-px outline = mask minus its erosion
    er[1:-1, 1:-1] = mk[1:-1, 1:-1] & mk[:-2, 1:-1] & mk[2:, 1:-1] & mk[1:-1, :-2] & mk[1:-1, 2:]
    ov[mk & ~er] = [255, 0, 0]
    if "foreground_png" in s:                  # v1: green outline = shared-QC foreground
        fg = np.asarray(Image.open(s["foreground_png"]).convert("L")) > 127
        fe = np.zeros_like(fg)
        fe[1:-1, 1:-1] = fg[1:-1, 1:-1] & fg[:-2, 1:-1] & fg[2:, 1:-1] & fg[1:-1, :-2] & fg[1:-1, 2:]
        ov[fg & ~fe] = [0, 255, 0]
    row = Image.new("RGB", (4 * S, S))
    for k, im in enumerate([src, m.convert("RGB"), gen, Image.fromarray(ov)]):
        row.paste(im, (k * S, 0))
    ImageDraw.Draw(row).text((4, 4), f"#{i} {s['source_image_id'].split('/')[-1]}", fill=(255, 255, 0))
    rows.append(row)
sheet = Image.new("RGB", (4 * S, S * len(rows)))
for r, row in enumerate(rows):
    sheet.paste(row, (0, r * S))
p = os.path.join(out, "contact_sheet.png")
sheet.save(p)
print("->", p)
