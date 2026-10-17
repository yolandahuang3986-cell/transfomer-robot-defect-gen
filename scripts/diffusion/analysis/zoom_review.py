"""Zoomed side-by-side for HUMAN labelling of 'is the defect visible?'.
Per sample: 64x64 crop around the mask centre, x4 (nearest): source | generated | generated + red mask outline.
The visibility ratio is printed small under each row ONLY after you label - see --show_ratio.
Runs in any env with numpy + PIL (light, head node OK).
Usage: python zoom_review.py <worker out_dir> [--show_ratio]"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

out = sys.argv[1]
show = "--show_ratio" in sys.argv
req = json.load(open(os.path.join(out, "request.json")))
res = json.load(open(os.path.join(out, "result.json")))
done = {int(r["index"]) for r in res["samples"]}
vis = {}
if os.path.exists(os.path.join(out, "visibility.json")):
    vis = {v["index"]: v["ratio"] for v in json.load(open(os.path.join(out, "visibility.json")))}
C, Z = 64, 4
rows = []
for s in req["samples"]:
    i = int(s["index"])
    if i not in done:
        continue
    src = np.asarray(Image.open(os.path.join(req["mvtec_root"], s["source_image_id"])).convert("RGB")
                     .resize((256, 256), Image.BICUBIC))
    gen = np.asarray(Image.open(os.path.join(out, "images", f"{i:05d}.png")).convert("RGB"))
    m = np.asarray(Image.open(os.path.join(out, "masks", f"{i:05d}.png")).convert("L")) > 127
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = min(max(cy - C // 2, 0), 256 - C), min(max(cx - C // 2, 0), 256 - C)
    crop = lambda a: a[y0:y0 + C, x0:x0 + C]
    mk = crop(m)
    er = np.zeros_like(mk)
    er[1:-1, 1:-1] = mk[1:-1, 1:-1] & mk[:-2, 1:-1] & mk[2:, 1:-1] & mk[1:-1, :-2] & mk[1:-1, 2:]
    ov = crop(gen).copy(); ov[mk & ~er] = [255, 0, 0]
    up = lambda a: Image.fromarray(a).resize((C * Z, C * Z), Image.NEAREST)
    row = Image.new("RGB", (3 * C * Z + 40, C * Z + 18), (255, 255, 255))
    for k, a in enumerate([crop(src), crop(gen), ov]):
        row.paste(up(a), (k * (C * Z + 20), 0))
    label = f"#{i}" + (f"   ratio {vis.get(i)}" if show else "")
    ImageDraw.Draw(row).text((4, C * Z + 3), label, fill=(0, 0, 0))
    rows.append(row)
sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height
p = os.path.join(out, "zoom_review_ratio.png" if show else "zoom_review.png")
sheet.save(p)
print("->", p)
