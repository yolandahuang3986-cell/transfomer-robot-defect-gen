"""Contact sheet for one exported bundle: one row per pair, all pairs in bundle order, none selected.
Columns: source (train/good) | mask | generated | outlines on generated
(red = mask, green = placement foreground from the worker request).

Reads the bundle's batch.npz and metadata.json directly, so the sheet shows exactly the pairs
in the bundle. make_contact_sheet.py reads a worker run dir instead; for a bundle built from
several worker calls (shared-QC retries) that dir also holds QC-rejected pairs.
Read-only: never writes into the bundle or the raw worker dirs; refuses to overwrite <out.png>.

Usage: python bundle_contact_sheet.py <bundle_dir> <out.png>   (light, CPU only)
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
bundle, dest = sys.argv[1], sys.argv[2]
if os.path.exists(dest): sys.exit(f"{dest} exists")
md = json.load(open(os.path.join(bundle, "metadata.json")))
md = md if isinstance(md, list) else md.get("metadata", md.get("samples"))
z = np.load(os.path.join(bundle, "batch.npz")); I, M = z["images"], z["masks"]
assert len(md) == len(I) == len(M)
S = 256; rows = []
def outline(b):
    e = np.zeros_like(b); e[1:-1, 1:-1] = b[1:-1, 1:-1] & b[:-2, 1:-1] & b[2:, 1:-1] & b[1:-1, :-2] & b[1:-1, 2:]
    return b & ~e
for k, m in enumerate(md):
    rd = m["generation_config"]["raw_output_dir"]; req = json.load(open(os.path.join(rd, "request.json")))
    src = Image.open(os.path.join(req["mvtec_root"], m["source_image_id"])).convert("RGB").resize((S, S), Image.BICUBIC)
    mk = M[k, 0] > 0.5
    gen = (I[k].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    ov = gen.copy(); ov[outline(mk)] = [255, 0, 0]
    fgp = [s["foreground_png"] for s in req["samples"] if s["source_image_id"] == m["source_image_id"] and "foreground_png" in s]
    if fgp:
        ov[outline(np.asarray(Image.open(fgp[0]).convert("L")) > 127)] = [0, 255, 0]
    row = Image.new("RGB", (4 * S, S))
    for c, im in enumerate([src, Image.fromarray((mk * 255).astype(np.uint8)).convert("RGB"), Image.fromarray(gen), Image.fromarray(ov)]):
        row.paste(im, (c * S, 0))
    ImageDraw.Draw(row).text((4, 4), f"#{k} {m['source_image_id'].split('/')[-1]} run {os.path.basename(rd)}", fill=(255, 255, 0))
    rows.append(row)
sheet = Image.new("RGB", (4 * S, S * len(rows)))
for r, row in enumerate(rows): sheet.paste(row, (0, r * S))
sheet.save(dest); print("->", dest, sheet.size)
