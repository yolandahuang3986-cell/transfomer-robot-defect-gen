"""Failure-taxonomy evidence: a sheet of rejected stage-1 masks, a few per reason.
Needs a worker run with save_rejects (rejects/ + rejects.json). Picks the first k per reason
in attempt order (deterministic, no cherry-picking). Each tile = raw grey sample | judged mask.
Runs in any env with numpy + PIL.  Usage: python rejects_sheet.py <raw run dir> [k=6]"""
import json, os, sys
from collections import defaultdict
from PIL import Image, ImageDraw

run = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rows = json.load(open(os.path.join(run, "rejects.json")))
by = defaultdict(list)
for r in rows:
    by[r["reason"]].append(r)
S = 96
reasons = sorted(by, key=lambda x: -len(by[x]))
sheet = Image.new("RGB", (220 + k * (2 * S + 8), len(reasons) * (S + 8) + 8), "white")
d = ImageDraw.Draw(sheet)
for row, reason in enumerate(reasons):
    y = 8 + row * (S + 8)
    d.text((6, y + S // 2 - 6), f"{reason} ({len(by[reason])})", fill="black")
    for j, r in enumerate(by[reason][:k]):
        stem = f"{r['attempt']:05d}_{reason}"
        x = 220 + j * (2 * S + 8)
        for t, suffix in enumerate(("_raw.png", "_mask.png")):
            tile = Image.open(os.path.join(run, "rejects", stem + suffix)).convert("RGB").resize((S, S))
            sheet.paste(tile, (x + t * S, y))
        d.text((x + 2, y + 2), str(r["attempt"]), fill=(255, 0, 0))
out = os.path.join(run, "rejects_sheet.png")
sheet.save(out)
print("->", out, {r: len(v) for r, v in by.items()})
