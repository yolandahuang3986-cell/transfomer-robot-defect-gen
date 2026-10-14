"""Probe tighter 'object core' foregrounds for placement (train/good images only, CPU, a few seconds).
Variants on the 256 output grid:
  qc        shared QC estimate_object_foreground (reference; asserted identical to our re-computation)
  thr x2/x3 same background model, distance threshold multiplied by 2 / 3
  thr x3 + fill   thr x3 with enclosed holes filled (specular highlights on metal look like background)
  core      (thr x3 + fill) AND qc  -> placement_v2 candidate: QC foreground minus the soft halo,
            never larger than QC (so a real through-hole that QC excludes stays excluded)
Writes fg_probe.png (contours: green=qc, cyan=thr x3, magenta=thr x3 + fill, yellow=core) and fg_probe.json.
Runs in 'defectgen' from the repo root (PYTHONPATH=$PWD).
Usage: python fg_probe.py --category screw --n 8 --out_dir <dir>"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))  # repo root
from src.generation.qc import DEFAULT_QC_POLICY as P, estimate_object_foreground, _resize_binary_mask

ap = argparse.ArgumentParser()
ap.add_argument("--category", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
root = os.path.expanduser("~/datasets/mvtec_ad")
files = sorted(os.listdir(f"{root}/{a.category}/train/good"))[: a.n]
os.makedirs(a.out_dir, exist_ok=True)


def foreground(image, mult):
    """Line-by-line copy of the QC estimator with the final threshold multiplied by `mult`."""
    h, w = image.shape[1:]
    band = max(1, int(round(min(h, w) * P.foreground_border_fraction)))
    hwc = np.moveaxis(image, 0, -1)
    border = np.zeros((h, w), bool)
    border[:band, :] = border[-band:, :] = border[:, :band] = border[:, -band:] = True
    bg_px = hwc[border]
    bg = np.median(bg_px, axis=0)
    d = np.linalg.norm(bg_px - bg, axis=1)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    thr = max(P.foreground_distance_floor, med + P.foreground_mad_multiplier * 1.4826 * mad) * mult
    fg = np.linalg.norm(hwc - bg, axis=2) > thr
    comp, cnt = label(fg, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(comp.ravel()); sizes[0] = 0
    return comp == int(np.argmax(sizes)), thr


def outline(m):
    e = np.zeros_like(m)
    e[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return m & ~e


colors = {"qc": (0, 255, 0), "thr_x3": (0, 255, 255), "thr_x3_fill": (255, 0, 255), "core": (255, 255, 0)}
rows, report = [], []
for i, f in enumerate(files):
    sid = f"{a.category}/train/good/{f}"
    img = np.moveaxis(np.asarray(Image.open(os.path.join(root, sid)).convert("RGB"), np.float32) / 255.0, -1, 0)
    qc_native = estimate_object_foreground(img, P)
    mine, thr = foreground(img, 1.0)
    assert np.array_equal(mine, qc_native), "re-computation differs from shared QC"
    v = {"qc": _resize_binary_mask(qc_native, (256, 256)),
         "thr_x3": _resize_binary_mask(foreground(img, 3.0)[0], (256, 256))}
    v["thr_x3_fill"] = binary_fill_holes(v["thr_x3"])
    v["core"] = v["thr_x3_fill"] & v["qc"]
    rgb = (np.moveaxis(img, 0, -1) * 255).astype(np.uint8)
    base = np.asarray(Image.fromarray(rgb).resize((256, 256), Image.BICUBIC)).copy()
    panels = [base.copy()]
    for k in ("qc", "thr_x3", "thr_x3_fill", "core"):
        p = base.copy(); p[outline(v[k])] = colors[k]; panels.append(p)
    rows.append(np.concatenate(panels, 1))
    rec = {"index": i, "source_image_id": sid, "qc_threshold": round(thr, 4),
           **{f"{k}_pct": round(float(m.mean()) * 100, 2) for k, m in v.items()},
           "fill_outside_qc_pct": round(float((v["thr_x3_fill"] & ~v["qc"]).sum()) / max(1, v["thr_x3_fill"].sum()) * 100, 2),
           "core_share_of_qc_pct": round(float(v["core"].sum()) / v["qc"].sum() * 100, 1)}
    report.append(rec)
    print(rec)
Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.out_dir, "fg_probe.png"))
json.dump(report, open(os.path.join(a.out_dir, "fg_probe.json"), "w"), indent=2)
print("->", a.out_dir)
