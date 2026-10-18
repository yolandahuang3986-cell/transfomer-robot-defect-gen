"""AnomalyDiffusion generation worker (runs in the Python 3.8 'Anomalydiffusion' env).

Reads one request JSON, generates image/mask pairs for ONE defect type on the
EXACT training-normal source images listed in the request, and writes:
  <out_dir>/images/<index>.png   generated defect image, RGB uint8
  <out_dir>/masks/<index>.png    binary defect mask, 0/255 uint8 (lossless PNG)
  <out_dir>/result.json          per-sample timing + provenance

v0 = faithful to upstream generate_mask.py + generate_with_mask.py:
  stage 1: mask sampling from the learned mask token with classifier-free
           guidance 5.0 (upstream's "unconditional_only" just skips the
           unguided sample), mean>0.8 binarisation,
           upstream check_mask border rule
  stage 2: latent inpainting of the source image inside the mask
Differences from upstream (all deliberate): explicit checkpoint paths (no logs/
symlink), caller-chosen source images instead of random.randint, PNG instead
of JPG, configurable DDIM steps, seeded RNGs, warm-up + synchronized timing,
stage-1 input is a zero tensor (sampling starts from noise; the input image
is only VAE-encoded for logging, so its content never reaches the sample),
so generation reads no support or held-out image at all.

Usage:  python ad_worker.py request.json
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from stage1 import run_stage1, stage1_result          # sibling module; sys.path[0] is this script's dir


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config_path, ldm_ckpt):
    """Same as upstream load_model_from_config."""
    from ldm.util import instantiate_from_config
    config = OmegaConf.load(config_path)
    sd = torch.load(ldm_ckpt, map_location="cpu")["state_dict"]
    config.model.params.ckpt_path = ldm_ckpt
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    return model


def check_mask(mask):
    """Upstream generate_mask.check_mask: reject masks covering >1/3 of any border."""
    w = mask.size(-1)
    for edge in (mask[:, 0, :], mask[:, w - 1, :], mask[:, :, 0], mask[:, :, w - 1]):
        if torch.count_nonzero(edge) > w / 3:
            return False
    return True


def mask_filter_reason(g, m, f):
    """Optional stage-1 plausibility filter (request sampling.mask_filter; None = upstream behaviour).
    g: raw grey output (H,W) in about [-1,1]; m: binary mask (1,H,W). Returns a reject reason or None.
    Rules come from the mask_probe failure taxonomy; the area window is derived from the support masks."""
    b = m[0]
    area = float(b.mean()) * 100
    border = float(torch.cat([b[0], b[-1], b[:, 0], b[:, -1]]).mean())
    if area < f["min_area_pct"]:
        return "too_small_or_empty"
    if border > f["max_border_frac"] or area > f["max_area_abs_pct"]:
        return "inverted"
    if float(g.median()) > f["max_bg_median"]:
        return "dirty_bg"
    if area > f["max_area_pct"]:
        return "too_large"
    return None


def place_mask(m, fg, rules, rng):
    """placement_v1: translate the mask's bounding-box crop onto the foreground.
    m, fg: bool arrays (H,W). Returns (placed mask or None, tries used, overlap)."""
    ys, xs = np.nonzero(m)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    H, W = fg.shape
    fy, fx = np.nonzero(fg)
    if h > H or w > W or len(fy) == 0:
        return None, 0, 0.0
    area = crop.sum()
    for t in range(int(rules["max_tries_per_mask"])):
        k = rng.integers(len(fy))
        top = int(np.clip(fy[k] - h // 2, 0, H - h))
        left = int(np.clip(fx[k] - w // 2, 0, W - w))
        overlap = float((crop & fg[top:top + h, left:left + w]).sum()) / area
        if overlap >= rules["min_overlap"]:
            out = np.zeros_like(m)
            out[top:top + h, left:left + w] = crop
            return out, t + 1, overlap
    return None, int(rules["max_tries_per_mask"]), 0.0


def latent_cells(m):
    """Cells of the 32x32 latent inpainting mask exactly as ddpm.log_images builds it."""
    from torchvision import transforms
    return int((transforms.Resize((32, 32))(m[None].float()) >= 0.5).sum())


def cuda_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def captions(n):
    from ldm.data.personalized import imagenet_templates_small
    return [random.choice(imagenet_templates_small).format("*") for _ in range(n)]


def load_source(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), resample=Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 127.5 - 1.0          # HWC in [-1, 1]


def main():
    req = json.load(open(sys.argv[1]))
    os.chdir(req["upstream_dir"])                     # configs and name-anomaly.txt are relative paths
    sys.path.insert(0, req["upstream_dir"])
    ck, sp = req["ckpt"], req["sampling"]
    cat, anom = req["category"], req["defect_type"]
    name, size, bs = f"{cat}+{anom}", 256, int(sp["batch_size"])
    samples = req["samples"]
    n = len(samples)
    out = req["out_dir"]
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    save_rejects = bool(req.get("save_rejects", False))     # keep every rejected stage-1 mask as evidence
    if save_rejects:
        os.makedirs(os.path.join(out, "rejects"), exist_ok=True)
    dev = torch.device("cuda")
    setup_seed(int(req["seed"]))

    # ---------------- stage 1: masks ----------------
    t0 = time.perf_counter()
    mmodel = load_model("configs/latent-diffusion/txt2img-1p4B-finetune.yaml", ck["ldm"])
    mmodel.embedding_manager.load(ck["mask_embeddings"])
    mmodel = mmodel.to(dev)
    mask_load_s = time.perf_counter() - t0

    def mask_batch(steps):
        batch = {"image": torch.zeros(bs, size, size, 3), "caption": captions(bs)}
        with torch.no_grad(), mmodel.ema_scope():
            logs = mmodel.log_images(batch, N=bs, sample=True, inpaint=False,
                                     unconditional_only=True, ddim_steps=steps)
        return logs["samples_scaled"].cpu()

    mask_batch(2)                                     # warm-up, not timed, discarded
    mfilter = sp.get("mask_filter")                   # None -> faithful upstream (check_mask only)
    placement = req.get("placement", "v0")
    if placement in ("v1", "v2"):                    # same algorithm; v2 differs only in its foreground PNGs
        prules = req["placement_rules"]
        fgs = [np.asarray(Image.open(s["foreground_png"]).convert("L")) > 127 for s in samples]
        assert all(f.shape == (size, size) for f in fgs), "foregrounds must be 256x256"
        place_rng = np.random.default_rng(int(req["seed"]))
    elif placement != "v0":
        raise SystemExit(f"unknown placement {placement}")
    rejects = []

    def judge(im, attempt, accepted_so_far):
        g = im.mean(0)
        m = (g > 0.8).float()[None]                               # upstream binarisation
        reason = None if check_mask(m) else "check_mask"
        if reason is None and mfilter:
            reason = mask_filter_reason(g, m, mfilter)
        info = {}
        if reason is None and placement in ("v1", "v2"):
            placed, tries, overlap = place_mask(m[0].numpy() > 0.5, fgs[accepted_so_far], prules, place_rng)
            if placed is None:
                reason = "placement_failed"
            else:
                m = torch.from_numpy(placed.astype(np.float32))[None]
                info = {"placement_tries": tries, "placement_overlap": round(overlap, 4)}
        if reason is None and mfilter and "min_latent_cells" in mfilter:
            cells = latent_cells(m)                               # after placement: the grid alignment matters
            info["latent_cells"] = cells
            if cells < int(mfilter["min_latent_cells"]):
                reason = "latent_too_small"
        info["mask_attempt"] = attempt
        if reason is not None and save_rejects:                   # raw grey sample + the binary mask the rule judged
            stem = f"{attempt:05d}_{reason}"
            grey = ((g.clamp(-1, 1) + 1) / 2 * 255).round().byte().numpy()
            Image.fromarray(grey).save(os.path.join(out, "rejects", stem + "_raw.png"))
            Image.fromarray((m[0].numpy() > 0.5).astype(np.uint8) * 255).save(
                os.path.join(out, "rejects", stem + "_mask.png"))
            rejects.append({"attempt": attempt, "reason": reason,
                            "target_index": int(samples[accepted_so_far]["index"]), **info})
        return m, reason, info

    # the budget counts judged masks; a full last batch past the budget is surplus, not extra attempts
    s1 = run_stage1(lambda: cuda_time(lambda: mask_batch(int(sp["mask_ddim_steps"]))), judge,
                    n, n * int(sp.get("mask_attempt_factor", 10)))
    masks, place_info, mask_s = s1["masks"], s1["infos"], s1["seconds"]
    requested = n
    underfilled = len(masks) < n
    if underfilled:                                   # keep what we have + all statistics, do not throw them away
        print(f"[worker] WARNING underfilled: only {len(masks)}/{n} masks after {s1['attempts']} "
              f"of {s1['budget']} attempts {s1['reasons']}", flush=True)
        n = len(masks)
        samples = samples[:n]
    if save_rejects:
        json.dump(rejects, open(os.path.join(out, "rejects.json"), "w"), indent=2)

    def provenance():
        here = os.path.dirname(os.path.abspath(__file__))
        sha = lambda f: __import__("hashlib").sha256(open(f, "rb").read()).hexdigest()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        patched = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True).stdout.strip()
        return {"upstream_commit": commit, "upstream_patches": patched,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "torchvision": __import__("torchvision").__version__,
                "pillow": Image.__version__, "numpy": np.__version__,
                "worker_sha256": sha(os.path.abspath(__file__)),
                "stage1_sha256": sha(os.path.join(here, "stage1.py")),
                "gpu": torch.cuda.get_device_name(0), "ckpt": ck, "sampling": sp,
                "seed": int(req["seed"])}

    note = "stage-1 counts are Diffusion-internal; shared QC statistics are reported separately by the adapter"
    if n == 0:                                        # same statistics as a normal run; stage 2 never loaded
        json.dump({"name": name, "placement": placement,
                   **stage1_result(requested, s1, mask_load_s, mask_filter=mfilter),
                   "placement_rules": req.get("placement_rules"), "note": note,
                   "provenance": provenance(), "samples": []},
                  open(os.path.join(out, "result.json"), "w"), indent=2)
        raise SystemExit(2)
    del mmodel
    torch.cuda.empty_cache()

    # ---------------- stage 2: inpainting ----------------
    t0 = time.perf_counter()
    amodel = load_model("configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml", ck["ldm"])
    amodel = amodel.to(dev)
    amodel.prepare_spatial_encoder(optimze_together=True)
    amodel.embedding_manager.spatial_encoder_model.load_state_dict(torch.load(ck["spatial_encoder"]))
    amodel.embedding_manager.load(ck["anomaly_embeddings"])
    image_load_s = time.perf_counter() - t0

    def inpaint(src, msk, steps):
        k = len(src)
        if k == 1:                                    # upstream get_input does mask.squeeze(): a batch of 1
            src, msk = src * 2, msk * 2               # would lose its batch dim -> pad to 2, keep the first
        batch = {"image": torch.from_numpy(np.stack(src)), "mask": torch.cat(msk, 0),
                 "caption": captions(len(src)), "name": [name] * len(src)}
        with torch.no_grad(), amodel.ema_scope():
            logs = amodel.log_images(batch, N=len(src), sample=False, inpaint=True, unconditional_only=True,
                                     ddim_steps=steps, adaptive_mask=bool(sp["adaptive_mask"]))
        return logs["samples_inpainting"][:k].cpu()

    first = load_source(os.path.join(req["mvtec_root"], samples[0]["source_image_id"]), size)
    inpaint([first], [masks[0]], 2)                   # warm-up, not timed, discarded

    idx_pos = {int(s["index"]): k for k, s in enumerate(samples)}
    mask_share = mask_s / n                           # stage-1 cost incl. rejected masks, per accepted sample
    records, inpaint_total = [], 0.0
    for start in range(0, n, bs):
        chunk = samples[start:start + bs]
        src = [load_source(os.path.join(req["mvtec_root"], s["source_image_id"]), size) for s in chunk]
        msk = masks[start:start + bs]
        gen, dt = cuda_time(lambda: inpaint(src, msk, int(sp["image_ddim_steps"])))
        inpaint_total += dt
        per = dt / len(chunk)
        for s, g, m in zip(chunk, gen, msk):
            idx = int(s["index"])
            rgb = ((g.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
            Image.fromarray(rgb).save(os.path.join(out, "images", f"{idx:05d}.png"))
            Image.fromarray((m[0].numpy() * 255).astype(np.uint8)).save(os.path.join(out, "masks", f"{idx:05d}.png"))
            records.append({"index": idx, "source_image_id": s["source_image_id"],
                            "mask_seconds": mask_share, "inpaint_seconds": per,
                            "generation_seconds": mask_share + per, **place_info[idx_pos[idx]]})

    json.dump({"name": name, "placement": placement,
               **stage1_result(requested, s1, mask_load_s, image_load_s, inpaint_total, mfilter),
               "placement_rules": req.get("placement_rules"), "note": note,
               "provenance": provenance(), "samples": records},
              open(os.path.join(out, "result.json"), "w"), indent=2)
    print(f"[worker] {name}: {n}/{requested} pairs -> {out} | mask attempts {s1['attempts']}/{s1['budget']} "
          f"(rejected {sum(s1['reasons'].values())} {s1['reasons']}, surplus {s1['surplus']}) | "
          f"load {mask_load_s + image_load_s:.0f}s | "
          f"mean gen {np.mean([r['generation_seconds'] for r in records]):.2f}s/img")


if __name__ == "__main__":
    main()
