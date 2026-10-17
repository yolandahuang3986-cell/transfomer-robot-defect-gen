"""Torch-free stage-1 bookkeeping, shared by ad_worker.py (Python 3.8) and the tests.

run_stage1 enforces the attempt budget per judged mask, not per sampled batch.
The upstream sampler always returns a full batch, so outputs beyond the budget
(or beyond the requested count) are discarded without being judged and are
counted as surplus. Their sampling time still counts as cost.

stage1_result builds the statistics block of result.json. The worker uses it for
both the normal path and the zero-output path, so both report the same keys.
"""


def run_stage1(sample_batch, judge, n, max_attempts):
    """sample_batch() -> (outputs, seconds).
    judge(output, attempt, accepted_so_far) -> (mask, reason, info); reason None = accepted.
    Never judges more than max_attempts outputs."""
    if n < 1 or max_attempts < 1:
        raise ValueError("n and max_attempts must be positive")
    masks, infos, reasons = [], [], {}
    attempts = surplus = 0
    seconds = 0.0
    while len(masks) < n and attempts < max_attempts:
        outputs, dt = sample_batch()
        outputs = list(outputs)
        if not outputs:
            raise RuntimeError("mask sampler returned an empty batch")
        seconds += dt
        for out in outputs:
            if len(masks) >= n or attempts >= max_attempts:
                surplus += 1                         # sampled and paid for, never judged
                continue
            attempts += 1
            mask, reason, info = judge(out, attempts, len(masks))
            if reason is None:
                masks.append(mask)
                infos.append(info)
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
    return {"masks": masks, "infos": infos, "attempts": attempts, "budget": max_attempts,
            "surplus": surplus, "seconds": seconds, "reasons": reasons}


def stage1_result(requested, s1, mask_load_seconds, image_load_seconds=0.0,
                  inpaint_seconds_total=0.0, mask_filter=None):
    """Statistics for result.json. image_load_seconds stays 0.0 only when stage 2 never ran."""
    n = len(s1["masks"])
    reasons = dict(s1["reasons"])
    mf = mask_filter or {}
    return {
        "status": "ok" if n == requested else "underfilled",
        "requested": requested, "n": n,
        "mask_attempts": s1["attempts"], "mask_attempt_budget": s1["budget"],
        "mask_surplus_discarded": s1["surplus"],
        "mask_reject_reasons": reasons,
        "stage1_upstream_check_mask_rejected": reasons.get("check_mask", 0),
        "diffusion_internal_filter": {
            "rules_version": mf.get("rules_version"), "rules_sha256": mf.get("rules_sha256"),
            "rejected": {k: v for k, v in reasons.items() if k != "check_mask"}},
        "mask_model_load_seconds": mask_load_seconds,
        "image_model_load_seconds": image_load_seconds,
        "mask_sampling_seconds_total": s1["seconds"],
        "inpaint_seconds_total": inpaint_seconds_total,
        "generation_seconds_total": s1["seconds"] + inpaint_seconds_total,
    }
