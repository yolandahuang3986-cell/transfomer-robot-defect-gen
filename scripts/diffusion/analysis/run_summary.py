"""Cost and outcome summary of one exported diffusion bundle (reads provenance/metadata/internal JSON).
Usage: python run_summary.py <bundle out dir>"""
import json, os, sys
import numpy as np

out = sys.argv[1]
prov = json.load(open(os.path.join(out, "provenance.json")))
internal = json.load(open(os.path.join(out, "diffusion_internal.json")))
meta = json.load(open(os.path.join(out, "metadata.json"))) if os.path.exists(os.path.join(out, "metadata.json")) else []
g = [m["generation_seconds"] for m in meta]
loads = sum(r["model_load_seconds"] for r in internal["runs"])
t = internal["totals"]
qc = prov.get("shared_qc") or {}
print(json.dumps({
    "status": prov["status"], "purpose": prov["purpose"], "commit": prov["code_commit"][:7], "dirty": prov["code_dirty"],
    "category": prov["category"], "defect_type": prov["defect_type"], "samples": len(meta),
    "worker_calls": len(internal["runs"]),
    "stage1_mask_attempts": t["mask_attempts"], "stage1_rejected_by_reason": t["rejected_by_reason"],
    "stage1_yield_pct": round(100 * t["produced_pairs_sent_to_shared_qc"] / max(1, t["mask_attempts"]), 2),
    "shared_qc": {k: qc.get(k) for k in ("policy_version", "attempted_count", "accepted_count", "rejection_counts")},
    "generation_seconds_per_accepted_sample_mean": round(float(np.mean(g)), 2) if g else None,
    "generation_seconds_sum": round(float(np.sum(g)), 1) if g else None,
    "model_load_seconds_total": round(loads, 1),
    "wall_seconds_including_model_loads": round(prov["wall_seconds_including_model_loads"], 1),
    "sources": sorted({m["source_image_id"].split("/")[-1] for m in meta}),
}, indent=2))
