#!/usr/bin/env python3
"""Training-only fixed-model timing; never compute held-out metrics."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.integrity import validate_manifest, validate_freeze_attestation
from src.experiments.formal import settings, fixed_step_loader
from src.experiments.scope import filter_in_scope
from src.inspection.data import RealDataset
from src.inspection.model import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', default=str(ROOT/'data/splits/fewshot_n5_seed42.json'))
    p.add_argument('--category', choices=['metal_nut','screw'], default='metal_nut')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        p.error('Output already exists')
    path = Path(a.manifest)
    m = validate_manifest(json.loads(path.read_text()), a.root)
    validate_freeze_attestation(m, json.loads(path.with_suffix('.freeze.json').read_text()))
    if m['manifest_sha256'] != settings()['split_sha256']:
        raise ValueError('Expected frozen primary split')
    rows = [r for r in m['train_normal'] + filter_in_scope(m['support']) if r['category']==a.category]
    torch.manual_seed(42); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    model=build_model().to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    if a.device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    measured_start=None
    for step,(x,y) in enumerate(fixed_step_loader(RealDataset(a.root,rows,256),25,8,42,num_workers=4)):
        if step==5:
            if a.device=='cuda': torch.cuda.synchronize()
            measured_start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x.to(a.device)),y.to(a.device))
        loss.backward(); optimizer.step()
    if a.device=='cuda': torch.cuda.synchronize()
    seconds=(time.perf_counter()-measured_start)/20
    result=dict(training_only=True, warmup_steps=5, measured_steps=20, seconds_per_step=seconds,
                category=a.category, batch_size=8,image_size=256,
                projected_training_seconds={k:v*seconds for k,v in settings()['optimizer_steps'].items()},
                projection_caveat='Excludes setup, evaluation and queue time; other-category projection is approximate',
                device=torch.cuda.get_device_name() if a.device=='cuda' else 'cpu',
                peak_allocated_vram_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None)
    with Path(a.out).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    main()
