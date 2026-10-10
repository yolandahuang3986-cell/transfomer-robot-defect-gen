#!/usr/bin/env python3
"""B integration check on generated samples only. Not a held-out benchmark."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.integrity import validate_manifest,validate_freeze_attestation
from src.generation.bundle import load_bundle
from src.inspection.data import generated_dataset
from src.inspection.model import build_model
from src.metrics.segmentation import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--category',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); path=Path(a.manifest)
    m=validate_manifest(json.loads(path.read_text()))
    validate_freeze_attestation(m,json.loads(path.with_suffix('.freeze.json').read_text()))
    batch=load_bundle(a.bundle,m,a.category)
    if len(batch.images)<4:raise ValueError('Need four generated samples')
    torch.manual_seed(42);torch.set_num_threads(2)
    data=generated_dataset(batch)
    x,y=data.tensors
    # Reduce resolution for a cheap CPU integration check; formal runs remain 256px.
    x=torch.nn.functional.interpolate(x[:4],size=(64,64),mode='bilinear',align_corners=False)
    y=torch.nn.functional.interpolate(y[:4],size=(64,64),mode='nearest')
    model=build_model();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    start=time.perf_counter();model.train();optimizer.zero_grad()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[:2]),y[:2])
    assert torch.isfinite(loss);loss.backward();optimizer.step();model.eval()
    with torch.no_grad():prob=model(x[2:]).sigmoid().numpy()
    metrics=evaluate(prob,y[2:].numpy())
    result=dict(check='procedural_loader_unet_evaluator',status='passed',smoke_only=True,
        evaluation_source='synthetic_samples_not_real_benchmark',category=a.category,
        optimizer_steps=1,device='cpu',image_size=64,loss=float(loss.detach()),metrics=metrics,
        seconds=time.perf_counter()-start,torch=torch.__version__,split_sha256=m['manifest_sha256'])
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
