#!/usr/bin/env python3
"""Gate 0: standardized dummy -> loader -> library U-Net -> evaluator -> JSON."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from src.data.integrity import validate_manifest
from src.generation.dummy import DummyGenerator
from src.inspection.data import RealDataset, generated_dataset
from src.inspection.model import build_model
from src.inspection.results import save_result
from src.metrics.segmentation import evaluate


def run(out, device='cpu', seed=42, root=None, manifest_path=None, category='metal_nut'):
    start = time.perf_counter()
    if bool(root) != bool(manifest_path):
        raise ValueError('--root and --manifest must be supplied together')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; run inside a GPU allocation')
    size = 64
    manifest, real_count = None, 0
    generator = DummyGenerator()
    if manifest_path:
        manifest = validate_manifest(json.loads(Path(manifest_path).read_text()), root)
        rows = lambda part: [r for r in manifest[part] if r['category'] == category]
        if not all(rows(part) for part in ('train_normal', 'test_normal', 'support', 'heldout')):
            raise ValueError('Requested category missing from manifest partitions')
        normal_rows, support_rows = rows('train_normal')[:2], rows('support')[:2]
        normal_set = RealDataset(root, normal_rows, size)
        normals = np.stack([normal_set[i][0].numpy() for i in range(len(normal_set))])
        source_ids = [r['image_path'] for r in normal_rows]
        real_set = RealDataset(root, normal_rows + support_rows, size)
        real_count = len(real_set)
        test_set = RealDataset(root, rows('test_normal') + rows('heldout'), size)
    else:
        normals = np.random.default_rng(seed).uniform(0, 0.2, (2, 3, size, size)).astype(np.float32)
        source_ids = ['fixture/train/0', 'fixture/train/1']
        real_set = None
        # Independent fixtures never reuse training source IDs or generated samples.
        test_normals = np.random.default_rng(seed + 1).uniform(0, 0.2, (2, 3, size, size)).astype(np.float32)
        test_batch = generator.generate(test_normals, 2, seed=seed + 1, category=category,
                                        source_image_ids=['fixture/test/0', 'fixture/test/1'])
        test_set = ConcatDataset([generated_dataset(test_batch), TensorDataset(
            torch.from_numpy(test_normals), torch.zeros(2, 1, size, size))])
    batch = generator.generate(normals, 2, seed=seed, category=category, source_image_ids=source_ids)
    train_set = generated_dataset(batch)
    if real_set is not None:
        train_set = ConcatDataset([real_set, train_set])
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    x, y = next(iter(DataLoader(train_set, batch_size=len(train_set))))
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device))
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    optimizer.step()
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in DataLoader(test_set, batch_size=8):
            predictions.append(model(x.to(device)).sigmoid().cpu().numpy())
            targets.append(y.numpy())
    metrics = evaluate(np.concatenate(predictions), np.concatenate(targets))
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    result = dict(schema_version=1, experiment_id=f'gate0_{category}_dummy_seed{seed}',
                  category=category, method='dummy', real_train_count=real_count,
                  synthetic_count=len(batch.images), seed=seed,
                  generator={'generation_seconds_per_image': float(np.mean([m['generation_seconds'] for m in batch.metadata])),
                             'config': batch.metadata[0]['generation_config']},
                  inspection=metrics,
                  training=dict(architecture='unet', encoder='resnet18', encoder_weights=None,
                                image_size=size, steps=1, learning_rate=1e-4, loss=float(loss.detach().cpu())),
                  evaluation=dict(source='heldout_real' if manifest else 'independent_fixture',
                                  count=len(test_set), threshold=0.5, smoke_only=True,
                                  split_sha256=manifest['manifest_sha256'] if manifest else None),
                  provenance=dict(git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
                                  python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                  device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                                  runtime_seconds=time.perf_counter()-start,
                                  packages={p: importlib.metadata.version(p) for p in (
                                      'segmentation-models-pytorch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'jsonschema')}))
    save_result(out, result)
    print(f'GATE 0 PASS: {out}; smoke only, not a benchmark')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results/metrics/gate0.json')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--root')
    p.add_argument('--manifest')
    p.add_argument('--category', default='metal_nut')
    a = p.parse_args()
    run(a.out, a.device, a.seed, a.root, a.manifest, a.category)


if __name__ == '__main__':
    main()
