#!/usr/bin/env python3
"""Train and evaluate the fixed supervised U-Net on one frozen split."""
import argparse
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from src.data.integrity import validate_manifest, validate_freeze_attestation
from src.generation.dummy import DummyGenerator
from src.inspection.data import RealDataset, generated_dataset
from src.inspection.model import build_model
from src.inspection.results import save_result
from src.metrics.segmentation import evaluate


def train(args):
    start = time.perf_counter()
    manifest_path = Path(args.manifest)
    manifest = validate_manifest(json.loads(manifest_path.read_text()), args.root)
    freeze_path = manifest_path.with_suffix('.freeze.json')
    attestation = json.loads(freeze_path.read_text())
    validate_freeze_attestation(manifest, attestation)
    if args.category not in manifest['categories']:
        raise ValueError('Category is not covered by the frozen split')
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.learning_rate <= 0:
        raise ValueError('Training parameters must be positive; image size must be >= 32')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; run inside a GPU allocation')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    rows = lambda part: [r for r in manifest[part] if r['category'] == args.category]
    for part in ('train_normal', 'test_normal', 'support', 'heldout'):
        if not rows(part):
            raise ValueError(f'Frozen split has no {part} samples for {args.category}')
    train_rows = rows('train_normal') + rows('support')
    train_set = RealDataset(args.root, train_rows, args.image_size)
    generated_seconds = 0.0
    if args.method == 'dummy':
        if args.synthetic_count < 1:
            raise ValueError('dummy method requires --synthetic-count >= 1')
        source_rows = rows('train_normal')[:min(args.synthetic_count, len(rows('train_normal')))]
        source_set = RealDataset(args.root, source_rows, args.image_size)
        normals = np.stack([source_set[i][0].numpy() for i in range(len(source_set))])
        generated = DummyGenerator().generate(
            normals, args.synthetic_count, seed=args.seed, category=args.category,
            source_image_ids=[r['image_path'] for r in source_rows])
        generated_seconds = float(np.mean([m['generation_seconds'] for m in generated.metadata]))
        train_set = ConcatDataset([train_set, generated_dataset(generated)])
    elif args.synthetic_count:
        raise ValueError('real_only method requires --synthetic-count 0')

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=generator,
                              pin_memory=args.device == 'cuda')
    test_rows = rows('test_normal') + rows('heldout')
    test_loader = DataLoader(RealDataset(args.root, test_rows, args.image_size),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.device == 'cuda')
    model = build_model().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []
    steps = 0
    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1

    model.eval()
    probs, masks = [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs.append(model(x.to(args.device, non_blocking=True)).sigmoid().cpu().numpy())
            masks.append(y.numpy())
    inspection = evaluate(np.concatenate(probs), np.concatenate(masks))
    git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a], text=True).strip()
    result = dict(
        schema_version=1,
        experiment_id=f'{args.method}_{args.category}_support{manifest["n_support_requested_per_defect_type"]}_seed{args.seed}',
        category=args.category, method=args.method,
        real_train_count=len(train_rows), synthetic_count=args.synthetic_count, seed=args.seed,
        generator={'generation_seconds_per_image': generated_seconds,
                   'config': {'fixture_only': args.method == 'dummy'}},
        inspection=inspection,
        training={'architecture': 'unet', 'encoder': 'resnet18', 'encoder_weights': None,
                  'image_size': args.image_size, 'steps': steps,
                  'learning_rate': args.learning_rate, 'loss': float(np.mean(losses))},
        evaluation={'source': 'heldout_real', 'count': len(test_rows), 'threshold': 0.5,
                    'smoke_only': False, 'split_sha256': manifest['manifest_sha256']},
        provenance={'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                    'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
                    'runtime_seconds': time.perf_counter() - start,
                    'packages': {p: importlib.metadata.version(p) for p in
                                 ('segmentation-models-pytorch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'jsonschema')}})
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open('xb') as f:
        torch.save({'state_dict': model.state_dict(), 'result': result,
                    'manifest_sha256': manifest['manifest_sha256']}, f)
    save_result(args.out, result)
    print(f'TRAINING COMPLETE: {args.out}; checkpoint: {checkpoint}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--category', required=True)
    p.add_argument('--method', choices=['real_only', 'dummy'], default='real_only')
    p.add_argument('--synthetic-count', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--num-threads', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', required=True)
    train(p.parse_args())


if __name__ == '__main__':
    main()
