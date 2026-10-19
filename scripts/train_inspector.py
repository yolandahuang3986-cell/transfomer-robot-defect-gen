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
from src.inspection.results import save_result, validate_result
from src.metrics.segmentation import evaluate
from src.experiments.formal import settings, fixed_step_loader
from src.inspection.formal_bundle import load_formal_pool, require_acceptance
from src.data.integrity import sha256
from src.experiments.scope import excluded_rows, filter_in_scope, PROTOCOL_VERSION


def train(args):
    start = time.perf_counter()
    protocol = settings()
    formal = getattr(args, 'formal', False)
    if formal:
        if args.category not in protocol['optimizer_steps'] or args.seed not in protocol['downstream_seeds']:
            raise ValueError('Category/seed outside the frozen primary protocol')
        if (args.image_size, args.batch_size, args.learning_rate) != (256, 8, 1e-4):
            raise ValueError('Formal training requires 256px, batch 8, learning rate 1e-4')
        if args.method == 'dummy':
            raise ValueError('Dummy cannot be a formal experiment')
        if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Commit all implementation changes before formal training')
    elif args.method not in {'real_only', 'dummy'}:
        raise ValueError('Synthetic training requires --formal and accepted bundles')
    if args.method in {'real_only', 'dummy'} and (getattr(args, 'bundle', None) or getattr(args, 'acceptance', None)):
        raise ValueError('Real-only/dummy must not receive synthetic bundles or acceptance')
    ratio_tier_count = getattr(args, 'ratio_tier_count', None)
    ratio_append_seeds = getattr(args, 'ratio_append_seed', None) or None
    if ratio_tier_count is not None and not (formal and args.method not in {'real_only', 'dummy'}):
        raise ValueError('--ratio-tier-count is only valid with --formal and a synthetic method')
    if ratio_append_seeds is not None and ratio_tier_count is None:
        raise ValueError('--ratio-append-seed is only valid together with --ratio-tier-count')
    for output in (args.out, args.checkpoint):
        if Path(output).exists():
            raise FileExistsError(f'Refusing to overwrite output: {output}')
    manifest_path = Path(args.manifest)
    manifest = validate_manifest(json.loads(manifest_path.read_text()), args.root)
    if formal and manifest['manifest_sha256'] != protocol['split_sha256']:
        raise ValueError('Formal experiment requires the approved frozen manifest')
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
    train_rows = rows('train_normal') + filter_in_scope(rows('support'))
    excluded_support = excluded_rows(rows('support'))
    train_set = RealDataset(args.root, train_rows, args.image_size)
    generated_seconds = 0.0
    generator_config = {'fixture_only': args.method == 'dummy'}
    synthetic_count = args.synthetic_count
    if formal and args.method != 'real_only':
        if args.synthetic_count:
            raise ValueError('Formal quota is fixed by protocol, not --synthetic-count')
        if not args.acceptance:
            raise ValueError('Synthetic training requires --acceptance')
        batch, identity, reports = load_formal_pool(args.bundle, manifest, args.category, args.method, args.root,
                                                    ratio_tier_count=ratio_tier_count,
                                                    ratio_append_seeds=ratio_append_seeds)
        require_acceptance(args.acceptance, identity)
        synthetic_count = len(batch.images)
        train_set = ConcatDataset([train_set, generated_dataset(batch)])
        generated_seconds = sum(r['generation_seconds'] for r in reports) / synthetic_count
        generator_config = {'pool': identity, 'qc_reports': reports,
                            'acceptance_sha256': sha256(args.acceptance),
                            'cost_basis': 'reported generation seconds including shared-QC rejected attempts; not total wall time'}
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
    if formal:
        train_loader = fixed_step_loader(train_set, protocol['optimizer_steps'][args.category],
                                         args.batch_size, args.seed, args.num_workers, args.device == 'cuda')
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, generator=generator,
                                  pin_memory=args.device == 'cuda')
    test_rows = rows('test_normal') + filter_in_scope(rows('heldout'))
    excluded_heldout = excluded_rows(rows('heldout'))
    test_loader = DataLoader(RealDataset(args.root, test_rows, args.image_size),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.device == 'cuda')
    model = build_model().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []
    steps = 0
    for _ in range(1 if formal else args.epochs):
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
        real_train_count=len(train_rows), synthetic_count=synthetic_count, seed=args.seed,
        generator={'generation_seconds_per_image': generated_seconds,
                   'config': generator_config},
        inspection=inspection,
        training={'architecture': 'unet', 'encoder': 'resnet18', 'encoder_weights': None,
                  'image_size': args.image_size, 'steps': steps,
                  'learning_rate': args.learning_rate, 'loss': float(np.mean(losses)),
                  'batch_size': args.batch_size, 'sampler': 'uniform_with_replacement' if formal else 'epoch_shuffle',
                  'checkpoint_selection': 'final_step'},
        evaluation={'source': 'heldout_real', 'count': len(test_rows), 'threshold': 0.5,
                    'scope_protocol': PROTOCOL_VERSION,
                    'excluded_support_count': len(excluded_support),
                    'excluded_heldout_count': len(excluded_heldout),
                    'formal_protocol': protocol['version'] if formal else None,
                    'smoke_only': False, 'split_sha256': manifest['manifest_sha256']},
        provenance={'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                    'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
                    'runtime_seconds': time.perf_counter() - start,
                    'packages': {p: importlib.metadata.version(p) for p in
                                 ('segmentation-models-pytorch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'jsonschema')}})
    validate_result(result)
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
    p.add_argument('--method', choices=['real_only', 'dummy', 'procedural', 'gan', 'diffusion'], default='real_only')
    p.add_argument('--formal', action='store_true', help='Enforce frozen primary protocol and fixed optimizer budget')
    p.add_argument('--bundle', action='append', default=[], help='Accepted bundle directory; repeat for conditional strata')
    p.add_argument('--acceptance', help='A whole-pool acceptance JSON bound to exact artifact hashes')
    p.add_argument('--synthetic-count', type=int, default=0)
    p.add_argument('--ratio-tier-count', type=int, default=None,
                   help='Secondary ratio-analysis quota override (e.g. 15 or 150 for metal_nut instead '
                        'of the frozen 75/125 or per-type 25); never the primary quota, only valid with '
                        '--formal and a synthetic method, and bound into the acceptance record it requires')
    p.add_argument('--ratio-append-seed', type=int, action='append', default=None,
                   help='Disclosed extra generation seed(s) a ratio-tier append bundle is allowed to use '
                        '(e.g. for an r=10 bundle built by keeping the primary r=5 pool and appending new '
                        'samples under a fresh seed to avoid a directory collision); repeat for multiple '
                        'seeds; only valid together with --ratio-tier-count, never for a primary pool')
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
