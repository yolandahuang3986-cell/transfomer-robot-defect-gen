#!/usr/bin/env python3
"""Export one AnomalyDiffusion bundle (one defect type) through shared QC.

Run in the project environment on a GPU node; the AnomalyDiffusion worker is started
as a subprocess in its own environment. Writes batch.npz, metadata.json,
provenance.json (with the full shared-QC report) and diffusion_internal.json (stage-1
statistics, reported separately from shared QC). An underfilled run writes its reports
and exits non-zero; nothing is backfilled.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.integrity import sha256, validate_freeze_attestation, validate_manifest
from src.experiments.scope import require_in_scope
from src.generation.diffusion_adapter import DiffusionDefectGenerator, DiffusionUnderfilledError, summarize_runs
from src.generation.qc import QCUnderfilledError

CODE = ['scripts/diffusion/ad_worker.py', 'scripts/diffusion/stage1.py', 'scripts/diffusion/support_stats.py', 'scripts/generate_diffusion.py',
        'src/generation/diffusion_adapter.py', 'src/generation/diffusion_foreground.py', 'src/generation/qc.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/diffusion/generation.json')
    p.add_argument('--manifest', default='data/splits/fewshot_n5_seed42.json')
    p.add_argument('--category', required=True, choices=['metal_nut', 'screw'])
    p.add_argument('--defect-type', required=True)
    p.add_argument('--num-samples', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--purpose', required=True, choices=['validation', 'formal'],
                   help='validation bundles are evidence only and never enter the formal pool')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    require_in_scope(a.category, a.defect_type)
    out = Path(a.out).expanduser().resolve()
    if out.is_relative_to(ROOT) or out.exists():
        p.error('Use a new output directory outside the repository')
    config = json.loads((ROOT / a.config).read_text())
    root = Path(config['root']).expanduser()
    mpath = ROOT / a.manifest
    manifest = validate_manifest(json.loads(mpath.read_text()), root)
    validate_freeze_attestation(manifest, json.loads(mpath.with_suffix('.freeze.json').read_text()))

    rows = [r for r in manifest['train_normal'] if r['category'] == a.category]
    normals = []
    for r in rows:                       # same grid/resize as the formal loader's QC recheck
        with Image.open(root / r['image_path']) as im:
            normals.append(np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR),
                                      np.float32).transpose(2, 0, 1) / 255)
    normals = np.stack(normals)
    ids = [r['image_path'] for r in rows]

    generator = DiffusionDefectGenerator(config, manifest)
    start = time.perf_counter()
    status, qc_report, batch, error = 'ok', None, None, None
    try:
        result = generator.generate_with_qc(normals, ids, target_count=a.num_samples,
                                            defect_type=a.defect_type, seed=a.seed, category=a.category)
        qc_report = result.report.to_dict()
        batch = result.require_target()
    except QCUnderfilledError as e:
        status, qc_report, error = 'underfilled_shared_qc', e.result.report.to_dict(), str(e)
    except DiffusionUnderfilledError as e:
        status, error = 'underfilled_stage1', str(e)
    wall = time.perf_counter() - start

    out.mkdir(parents=True)
    internal = {'note': 'Diffusion-internal stage 1 (mask filter + placement). Not shared QC. '
                        'generation_seconds_total includes rejected attempts and QC-rejected pairs; '
                        'model loads are separate.',
                'runs': generator.runs, 'totals': summarize_runs(generator.runs)}
    (out / 'diffusion_internal.json').write_text(json.dumps(internal, indent=2) + '\n')
    if qc_report is not None:
        (out / 'qc_report.json').write_text(json.dumps(qc_report, indent=2) + '\n')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
    provenance = dict(method='diffusion', purpose=a.purpose, status=status, error=error,
                      category=a.category, defect_type=a.defect_type, seed=a.seed,
                      num_samples=a.num_samples, split_sha256=manifest['manifest_sha256'],
                      code_commit=commit, code_dirty=dirty, code_sha256={f: sha256(ROOT / f) for f in CODE},
                      config=config, config_sha256=sha256(ROOT / a.config), shared_qc=qc_report,
                      diffusion_internal_totals=internal['totals'], wall_seconds_including_model_loads=wall,
                      python=platform.python_version(), numpy=np.__version__)
    if batch is not None:
        np.savez_compressed(out / 'batch.npz', images=batch.images, masks=batch.masks)
        (out / 'metadata.json').write_text(json.dumps(batch.metadata, indent=2, allow_nan=False) + '\n')
        provenance['file_sha256'] = {f: sha256(out / f) for f in ('batch.npz', 'metadata.json')}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(out), 'status': status, 'purpose': a.purpose,
                      'mask_attempts': internal['totals']['mask_attempts'],
                      'generation_seconds_total': round(internal['totals']['generation_seconds_total'], 1),
                      'shared_qc_accepted': qc_report and qc_report['accepted_count'],
                      'shared_qc_attempted': qc_report and qc_report['attempted_count']}))
    if status != 'ok':
        sys.exit(1)


if __name__ == '__main__':
    main()
