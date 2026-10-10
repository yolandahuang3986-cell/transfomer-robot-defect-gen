#!/usr/bin/env python3
"""Plan or execute primary experiments, explicitly listing unavailable methods."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.formal import settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', default=str(ROOT / 'data/splits/fewshot_n5_seed42.json'))
    p.add_argument('--deliveries', help='JSON list of category/method/bundles/acceptance records')
    p.add_argument('--out', required=True, help='New output directory outside the repository')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--execute', action='store_true', help='Default only writes a plan; no training')
    a = p.parse_args()
    output = Path(a.out).resolve()
    if output.is_relative_to(ROOT) or output.exists():
        p.error('Use a new output directory outside the repository')
    protocol = settings()
    deliveries = json.loads(Path(a.deliveries).read_text()) if a.deliveries else []
    lookup = {}
    for delivery in deliveries:
        key = (delivery['category'], delivery['method'])
        if key in lookup or key[0] not in protocol['optimizer_steps'] or key[1] not in {'procedural', 'gan', 'diffusion'}:
            p.error('Duplicate or unsupported delivery')
        lookup[key] = delivery
    runs = []
    for category in protocol['optimizer_steps']:
        for method in ['real_only', 'procedural', 'gan', 'diffusion']:
            delivery = lookup.get((category, method))
            if method != 'real_only' and not delivery:
                runs.append(dict(category=category, method=method, status='not_run_missing_delivery'))
                continue
            if delivery and delivery.get('status') == 'not_run_quality_gate':
                runs.append(dict(category=category, method=method, status='not_run_quality_gate',
                                 reason=delivery.get('reason', 'See Gate 1 review')))
                continue
            for seed in protocol['downstream_seeds']:
                run_id = f'{method}_{category}_seed{seed}'
                command = [sys.executable, str(ROOT / 'scripts/train_inspector.py'), '--formal',
                           '--root', str(Path(a.root).expanduser().resolve()), '--manifest', str(Path(a.manifest).resolve()),
                           '--category', category, '--method', method, '--seed', str(seed), '--device', a.device,
                           '--out', str(output / f'{run_id}.json'),
                           '--checkpoint', str(output / f'{run_id}.pt')]
                if delivery:
                    if not delivery.get('bundles') or not delivery.get('acceptance'):
                        p.error('Synthetic delivery requires bundles and acceptance')
                    for bundle in delivery['bundles']:
                        command += ['--bundle', str(Path(bundle).expanduser().resolve())]
                    command += ['--acceptance', str(Path(delivery['acceptance']).expanduser().resolve())]
                runs.append(dict(category=category, method=method, seed=seed, status='planned', command=command))
    output.mkdir(parents=True)
    plan = dict(protocol_version=protocol['version'], runs=runs)
    path = output / 'run_plan.json'
    def save():
        path.write_text(json.dumps(plan, indent=2) + '\n')
    save()
    if a.execute:
        for run in runs:
            if run['status'] != 'planned':
                continue
            stem = f"{run['method']}_{run['category']}_seed{run['seed']}"
            run['status'] = 'running'; save()
            with (output / f'{stem}.log').open('x') as log:
                result = subprocess.run(run['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            run['status'] = 'completed' if result.returncode == 0 else 'failed'
            run['exit_code'] = result.returncode; save()
            if result.returncode:
                raise SystemExit(f'Run failed; inspect {stem}.log. Remaining runs stay planned.')
    print(path)


if __name__ == '__main__':
    main()
