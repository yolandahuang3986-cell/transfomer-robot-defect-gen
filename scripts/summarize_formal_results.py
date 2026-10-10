#!/usr/bin/env python3
"""Summarize complete primary runs and paired differences without imputing failures."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import jsonschema
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.formal import settings

METRICS = ['pixel_auroc', 'dice', 'iou', 'defect_recall', 'image_false_positive_rate']


def summarize(results):
    config = settings()
    schema = json.loads((ROOT / 'schemas/result.schema.json').read_text())
    rows = {}
    splits = set()
    for result in results:
        jsonschema.validate(result, schema)
        cat, method, seed = result['category'], result['method'], result['seed']
        if (result['evaluation'].get('formal_protocol') != config['version']
                or result['evaluation']['split_sha256'] != config['split_sha256']
                or result['evaluation']['smoke_only'] or result['evaluation']['source'] != 'heldout_real'
                or cat not in config['optimizer_steps'] or seed not in config['downstream_seeds']
                or method not in {'real_only', 'procedural', 'gan', 'diffusion'}
                or result['training']['steps'] != config['optimizer_steps'][cat]
                or result['training'].get('sampler') != 'uniform_with_replacement'
                or result['training'].get('batch_size') != 8
                or result['synthetic_count'] != (0 if method == 'real_only' else config['synthetic_counts'][cat])):
            raise ValueError('Not a frozen primary result')
        key = (cat, method, seed)
        if key in rows:
            raise ValueError('Duplicate category/method/seed result')
        rows[key] = result
        splits.add(result['evaluation']['split_sha256'])
    if len(splits) > 1:
        raise ValueError('Cannot mix frozen splits')
    def stats(values):
        return None if any(v is None for v in values) else dict(mean=statistics.mean(values), std=statistics.stdev(values))
    summary = []
    for method in ['real_only', 'procedural', 'gan', 'diffusion']:
        per_category = {}
        for cat in config['optimizer_steps']:
            group = [rows.get((cat, method, seed)) for seed in config['downstream_seeds']]
            if any(r is None for r in group):
                per_category[cat] = {'status': 'incomplete', 'available_seeds': [s for s, r in zip(config['downstream_seeds'], group) if r is not None]}
                continue
            if method != 'real_only' and len({r['generator']['config']['pool']['pool_sha256'] for r in group}) != 1:
                raise ValueError('Training seeds must use the same frozen synthetic pool')
            metrics = {m: stats([r['inspection'].get(m) for r in group]) for m in METRICS}
            paired = {}
            baseline = [rows.get((cat, 'real_only', s)) for s in config['downstream_seeds']]
            if all(r is not None for r in baseline):
                for metric in METRICS:
                    pairs = [(r['inspection'].get(metric), b['inspection'].get(metric)) for r, b in zip(group, baseline)]
                    paired[metric] = stats([None if x is None or y is None else x-y for x,y in pairs])
            per_category[cat] = dict(status='complete', metrics=metrics, paired_delta_vs_real_only=paired)
        macro = None
        if all(r['status'] == 'complete' for r in per_category.values()):
            macro = {}
            for metric in METRICS:
                values = []
                for seed in config['downstream_seeds']:
                    pair = [rows[(cat, method, seed)]['inspection'].get(metric) for cat in config['optimizer_steps']]
                    values.append(None if any(v is None for v in pair) else statistics.mean(pair))
                macro[metric] = stats(values)
        summary.append(dict(method=method, categories=per_category, equal_category_macro=macro))
    return dict(protocol_version=config['version'], uncertainty='sample SD across downstream seeds only; fixed synthetic pool', methods=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', nargs='+')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    report = summarize([json.loads(Path(f).read_text()) for f in a.results])
    with Path(a.out).open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
