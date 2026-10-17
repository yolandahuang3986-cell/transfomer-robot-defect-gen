#!/usr/bin/env python3
"""Plot the frozen, paired metal_nut E0/E1 results without rerunning training."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
METRICS = ('pixel_auroc', 'dice', 'iou', 'image_false_positive_rate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/analysis/figures')
    args = parser.parse_args()
    sources = []
    rows = {}
    for method, directory in (
        ('real_only', 'results/metrics/e0_20261010'),
        ('procedural', 'results/metrics/e1_metal_nut_20261017'),
    ):
        for seed in SEEDS:
            path = ROOT / directory / f'{method}_metal_nut_seed{seed}.json'
            result = json.loads(path.read_text())
            assert result['category'] == 'metal_nut' and result['seed'] == seed
            assert result['training']['steps'] == 1500
            assert result['evaluation']['formal_protocol'] == 'v1.0-frozen'
            assert result['provenance']['git_dirty'] is False
            rows[method, seed] = result
            sources.append(dict(file=str(path.relative_to(ROOT)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for seed in SEEDS:
        assert rows['real_only', seed]['evaluation'] == rows['procedural', seed]['evaluation']
    assert len({rows['procedural', seed]['generator']['config']['pool']['pool_sha256']
                for seed in SEEDS}) == 1
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'ca6127-e0-e1-metal-nut'})
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 4.2), layout='constrained')
    labels = ('Pixel AUROC', 'Dice at threshold 0.5', 'IoU at threshold 0.5',
              'Normal-image false-positive rate')
    limits = ((0, 1.05), (0, .30), (0, .18), (0, 1.08))
    colors = ('#0072B2', '#D55E00', '#009E73')
    for ax, metric, label, limit in zip(axes.flat, METRICS, labels, limits):
        for position, (seed, color) in enumerate(zip(SEEDS, colors)):
            offset = (position - 1) * .025
            values = [rows[method, seed]['inspection'][metric]
                      for method in ('real_only', 'procedural')]
            ax.plot([offset, 1 + offset], values, marker=('o', 's', '^')[position],
                    color=color, linewidth=1.1, markersize=4.5, alpha=.85,
                    label=f'Seed {seed}')
        for x, method in enumerate(('real_only', 'procedural')):
            mean = statistics.mean(rows[method, seed]['inspection'][metric] for seed in SEEDS)
            ax.plot([x - .10, x + .10], [mean, mean], color='black', linewidth=2.2)
        ax.set(title=label, xticks=[0, 1], xticklabels=['E0  Real only', 'E1  + Procedural'],
               ylim=limit, xlim=(-.3, 1.3))
        ax.grid(axis='y', color='#DDDDDD', linewidth=.6)
        ax.set_axisbelow(True)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc='outside upper center', ncol=3, frameon=False)
    figure.savefig(args.out / 'e0_e1_metal_nut.png', dpi=300)
    svg_path = args.out / 'e0_e1_metal_nut.svg'
    figure.savefig(svg_path, metadata={'Date': None})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    evidence = dict(category='metal_nut', seeds=list(SEEDS), sources=sources,
                    uncertainty='Sample SD across downstream seeds; one fixed synthetic pool',
                    paired_deltas={metric: {
                        'values': [rows['procedural', seed]['inspection'][metric]
                                   - rows['real_only', seed]['inspection'][metric] for seed in SEEDS],
                        'mean': statistics.mean(rows['procedural', seed]['inspection'][metric]
                                                - rows['real_only', seed]['inspection'][metric] for seed in SEEDS),
                        'sample_sd': statistics.stdev(rows['procedural', seed]['inspection'][metric]
                                                      - rows['real_only', seed]['inspection'][metric] for seed in SEEDS)
                    } for metric in METRICS})
    (args.out.parent / 'e0_e1_metal_nut_plot_data.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
