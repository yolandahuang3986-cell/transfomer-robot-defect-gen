#!/usr/bin/env python3
"""Plot all four frozen E0/E1/E3 run-sets (metal_nut/screw x Procedural/Diffusion)
without rerunning training. Extends plot_e0_e1_metal_nut.py's single-category,
two-condition figure to both categories and both accepted synthetic methods."""
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
CATEGORIES = ('metal_nut', 'screw')
CONDITIONS = ('real_only', 'procedural', 'diffusion')
SOURCE_DIRS = {
    ('real_only', 'metal_nut'): 'results/metrics/e0_20261010',
    ('real_only', 'screw'): 'results/metrics/e0_20261010',
    ('procedural', 'metal_nut'): 'results/metrics/e1_metal_nut_20261017',
    ('procedural', 'screw'): 'results/metrics/e1_screw_20261018',
    ('diffusion', 'metal_nut'): 'results/metrics/e3_metal_nut_20261018',
    ('diffusion', 'screw'): 'results/metrics/e3_screw_20261018',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/analysis/figures')
    args = parser.parse_args()
    sources = []
    rows = {}
    for category in CATEGORIES:
        for condition in CONDITIONS:
            directory = SOURCE_DIRS[condition, category]
            for seed in SEEDS:
                path = ROOT / directory / f'{condition}_{category}_seed{seed}.json'
                result = json.loads(path.read_text())
                assert result['category'] == category and result['seed'] == seed
                assert result['evaluation']['formal_protocol'] == 'v1.0-frozen'
                assert result['provenance']['git_dirty'] is False
                rows[category, condition, seed] = result
                sources.append(dict(file=str(path.relative_to(ROOT)),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for category in CATEGORIES:
        for seed in SEEDS:
            assert (rows[category, 'real_only', seed]['evaluation']
                    == rows[category, 'procedural', seed]['evaluation']
                    == rows[category, 'diffusion', seed]['evaluation'])
        for condition in ('procedural', 'diffusion'):
            assert len({rows[category, condition, seed]['generator']['config']['pool']['pool_sha256']
                        for seed in SEEDS}) == 1

    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'ca6127-e1-e3-all-categories'})
    figure, axes = plt.subplots(2, 4, figsize=(12.4, 5.6), layout='constrained')
    labels = ('Pixel AUROC', 'Dice at threshold 0.5', 'IoU at threshold 0.5',
              'Normal-image false-positive rate')
    limits = ((0, 1.05), (0, .70), (0, .55), (0, 1.08))
    seed_colors = ('#0072B2', '#D55E00', '#009E73')
    x_labels = ['E0\nReal only', 'E1\n+Procedural', 'E3\n+Diffusion']
    for row, category in enumerate(CATEGORIES):
        for col, (metric, label, limit) in enumerate(zip(METRICS, labels, limits)):
            ax = axes[row, col]
            for position, (seed, color) in enumerate(zip(SEEDS, seed_colors)):
                offset = (position - 1) * .035
                values = [rows[category, condition, seed]['inspection'][metric]
                          for condition in CONDITIONS]
                xs = [x + offset for x in range(3)]
                ax.plot(xs, values, marker=('o', 's', '^')[position],
                        color=color, linewidth=1.1, markersize=4, alpha=.85,
                        label=f'Seed {seed}')
            for x, condition in enumerate(CONDITIONS):
                mean = statistics.mean(rows[category, condition, seed]['inspection'][metric]
                                       for seed in SEEDS)
                ax.plot([x - .12, x + .12], [mean, mean], color='black', linewidth=2.0)
            ax.set(xticks=[0, 1, 2], xticklabels=x_labels, ylim=limit, xlim=(-.35, 2.35))
            ax.grid(axis='y', color='#DDDDDD', linewidth=.6)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(category, fontsize=9, fontweight='bold')
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc='outside upper center', ncol=3, frameon=False)
    figure.savefig(args.out / 'e1_e3_all_categories.png', dpi=300)
    svg_path = args.out / 'e1_e3_all_categories.svg'
    figure.savefig(svg_path, metadata={'Date': None})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')

    evidence = dict(categories=list(CATEGORIES), conditions=list(CONDITIONS), seeds=list(SEEDS),
                    sources=sources,
                    uncertainty='Sample SD across downstream seeds; one fixed synthetic pool per method/category',
                    paired_deltas={
                        category: {
                            condition: {metric: {
                                'values': [rows[category, condition, seed]['inspection'][metric]
                                           - rows[category, 'real_only', seed]['inspection'][metric]
                                           for seed in SEEDS],
                                'mean': statistics.mean(
                                    rows[category, condition, seed]['inspection'][metric]
                                    - rows[category, 'real_only', seed]['inspection'][metric]
                                    for seed in SEEDS),
                                'sample_sd': statistics.stdev(
                                    rows[category, condition, seed]['inspection'][metric]
                                    - rows[category, 'real_only', seed]['inspection'][metric]
                                    for seed in SEEDS),
                            } for metric in METRICS}
                            for condition in ('procedural', 'diffusion')
                        } for category in CATEGORIES
                    })
    (args.out.parent / 'e1_e3_all_categories_plot_data.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f"Wrote {args.out / 'e1_e3_all_categories.png'}")


if __name__ == '__main__':
    main()
