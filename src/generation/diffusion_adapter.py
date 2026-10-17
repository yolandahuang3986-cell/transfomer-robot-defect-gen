"""Subprocess adapter for the pinned AnomalyDiffusion implementation (Member C).

The upstream model runs in its own Python 3.8 environment through
scripts/diffusion/ad_worker.py. This adapter never imports upstream code; it
writes a request JSON, runs the worker, and reads back PNG image/mask pairs.

Diffusion-internal rules (configs/diffusion/*.json) run inside the worker before
shared QC. Their rejections are reported in `generation_config.stage1` and in
`self.runs`, separately from shared QC. Each sample's generation_seconds includes
its share of every rejected stage-1 attempt.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .base import DefectGenerator, GeneratedBatch, validate_batch
from .diffusion_foreground import placement_foreground
from src.experiments.scope import filter_in_scope, require_in_scope

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'scripts/diffusion/ad_worker.py'
SUPPORT_STATS = ROOT / 'scripts/diffusion/support_stats.py'
UPSTREAM_COMMIT = 'b85297f'
GRID = 256


class DiffusionUnderfilledError(RuntimeError):
    """The worker could not produce the requested count within its stage-1 budget."""

    def __init__(self, message, run_dir):
        super().__init__(message)
        self.run_dir = Path(run_dir)


def _sha256(path, cache={}):
    path = Path(path).resolve()
    key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    if key not in cache:
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                digest.update(chunk)
        cache[key] = digest.hexdigest()
    return cache[key]


def _rules(path, expected_prefix):
    raw = Path(path).read_bytes()
    rules = json.loads(raw)
    if not rules.get('rules_version', '').startswith(expected_prefix):
        raise ValueError(f'{path} is not a {expected_prefix} rule file')
    if 'DRAFT' in rules.get('status', ''):
        raise ValueError(f'{path} is still a DRAFT; freeze it before generating')
    return rules, hashlib.sha256(raw).hexdigest()


class DiffusionDefectGenerator(DefectGenerator):
    """config keys (see configs/diffusion/generation.json):
    upstream, python, root, manifest (validated dict), checkpoints{ldm, anomaly_embeddings,
    spatial_encoder, mask_embeddings_template}, sampling{...}, mask_filter_rules,
    placement_rules, work_dir, save_rejects, training_provenance."""

    def __init__(self, config, manifest):
        self.config = config
        self.manifest = manifest
        self.upstream = Path(os.path.expanduser(config['upstream'])).resolve()
        self.python = os.path.expanduser(config['python'])
        self.root = Path(os.path.expanduser(config['root'])).resolve()
        self.work_dir = Path(os.path.expanduser(config['work_dir'])).resolve()
        if self.work_dir.is_relative_to(ROOT):
            raise ValueError('Keep raw diffusion outputs outside Git')
        self.mask_rules, self.mask_rules_sha = _rules(ROOT / config['mask_filter_rules'], 'mask_filter_')
        self.place_rules, self.place_rules_sha = _rules(ROOT / config['placement_rules'], 'placement_')
        self.placement = self.place_rules['rules_version'].split('_')[-1]
        self.fg_variant = self.place_rules.get('foreground_variant', 'qc')
        self.runs = []
        self._support_cache = {}

    def fit(self, dataset, config):
        raise NotImplementedError('AnomalyDiffusion is trained with upstream scripts and SUPPORT_LIST; '
                                  'see external/anomalydiffusion/README.md')

    # ---------- helpers ----------
    def _support_rows(self, category, defect_type):
        rows = [r for r in filter_in_scope(self.manifest['support'])
                if r['category'] == category and r['defect_type'] == defect_type]
        if len(rows) != 5:
            raise ValueError(f'Expected 5 frozen support masks for {category}/{defect_type}, got {len(rows)}')
        return rows

    def _support_stats(self, category, defect_type, run_dir):
        key = (category, defect_type)
        if key not in self._support_cache:
            rows = self._support_rows(category, defect_type)
            path = run_dir / 'support_stats_request.json'
            path.write_text(json.dumps({'root': str(self.root), 'masks': [r['mask_path'] for r in rows]}))
            out = subprocess.run([self.python, str(SUPPORT_STATS), '--request', str(path)],
                                 capture_output=True, text=True, check=True)
            stats = json.loads(out.stdout.strip().splitlines()[-1])
            stats['support_mask_paths'] = [r['mask_path'] for r in rows]
            self._support_cache[key] = stats
        return self._support_cache[key]

    def _mask_filter(self, stats):
        r = self.mask_rules
        window = r['area_window']
        f = {'rules_version': r['rules_version'], 'rules_file': self.config['mask_filter_rules'],
             'rules_sha256': self.mask_rules_sha, 'support_area_pct_256': stats['area_pct_256'],
             'min_area_pct': round(min(stats['area_pct_256']) * window['min_factor_of_smallest_support'], 4),
             'max_area_pct': round(max(stats['area_pct_256']) * window['max_factor_of_largest_support'], 4),
             'max_area_abs_pct': r.get('max_area_abs_pct', 20.0),
             'max_border_frac': r['max_border_frac'], 'max_bg_median': r['max_bg_median']}
        if 'min_latent_cells' in r:
            f['support_latent_cells'] = stats['latent_cells']
            f['min_latent_cells'] = max(int(r['min_latent_cells']['floor']), min(stats['latent_cells']))
        return f

    def _checkpoints(self, category, defect_type):
        c = self.config['checkpoints']
        paths = {'ldm': c['ldm'], 'anomaly_embeddings': c['anomaly_embeddings'],
                 'spatial_encoder': c['spatial_encoder'],
                 'mask_embeddings': c['mask_embeddings_template'].format(category=category, defect_type=defect_type)}
        paths = {k: str(Path(os.path.expanduser(v)).resolve()) for k, v in paths.items()}
        for k, v in paths.items():
            if not Path(v).is_file():
                raise FileNotFoundError(f'Missing {k} checkpoint: {v}')
        return paths

    # ---------- contract ----------
    def generate(self, normal_images, num_samples, defect_type=None, seed=42, **kwargs):
        category = kwargs.get('category')
        ids = kwargs.get('source_image_ids')
        if not isinstance(category, str) or not isinstance(defect_type, str):
            raise ValueError('Diffusion generation needs an explicit category and defect_type')
        require_in_scope(category, defect_type)
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError('num_samples must be a positive integer')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        images = np.asarray(normal_images)
        if images.ndim != 4 or images.shape[1] != 3 or len(images) == 0:
            raise ValueError('Expected nonempty NCHW normal images')
        if ids is None or len(ids) != len(images):
            raise ValueError('One source_image_id per normal image is required')
        allowed = {r['image_path'] for r in self.manifest['train_normal'] if r['category'] == category}
        for sid in ids:
            p = PurePosixPath(sid)
            if sid not in allowed or p.parts[:3] != (category, 'train', 'good'):
                raise ValueError(f'Source {sid!r} is not a frozen {category}/train/good image')

        run_dir = self.work_dir / f'{category}_{defect_type}' / f'seed{seed}'
        run_dir.mkdir(parents=True, exist_ok=False)
        rng = np.random.default_rng(seed)
        selection = np.resize(rng.permutation(len(ids)), num_samples)
        chosen = [ids[int(i)] for i in selection]

        fg_dir = run_dir / 'foregrounds'
        fg_dir.mkdir()
        samples = []
        fg_cache = {}
        for index, sid in enumerate(chosen):
            if sid not in fg_cache:
                with Image.open(self.root / sid) as im:
                    native = np.asarray(im.convert('RGB'), dtype=np.float32).transpose(2, 0, 1) / 255
                fg = placement_foreground(native, self.fg_variant, GRID)
                path = fg_dir / (PurePosixPath(sid).stem + '.png')
                Image.fromarray(fg.astype(np.uint8) * 255).save(path)
                fg_cache[sid] = str(path)
            samples.append({'index': index, 'source_image_id': sid, 'foreground_png': fg_cache[sid]})

        stats = self._support_stats(category, defect_type, run_dir)
        ckpt = self._checkpoints(category, defect_type)
        sampling = dict(self.config['sampling'])
        sampling['mask_filter'] = self._mask_filter(stats)
        request = {'upstream_dir': str(self.upstream), 'mvtec_root': str(self.root),
                   'category': category, 'defect_type': defect_type, 'ckpt': ckpt,
                   'sampling': sampling, 'seed': seed, 'placement': self.placement,
                   'placement_rules': {**self.place_rules, 'rules_file': self.config['placement_rules'],
                                       'rules_sha256': self.place_rules_sha},
                   'samples': samples, 'out_dir': str(run_dir),
                   'save_rejects': bool(self.config.get('save_rejects', True))}
        (run_dir / 'request.json').write_text(json.dumps(request, indent=2))

        start = time.perf_counter()
        with open(run_dir / 'worker.log', 'x') as log:
            code = subprocess.run([self.python, str(WORKER), str(run_dir / 'request.json')],
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.perf_counter() - start
        result_path = run_dir / 'result.json'
        if code not in (0, 2) or not result_path.exists():
            raise RuntimeError(f'Diffusion worker failed (exit {code}); see {run_dir / "worker.log"}')
        result = json.loads(result_path.read_text())
        stage1 = {'run_dir': str(run_dir), 'status': result.get('status', 'ok'),
                  'requested': num_samples, 'produced': result.get('n', 0),
                  'mask_attempts': result['mask_attempts'],
                  'upstream_check_mask_rejected': result.get('stage1_upstream_check_mask_rejected'),
                  'internal_rejected': (result.get('diffusion_internal_filter') or {}).get('rejected'),
                  'mask_sampling_seconds_total': result.get('mask_sampling_seconds_total'),
                  'model_load_seconds': (result.get('mask_model_load_seconds') or 0)
                                        + (result.get('image_model_load_seconds') or 0),
                  'worker_wall_seconds': wall}
        self.runs.append(stage1)
        if result.get('status') != 'ok' or result.get('n') != num_samples:
            raise DiffusionUnderfilledError(
                f'Diffusion stage 1 underfilled: {result.get("n", 0)}/{num_samples} after '
                f'{result["mask_attempts"]} mask attempts', run_dir)

        records = {int(r['index']): r for r in result['samples']}
        provenance = result.get('provenance', {})
        out_images, out_masks, metadata = [], [], []
        for s in samples:
            i = s['index']
            rgb = np.asarray(Image.open(run_dir / 'images' / f'{i:05d}.png').convert('RGB'), dtype=np.float32) / 255
            mask = (np.asarray(Image.open(run_dir / 'masks' / f'{i:05d}.png').convert('L')) > 127).astype(np.float32)
            out_images.append(rgb.transpose(2, 0, 1))
            out_masks.append(mask[None])
            r = records[i]
            metadata.append(dict(
                method='diffusion', category=category, defect_type=defect_type, seed=seed,
                source_image_id=s['source_image_id'], generation_seconds=float(r['generation_seconds']),
                generation_config=dict(
                    upstream_commit=UPSTREAM_COMMIT, worker_sha256=provenance.get('worker_sha256'),
                    checkpoints=ckpt, checkpoint_sha256={k: _sha256(v) for k, v in ckpt.items()},
                    sampling={k: v for k, v in sampling.items() if k != 'mask_filter'},
                    mask_filter=sampling['mask_filter'],
                    placement=self.placement, placement_rules_sha256=self.place_rules_sha,
                    placement_foreground=self.fg_variant,
                    sample_stage1={k: r[k] for k in ('mask_attempt', 'placement_tries', 'placement_overlap',
                                                     'latent_cells', 'mask_seconds', 'inpaint_seconds') if k in r},
                    stage1=stage1, training_provenance=self.config.get('training_provenance'),
                    mask_area_fraction=float(mask.mean()), raw_output_dir=str(run_dir))))
        batch = GeneratedBatch(np.stack(out_images).astype(np.float32), np.stack(out_masks).astype(np.float32), metadata)
        return validate_batch(batch, num_samples)
