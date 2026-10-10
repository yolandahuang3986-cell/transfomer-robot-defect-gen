"""Subprocess adapter for the pinned official SyntheticDefectGeneration model."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from .base import DefectGenerator, GeneratedBatch, validate_batch
from src.data.integrity import sha256
from .gan_support import validate_stage

ROOT=Path(__file__).resolve().parents[2]


class GANDefectGenerator(DefectGenerator):
    def __init__(self, upstream=None, python=None, checkpoint=None, device='cpu', mask_threshold=.2, allow_smoke=False):
        self.upstream=upstream
        self.python=python
        self.checkpoint=Path(checkpoint) if checkpoint else None
        self.device=device
        if not 0<mask_threshold<1:raise ValueError('Mask threshold must be between zero and one')
        self.mask_threshold=mask_threshold
        self.allow_smoke=allow_smoke

    def fit(self, dataset, config):
        # dataset is the explicit training-only stage, with root and frozen manifest in config.
        stage=validate_stage(dataset,config['manifest'],config['root'])
        if not self.upstream or not self.python:raise ValueError('Explicit upstream and isolated Python are required')
        run=Path(config['out']).resolve()
        if run.is_relative_to(ROOT):raise ValueError('Keep GAN weights outside Git')
        run.mkdir(parents=True,exist_ok=False)
        request=dict(operation='train',upstream=str(Path(self.upstream).resolve()),stage=str(Path(dataset).resolve()),
            run_dir=str(run),device=self.device,seed=config.get('seed',42),image_size=config.get('image_size',256),
            epochs=config.get('epochs',2),decay_epochs=config.get('decay_epochs',1),
            max_steps=config.get('max_steps',0),learning_rate=config.get('learning_rate',.0002),num_threads=config.get('num_threads',2))
        path=run/'request.json';path.write_text(json.dumps(request,indent=2))
        with (run/'worker.log').open('x') as log:
            subprocess.run([str(self.python),str(ROOT/'scripts/gan_worker.py'),'--request',str(path)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        self.checkpoint=run/'generator_A2B.pt'
        return self

    def generate(self, normal_images, num_samples, defect_type=None, seed=42, **kwargs):
        if self.checkpoint is None or not self.python or not self.upstream:
            raise ValueError('Configure a trained checkpoint, upstream and isolated Python first')
        report=json.loads((self.checkpoint.parent/'training_report.json').read_text())
        if report['smoke_only'] and not self.allow_smoke:
            raise ValueError('Smoke checkpoint cannot be used as a formal generator')
        if sha256(self.checkpoint)!=report['checkpoint_sha256']:raise ValueError('Checkpoint checksum mismatch')
        category=kwargs.get('category');ids=kwargs.get('source_image_ids')
        if category!=report['category'] or defect_type!=report['defect_type']:
            raise ValueError('Category/defect must match the trained checkpoint')
        a=np.asarray(normal_images)
        if type(num_samples)is not int or num_samples<1:raise ValueError('Positive sample count required')
        if a.dtype!=np.float32 or a.ndim!=4 or a.shape[0]<1 or a.shape[1]!=3:
            raise ValueError('Expected nonempty float32 NCHW normal images')
        if a.shape[2:]!=(report['image_size'],report['image_size']):raise ValueError('Input resolution differs from checkpoint')
        if not np.isfinite(a).all() or not ((a>=0)&(a<=1)).all():raise ValueError('Invalid image range')
        if ids is None or len(ids)!=len(a) or any(not isinstance(x,str) or not x for x in ids):raise ValueError('Source IDs required')
        rng=np.random.default_rng(seed)
        # Deterministic I2I network: diversity comes from source normals, not latent noise.
        order=rng.permutation(len(a));selection=np.resize(order,num_samples)
        start=time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='gan-infer-') as temporary:
            temp=Path(temporary)
            np.savez_compressed(temp/'input.npz',images=a[selection])
            req=dict(operation='infer',upstream=str(Path(self.upstream).resolve()),checkpoint=str(self.checkpoint.resolve()),
                input=str(temp/'input.npz'),output=str(temp/'output.npz'),device=self.device,seed=seed)
            (temp/'request.json').write_text(json.dumps(req))
            result=subprocess.run([str(self.python),str(ROOT/'scripts/gan_worker.py'),'--request',str(temp/'request.json')],capture_output=True,text=True)
            if result.returncode:raise RuntimeError('GAN inference failed: '+result.stderr[-4000:])
            inference=json.loads(result.stdout.strip().splitlines()[-1])
            with np.load(temp/'output.npz',allow_pickle=False) as f:
                images=f['images'];probabilities=f['probabilities']
        if not np.isfinite(probabilities).all() or not ((probabilities>=0)&(probabilities<=1)).all():raise ValueError('Invalid generated mask probabilities')
        masks=(probabilities>=self.mask_threshold).astype(np.float32)
        elapsed=time.perf_counter()-start
        metadata=[dict(method='gan',category=category,defect_type=defect_type,seed=seed,source_image_id=ids[int(index)],
            generation_seconds=inference['inference_seconds']/num_samples,
            generation_config=dict(upstream_commit=report['upstream_commit'],checkpoint_sha256=report['checkpoint_sha256'],
                split_sha256=report['split_sha256'],mask_threshold=self.mask_threshold,mask_source='learned_sigmoid',
                smoke_only=report['smoke_only'],training_steps=report['steps'],
                batch_wall_seconds_including_model_load=elapsed,diversity='different normal inputs; repeated input is deterministic',
                mask_area_fraction=float(masks[i].mean()))) for i,index in enumerate(selection)]
        return validate_batch(GeneratedBatch(images,masks,metadata),num_samples)
