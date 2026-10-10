#!/usr/bin/env python3
"""Subprocess boundary avoids collision between upstream src and project src.

Training uses upstream CycleGAN_Mask loss/update, architecture and augmentation.
No upstream evaluation against the official test pool is invoked.
"""
import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

PIN = '5a8099204bc0858da314f8e0b5e8b6770c298e98'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    request = json.loads(Path(parser.parse_args().request).read_text())
    upstream = Path(request['upstream']).resolve()
    actual = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if actual != PIN or subprocess.check_output(['git','-C',str(upstream),'diff','HEAD','--'],text=True):
        raise ValueError('Expected clean pinned upstream commit '+PIN)
    sys.path.insert(0,str(upstream))
    import numpy as np
    import torch
    torch.set_num_threads(request.get('num_threads',2))
    seed = request['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = request['device']
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device is unavailable')
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    if request['operation']=='infer':
        from src.models_mask import ResNetGenerator9
        network=ResNetGenerator9(3,3,64,norm_layer=functools.partial(torch.nn.BatchNorm2d,affine=True,track_running_stats=True),n_blocks=9).to(device)
        state=torch.load(request['checkpoint'],map_location=device,weights_only=True)
        network.load_state_dict(state,strict=True);network.eval()
        with np.load(request['input'],allow_pickle=False) as f:images=f['images']
        generated=[];masks=[];start=time.perf_counter()
        with torch.no_grad():
            for image in images:
                x=torch.from_numpy(image[None]).to(device)*2-1
                rgb,mask,_=network(x)
                generated.append(((rgb[0]+1)/2).clamp(0,1).cpu().numpy())
                masks.append(mask[0,None].cpu().numpy())
        if device=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        np.savez_compressed(request['output'],images=np.stack(generated).astype(np.float32),probabilities=np.stack(masks).astype(np.float32))
        print(json.dumps({'inference_seconds':seconds,'num_samples':len(images)}),flush=True)
        return

    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2
    from src.train_options import TrainOptions
    from src.cycle_gan_mask import CycleGAN_Mask
    from src.utils_datasets import ImageDataset
    options = TrainOptions().initialize(argparse.ArgumentParser()).parse_args([])
    options.cuda=device=='cuda'
    options.image_size=request['image_size']
    options.epochs=request['epochs'];options.decay_epochs=request['decay_epochs']
    options.lr=request.get('learning_rate',.0002)
    if options.image_size<64 or options.image_size%4 or not 0<=options.decay_epochs<options.epochs:
        raise ValueError('Require image size >=64 divisible by 4 and decay before final epoch')
    transform=A.Compose([
        A.Resize(int(options.image_size*1.12),int(options.image_size*1.12),interpolation=cv2.INTER_CUBIC),
        A.RandomCrop(options.image_size,options.image_size,p=1),A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5),
        A.ElasticTransform(alpha=1,sigma=2,alpha_affine=.5,interpolation=cv2.INTER_CUBIC,border_mode=cv2.BORDER_REFLECT,
            value=None,mask_value=0,always_apply=False,approximate=False,same_dxdy=False,p=.5),
        A.Normalize(mean=(.5,.5,.5),std=(.5,.5,.5)),ToTensorV2()
    ],additional_targets={'imageB':'image','maskB':'mask'})
    dataset=ImageDataset(request['stage'],transform=transform,unaligned=True,mask=True)
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(seed))
    model=CycleGAN_Mask(options)
    model.create_network();model.define_losses();model.set_optimizer();model.set_decayLR()
    run=Path(request['run_dir'])
    start=time.perf_counter();step=0;last={}
    max_steps=request.get('max_steps',0)
    with (run/'losses.jsonl').open('x') as log:
        for epoch in range(1,options.epochs+1):
            for data in loader:
                inputs=model.prepare_inputs((data,{'real_label':1,'fake_label':0}))
                output,losses=model.compute_loss_and_update(inputs)
                if not all(torch.isfinite(x).all() for x in losses.values()):raise ValueError('Nonfinite GAN loss')
                step+=1
                last={k:float(v.detach().cpu()) for k,v in losses.items()}
                record={'epoch':epoch,'step':step,'losses':last}
                log.write(json.dumps(record)+'\n');log.flush()
                print(json.dumps({'step':step,'epoch':epoch,'generator_loss':last['errG']}),flush=True)
                if max_steps and step>=max_steps:break
            model.update_learning_rates()
            if max_steps and step>=max_steps:break
    if device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    checkpoint=run/'generator_A2B.pt'
    with checkpoint.open('xb') as f:torch.save(model.netG_A2B.state_dict(),f)
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    stage=json.loads((Path(request['stage'])/'provenance.json').read_text())
    report=dict(upstream_commit=PIN,category=stage['category'],defect_type=stage['defect_type'],
        split_sha256=stage['split_sha256'],support_count=stage['support_count'],normal_count=stage['normal_count'],
        image_size=options.image_size,ngf=64,n_blocks=9,seed=seed,device=device,steps=step,
        epochs_requested=options.epochs,epochs_completed=epoch if not max_steps else None,
        truncated_by_step_limit=bool(max_steps),smoke_only=bool(max_steps),training_seconds=seconds,
        last_losses=last,checkpoint_sha256=checkpoint_hash,python=platform.python_version(),
        packages={x:importlib.metadata.version(x) for x in ['torch','torchvision','numpy','albumentations','opencv-python-headless','einops','torchsummary']},
        gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        options=vars(options),source='official losses and networks, project-owned loop; batch size 1',
        checkpoint_purpose='generator inference only; not resumable optimizer state',
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage_provenance_sha256=hashlib.sha256((Path(request['stage'])/'provenance.json').read_bytes()).hexdigest())
    (run/'training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'completed','steps':step,'training_seconds':seconds,'smoke_only':bool(max_steps)}),flush=True)


if __name__=='__main__':main()
