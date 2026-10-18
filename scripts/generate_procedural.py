#!/usr/bin/env python3
"""Export a reproducible procedural batch from frozen normal-training rows only."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.integrity import validate_manifest, validate_freeze_attestation, sha256
from src.generation.procedural import ProceduralGenerator, estimate_foreground
from src.generation.bundle import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--category',choices=['metal_nut','screw'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--num-samples',type=int,default=30)
    p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--purpose',choices=['validation','formal'],default='validation')
    p.add_argument('--shared-qc',action='store_true',help='Require A shared QC (generic synthetic_patch integration gate)')
    a=p.parse_args()
    if a.purpose=='formal' and not a.shared_qc:p.error('Formal generation requires --shared-qc')
    if a.image_size<32 or a.num_samples<1:p.error('Invalid image size or sample count')
    out=Path(a.out).resolve()
    if out.is_relative_to(ROOT):p.error('Keep generated image bundles outside the code repository')
    if out.exists():p.error('Output directory already exists; choose a new run name')
    manifest_path=Path(a.manifest)
    manifest=validate_manifest(json.loads(manifest_path.read_text()),a.root)
    validate_freeze_attestation(manifest,json.loads(manifest_path.with_suffix('.freeze.json').read_text()))
    rows=[r for r in manifest['train_normal'] if r['category']==a.category]
    if not rows:raise ValueError('No allowed normal training images')
    images=[]
    for row in rows:
        with Image.open(Path(a.root)/row['image_path']) as im:
            images.append(np.asarray(im.convert('RGB').resize((a.image_size,a.image_size),Image.Resampling.BILINEAR),np.float32).transpose(2,0,1)/255)
    normals=np.stack(images)
    start=time.perf_counter()
    qc_report=None
    generator=ProceduralGenerator()
    if a.shared_qc:
        result=generator.generate_with_qc(normals,[r['image_path'] for r in rows],target_count=a.num_samples,
            defect_type='synthetic_patch',seed=a.seed,category=a.category)
        qc_report=result.report.to_dict()
        if not result.report.target_met:
            out.mkdir(parents=True,exist_ok=False)
            (out/'qc_report.json').write_text(json.dumps(qc_report,indent=2))
        batch=result.require_target()
    else:
        batch=generator.generate(normals,a.num_samples,seed=a.seed,category=a.category,
            source_image_ids=[r['image_path'] for r in rows])
    generation_seconds=time.perf_counter()-start
    out.mkdir(parents=True,exist_ok=False)
    if qc_report is not None:(out/'qc_report.json').write_text(json.dumps(qc_report,indent=2))
    np.savez_compressed(out/'batch.npz',images=batch.images,masks=batch.masks)
    (out/'metadata.json').write_text(json.dumps(batch.metadata,indent=2,allow_nan=False)+'\n')
    provenance=dict(purpose=a.purpose,category=a.category,num_samples=a.num_samples,image_size=a.image_size,seed=a.seed,
        split_sha256=manifest['manifest_sha256'],source_partition='train_normal',normal_source_count=len(rows),
        code_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        code_dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),
        source_sha256={f:sha256(ROOT/f) for f in ['src/generation/procedural.py','scripts/generate_procedural.py','src/generation/bundle.py']},
        python=platform.python_version(),numpy=np.__version__,generation_seconds=generation_seconds,
        file_sha256={f:sha256(out/f) for f in ['batch.npz','metadata.json']},
        shared_qc=qc_report,benchmark=False,limitations=['foreground is a category-specific intensity heuristic','synthetic_patch is not a semantic defect type','no downstream improvement claimed'])
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    load_bundle(out,manifest,a.category)
    ids=[r['image_path'] for r in rows]
    count=min(8,len(batch.images)); size=a.image_size
    grid=Image.new('RGB',(size*4,(size+24)*count),'white'); draw=ImageDraw.Draw(grid)
    for i in range(count):
        source=normals[ids.index(batch.metadata[i]['source_image_id'])]
        fg,_=estimate_foreground(source,a.category)
        overlay=batch.images[i].transpose(1,2,0).copy()
        selected=batch.masks[i,0].astype(bool)
        overlay[selected]=.5*overlay[selected]+.5*np.array([1,0,0])
        panels=[source.transpose(1,2,0),np.repeat(fg[:,:,None],3,axis=2),batch.images[i].transpose(1,2,0),overlay]
        for j,(title,panel) in enumerate(zip(['Normal','Estimated object','Synthetic','Mask overlay'],panels)):
            grid.paste(Image.fromarray(np.rint(panel*255).astype(np.uint8)),(j*size,i*(size+24)+24))
            draw.text((j*size+5,i*(size+24)+5),title,fill='black')
    grid.save(out/'preview.png')
    print(json.dumps({'output':str(out),'num_samples':len(batch.images),'generation_seconds':generation_seconds,'bundle_validation':'passed'}))


if __name__=='__main__':main()
