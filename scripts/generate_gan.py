#!/usr/bin/env python3
"""Generate raw image/mask pairs, retaining smoke status and learned-mask failures."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.integrity import validate_manifest,validate_freeze_attestation,sha256
from src.generation.gan_adapter import GANDefectGenerator

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','manifest','upstream','python','checkpoint','category','defect-type','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--num-samples',type=int,default=20);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--mask-threshold',type=float,default=.2)
    p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args();out=Path(a.out).resolve()
    if out.exists() or out.is_relative_to(ROOT):p.error('Use a new output directory outside Git')
    mp=Path(a.manifest);m=validate_manifest(json.loads(mp.read_text()),a.root)
    validate_freeze_attestation(m,json.loads(mp.with_suffix('.freeze.json').read_text()))
    checkpoint=Path(a.checkpoint);report=json.loads((checkpoint.parent/'training_report.json').read_text())
    if report['split_sha256']!=m['manifest_sha256']:raise ValueError('Checkpoint trained on different split')
    rows=[x for x in m['train_normal'] if x['category']==a.category]
    size=report['image_size'];normal=[]
    for row in rows:
        with Image.open(Path(a.root)/row['image_path']) as im:
            normal.append(np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),np.float32).transpose(2,0,1)/255)
    normals=np.stack(normal);ids=[x['image_path'] for x in rows]
    g=GANDefectGenerator(a.upstream,a.python,a.checkpoint,a.device,a.mask_threshold,a.allow_smoke)
    batch=g.generate(normals,a.num_samples,defect_type=a.defect_type,seed=a.seed,category=a.category,source_image_ids=ids)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'batch.npz',images=batch.images,masks=batch.masks)
    (out/'metadata.json').write_text(json.dumps(batch.metadata,indent=2)+'\n')
    areas=batch.masks.mean(axis=(1,2,3))
    summary=dict(method='gan',num_samples=len(batch.images),category=a.category,defect_type=a.defect_type,
        split_sha256=m['manifest_sha256'],checkpoint_sha256=report['checkpoint_sha256'],upstream_commit=report['upstream_commit'],
        smoke_only=report['smoke_only'],gate1_accepted=False,mask_threshold=a.mask_threshold,
        mask_area_min=float(areas.min()),mask_area_max=float(areas.max()),mask_area_mean=float(areas.mean()),
        empty_masks=int((areas==0).sum()),nearly_full_masks=int((areas>.9).sum()),
        note='Mask statistics are diagnostics, not evidence of real defect quality or downstream utility.',
        code_sha256={x:sha256(ROOT/x) for x in ['scripts/gan_worker.py','src/generation/gan_adapter.py','scripts/generate_gan.py']},
        file_sha256={x:sha256(out/x) for x in ['batch.npz','metadata.json']})
    (out/'provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
    n=min(6,len(batch.images));grid=Image.new('RGB',(size*3,(size+24)*n),'white');d=ImageDraw.Draw(grid)
    for i in range(n):
        source=normals[ids.index(batch.metadata[i]['source_image_id'])]
        panels=[source,batch.images[i],np.repeat(batch.masks[i],3,axis=0)]
        for j,(label,img) in enumerate(zip(['Normal','GAN (smoke)' if report['smoke_only'] else 'GAN','Learned mask'],panels)):
            grid.paste(Image.fromarray(np.rint(img.transpose(1,2,0)*255).astype('uint8')),(j*size,i*(size+24)+24))
            d.text((j*size+5,i*(size+24)+5),label,fill='black')
    grid.save(out/'preview.png')
    print(json.dumps(summary))
