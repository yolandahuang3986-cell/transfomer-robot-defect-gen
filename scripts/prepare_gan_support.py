#!/usr/bin/env python3
"""Stage ONLY frozen normal-training/support rows for upstream GAN training."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.integrity import validate_manifest,validate_freeze_attestation,sha256


def prepare(root, manifest_path, category, defect_type, out):
    root,manifest_path,out=Path(root),Path(manifest_path),Path(out).resolve()
    if out.is_relative_to(ROOT):raise ValueError('Keep image data outside code repository')
    manifest=validate_manifest(json.loads(manifest_path.read_text()),root)
    validate_freeze_attestation(manifest,json.loads(manifest_path.with_suffix('.freeze.json').read_text()))
    normals=[r for r in manifest['train_normal'] if r['category']==category]
    support=[r for r in manifest['support'] if r['category']==category and r['defect_type']==defect_type]
    expected=manifest['n_support_requested_per_defect_type']
    if not normals or len(support)!=expected:raise ValueError('Missing category/defect or wrong shot count')
    if out.exists():raise FileExistsError(out)
    for d in ['train/A','train/B','train/mask']:(out/d).mkdir(parents=True)
    mapping=[]
    for domain,rows in [('A',normals),('B',support)]:
        for i,row in enumerate(rows):
            target=f'train/{domain}/{i:05d}.png'
            shutil.copyfile(root/row['image_path'],out/target)
            if sha256(out/target)!=row['image_sha256']:raise ValueError('Copy checksum mismatch')
            record=dict(source_image_id=row['image_path'],image_sha256=row['image_sha256'],target=target)
            if domain=='B':
                mask_target=f'train/mask/{i:05d}_mask.png'
                shutil.copyfile(root/row['mask_path'],out/mask_target)
                if sha256(out/mask_target)!=row['mask_sha256']:raise ValueError('Mask copy checksum mismatch')
                record.update(source_mask_id=row['mask_path'],mask_target=mask_target,mask_sha256=row['mask_sha256'])
            mapping.append(record)
    forbidden={r['image_path'] for part in ['heldout','test_normal'] for r in manifest[part]}
    assert not forbidden.intersection(r['source_image_id'] for r in mapping)
    report=dict(category=category,defect_type=defect_type,normal_count=len(normals),support_count=len(support),
        split_sha256=manifest['manifest_sha256'],heldout_copied=0,mapping=mapping,
        note='Training-only stage. No upstream 80/20 split; no test folders; inference uses an explicit allowed normal input list.')
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','manifest','category','defect-type','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args(); result=prepare(a.root,a.manifest,a.category,a.defect_type,a.out)
    print(json.dumps({k:v for k,v in result.items() if k!='mapping'}))
