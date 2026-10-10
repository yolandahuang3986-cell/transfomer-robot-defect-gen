import json
import numpy as np
import pytest
from src.data.integrity import sha256
from src.generation.procedural import ProceduralGenerator
from src.generation.bundle import load_bundle


def bundle(tmp_path):
    x=np.zeros((1,3,32,32),np.float32)
    b=ProceduralGenerator(placement='uniform').generate(x,2,category='metal_nut',source_image_ids=['metal_nut/train/good/0.png'])
    np.savez_compressed(tmp_path/'batch.npz',images=b.images,masks=b.masks)
    (tmp_path/'metadata.json').write_text(json.dumps(b.metadata))
    p=dict(split_sha256='frozen',category='metal_nut',num_samples=2,file_sha256={f:sha256(tmp_path/f) for f in ['batch.npz','metadata.json']})
    (tmp_path/'provenance.json').write_text(json.dumps(p))
    m=dict(manifest_sha256='frozen',train_normal=[dict(category='metal_nut',image_path='metal_nut/train/good/0.png')])
    return m,p


def test_bundle_roundtrip_and_split_rejection(tmp_path):
    m,_=bundle(tmp_path)
    assert len(load_bundle(tmp_path,m,'metal_nut').images)==2
    with pytest.raises(ValueError,match='split/category'):
        load_bundle(tmp_path,{**m,'manifest_sha256':'different'},'metal_nut')


def test_bundle_rejects_corruption(tmp_path):
    m,_=bundle(tmp_path)
    with (tmp_path/'batch.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):load_bundle(tmp_path,m,'metal_nut')


def test_bundle_rejects_test_source_even_with_updated_checksum(tmp_path):
    m,p=bundle(tmp_path)
    records=json.loads((tmp_path/'metadata.json').read_text())
    records[0]['generation_config']['donor_image_id']='metal_nut/test/scratch/0.png'
    (tmp_path/'metadata.json').write_text(json.dumps(records))
    p['file_sha256']['metadata.json']=sha256(tmp_path/'metadata.json')
    (tmp_path/'provenance.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='allowed normal'):load_bundle(tmp_path,m,'metal_nut')
