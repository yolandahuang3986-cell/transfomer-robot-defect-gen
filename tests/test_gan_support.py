import json
from pathlib import Path
import numpy as np
import pytest
from tests.test_dataset_integrity import _dataset
from src.data.integrity import audit,digest
from scripts.prepare_gan_support import prepare
from src.generation.gan_support import validate_stage
from src.generation.gan_adapter import GANDefectGenerator


def fixture(tmp_path):
    root=tmp_path/'raw';_dataset(root)
    rows,_=audit(root,['metal_nut'])
    defects=[x for x in rows if x['defect_type']=='scratch']
    body=dict(status='frozen',categories=['metal_nut'],n_support_requested_per_defect_type=5,
        train_normal=[x for x in rows if x['split']=='train'],test_normal=[x for x in rows if x['defect_type']=='good' and x['split']=='test'],
        support=defects[:5],heldout=defects[5:])
    m={**body,'manifest_sha256':digest(body)}
    path=tmp_path/'split.json';path.write_text(json.dumps(m))
    path.with_suffix('.freeze.json').write_text(json.dumps(dict(status='frozen',manifest_sha256=m['manifest_sha256'],reviewer='test',reason='fixture')))
    stage=tmp_path/'stage';prepare(root,path,'metal_nut','scratch',stage)
    return root,path,stage


def test_stage_contains_exact_frozen_support_and_masks(tmp_path):
    root,path,stage=fixture(tmp_path)
    record=validate_stage(stage,path,root)
    assert record['support_count']==5
    assert len(list((stage/'train/B').glob('*.png')))==5
    assert not (stage/'test').exists()
    with pytest.raises(FileExistsError):prepare(root,path,'metal_nut','scratch',stage)


def test_stage_rejects_extra_training_image(tmp_path):
    root,path,stage=fixture(tmp_path)
    (stage/'train/B/extra.png').write_bytes(b'extra')
    with pytest.raises(ValueError,match='Unexpected'):validate_stage(stage,path,root)


def test_stage_rejects_changed_mask(tmp_path):
    root,path,stage=fixture(tmp_path)
    next((stage/'train/mask').glob('*.png')).write_bytes(b'changed')
    with pytest.raises(ValueError,match='mask checksum'):validate_stage(stage,path,root)


def test_gan_refuses_smoke_checkpoint_by_default(tmp_path):
    (tmp_path/'training_report.json').write_text(json.dumps({'smoke_only':True}))
    g=GANDefectGenerator('/upstream','/python',tmp_path/'generator_A2B.pt')
    with pytest.raises(ValueError,match='Smoke checkpoint'):
        g.generate(np.zeros((1,3,64,64),np.float32),1)


def test_gan_requires_checkpoint_and_valid_mask_threshold():
    with pytest.raises(ValueError,match='threshold'):GANDefectGenerator(mask_threshold=2)
    with pytest.raises(ValueError,match='checkpoint'):GANDefectGenerator().generate(np.zeros((1,3,64,64),np.float32),1)
