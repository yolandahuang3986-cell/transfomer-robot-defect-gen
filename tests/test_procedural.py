import numpy as np
import pytest
from src.generation.procedural import ProceduralGenerator
from src.generation.base import validate_batch


def test_reproducibility_and_exact_edit_mask():
    normals=np.random.default_rng(7).random((3,3,48,64),dtype=np.float32)
    ids=["train/0","train/1","train/2"]
    args=dict(category="fixture",source_image_ids=ids)
    gen=ProceduralGenerator(placement="uniform")
    a=gen.generate(normals,7,seed=42,**args)
    b=gen.generate(normals,7,seed=42,**args)
    validate_batch(a,7)
    np.testing.assert_array_equal(a.images,b.images)
    np.testing.assert_array_equal(a.masks,b.masks)
    for image,mask,meta in zip(a.images,a.masks,a.metadata):
        original=normals[ids.index(meta["source_image_id"])]
        np.testing.assert_array_equal(mask[0].astype(bool),np.any(image!=original,axis=0))
        assert 0 < mask.sum() < mask.size
    c=gen.generate(normals,7,seed=43,**args)
    assert not np.array_equal(a.images,c.images)


def test_uniform_images_still_have_edit_and_input_unchanged():
    x=np.ones((1,3,32,32),np.float32)
    a=ProceduralGenerator(placement="uniform").generate(x,2,category="fixture",source_image_ids=["train/0"])
    assert (x==1).all()
    assert (a.masks.sum(axis=(1,2,3))>0).all()


def test_rejects_semantic_claim_and_missing_provenance():
    x=np.zeros((1,3,32,32),np.float32)
    with pytest.raises(ValueError):ProceduralGenerator(placement="uniform").generate(x,1)
    with pytest.raises(ValueError):
        ProceduralGenerator(placement="uniform").generate(x,1,defect_type="scratch",category="fixture",source_image_ids=["train/0"])


@pytest.mark.parametrize("category,background,object_value", [("metal_nut",0.,.8),("screw",.9,.2)])
def test_foreground_placement_stays_inside_object(category, background, object_value):
    x=np.full((1,3,64,64),background,np.float32)
    x[:,:,12:52,20:44]=object_value
    from src.generation.procedural import estimate_foreground
    fg,_=estimate_foreground(x[0],category)
    batch=ProceduralGenerator().generate(x,15,category=category,source_image_ids=[f"{category}/train/good/fixture.png"])
    assert not batch.masks[:,0,:, :][:,~fg].any()
    assert (batch.masks.sum(axis=(1,2,3))>0).all()
    np.testing.assert_array_equal(batch.images[:,:,~fg],np.broadcast_to(x[:,:,~fg],batch.images[:,:,~fg].shape))


def test_foreground_does_not_silently_fallback_on_uniform_image():
    with pytest.raises(ValueError,match="foreground"):
        ProceduralGenerator().generate(np.ones((1,3,64,64),np.float32),1,category="metal_nut",source_image_ids=["metal_nut/train/good/fixture.png"])


def test_screw_dark_core_excludes_cast_shadow():
    # Three-tone scene: pale background, midtone cast shadow, dark metal.
    x=np.full((1,3,96,96),.92,np.float32)
    x[:,:,12:84,25:65]=.52
    x[:,:,16:80,27:43]=.18
    source=x.copy()
    batch=ProceduralGenerator().generate(x,25,category="screw",source_image_ids=["screw/train/good/fixture.png"])
    object_mask=np.zeros((96,96),bool);object_mask[16:80,27:43]=True
    assert not batch.masks[:,0][:,~object_mask].any()
    assert (batch.masks.sum(axis=(1,2,3))>0).all()
    np.testing.assert_array_equal(x,source)
