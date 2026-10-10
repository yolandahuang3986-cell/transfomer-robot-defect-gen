"""Portable B-side bundle reader. No change to A's GeneratedBatch contract."""
import json
from pathlib import Path
import numpy as np
from .base import GeneratedBatch, validate_batch
from src.data.integrity import sha256


def load_bundle(directory, manifest, category):
    directory=Path(directory)
    provenance=json.loads((directory/'provenance.json').read_text())
    if provenance['split_sha256'] != manifest['manifest_sha256'] or provenance['category'] != category:
        raise ValueError('Bundle split/category mismatch')
    for name in ('batch.npz','metadata.json'):
        if sha256(directory/name) != provenance['file_sha256'][name]:
            raise ValueError('Bundle content checksum mismatch')
    with np.load(directory/'batch.npz',allow_pickle=False) as arrays:
        batch=GeneratedBatch(arrays['images'],arrays['masks'],json.loads((directory/'metadata.json').read_text()))
    validate_batch(batch,provenance['num_samples'])
    allowed={x['image_path'] for x in manifest['train_normal'] if x['category']==category}
    for row in batch.metadata:
        if row['category']!=category or row['method']!='procedural':
            raise ValueError('Unexpected bundle category/method')
        if row['source_image_id'] not in allowed or row['generation_config']['donor_image_id'] not in allowed:
            raise ValueError('Synthetic source/donor is not an allowed normal training image')
    return batch
