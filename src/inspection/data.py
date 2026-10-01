"""Load standardized outputs and portable real-data manifest rows."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset, Dataset
from src.generation.base import validate_batch


def generated_dataset(batch):
    validate_batch(batch, len(batch.images))
    return TensorDataset(torch.from_numpy(batch.images), torch.from_numpy(batch.masks))


class RealDataset(Dataset):
    def __init__(self, root, rows, size=64):
        self.root, self.rows, self.size = Path(root), rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row['image_path']) as image:
            x = np.asarray(image.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        if row['mask_path']:
            with Image.open(self.root / row['mask_path']) as mask:
                y = (np.asarray(mask.convert('L').resize((self.size, self.size), Image.Resampling.NEAREST)) > 0).astype(np.float32)
        else:
            y = np.zeros((self.size, self.size), dtype=np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y[None].copy())
