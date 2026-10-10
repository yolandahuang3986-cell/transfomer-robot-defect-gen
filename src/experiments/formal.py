"""Frozen formal-evaluation settings and deterministic optimization budgets."""
from pathlib import Path
import yaml
from torch.utils.data import DataLoader, RandomSampler
import torch

ROOT = Path(__file__).resolve().parents[2]


def settings():
    return yaml.safe_load((ROOT / 'configs/experiments.yaml').read_text())['formal_protocol']


def fixed_step_loader(dataset, steps, batch_size, seed, num_workers=0, pin_memory=False):
    if steps < 1 or batch_size < 1 or not len(dataset):
        raise ValueError('Nonempty data and positive step/batch budgets required')
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=steps * batch_size,
                            generator=generator)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                      num_workers=num_workers, pin_memory=pin_memory,
                      generator=torch.Generator().manual_seed(seed))
