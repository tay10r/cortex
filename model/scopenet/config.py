from pathlib import Path
from dataclasses import dataclass
import json
import torch


@dataclass
class Config:
    device: str = ''
    dataset_root: str = ''
    batch_size: int = 16
    num_epochs: int = 100


def open_config(filename: str = 'config.json'):
    p = Path(filename)
    config = Config()
    if p.exists():
        with open(p, 'r') as f:
            config = Config(**json.load(f))
    if config.device == '':
        if torch.cuda.is_available():
            config.device = 'cuda:0'
        else:
            config.device = 'cpu:0'
    return config
