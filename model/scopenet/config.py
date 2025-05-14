from pathlib import Path
from dataclasses import dataclass
import json
import torch
import kagglehub


@dataclass
class UserConfig:
    device: str = ''
    dataset_root: str = ''
    num_epochs: int = 100
    use_tensorboard: bool = True


@dataclass
class NetConfig:
    batch_size: int = 16
    lr: float = 0.002
    lr_gamma: float = 1.0
    weight_multiplier: int = 2
    num_encoder_res_blocks: int = 0
    num_decoder_res_blocks: int = 0
    num_bottleneck_res_blocks: int = 4
    final_activation: bool = False
    se_enabled: bool = True
    loss: str = 'l1'
    adversarial: bool = True
    optimizer: str = 'adam_w'


def open_net_config(filename: str) -> NetConfig:
    p = Path(filename)
    config = NetConfig()
    if p.exists():
        with open(p, 'r') as f:
            config = NetConfig(**json.load(f))
    return config


def open_user_config(filename: str = 'config/user.json'):
    p = Path(filename)
    config = UserConfig()
    if p.exists():
        with open(p, 'r') as f:
            config = UserConfig(**json.load(f))

    # If no torch device specified, choose CUDA if available.
    if config.device == '':
        if torch.cuda.is_available():
            config.device = 'cuda:0'
        else:
            config.device = 'cpu:0'

    # If no dataset root specified, download the one we host on Kaggle.
    if config.dataset_root == '':
        config.dataset_root = kagglehub.dataset_download("tay10r/scopenet")

    return config
