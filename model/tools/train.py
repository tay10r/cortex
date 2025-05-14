from pathlib import Path
from random import Random
import torch
from torch.nn import functional as F

from loguru import logger

from scopenet.config import open_user_config, open_net_config, UserConfig, NetConfig
from scopenet.net import Net
from scopenet.dataset import Dataset
from scopenet.train import TrainingSession
from scopenet.loss import SSIMLoss

from torch import Tensor, nn
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as FT


class _Transform(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        self.rng = Random(seed)

    def forward(self, x: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if self.rng.randint(0, 1) == 1:
            x = FT.horizontal_flip(x)
            target = FT.horizontal_flip(target)
        if self.rng.randint(0, 1) == 1:
            x = FT.vertical_flip(x)
            target = FT.vertical_flip(target)
        return x, target


def main():
    user_config: UserConfig = open_user_config()
    net_config: NetConfig = open_net_config('config/net.json')
    device = torch.device(user_config.device)
    logger.info(f'Torch Device: {device.type}:{device.index}')
    train_data = Dataset(Path(user_config.dataset_root) /
                         'train', transform=_Transform(seed=0))
    test_data = Dataset(Path(user_config.dataset_root) / 'test')
    preview_data = Dataset(Path(user_config.dataset_root) / 'preview')
    net = Net(in_channels=3, config=net_config)
    net.to(device)
    session = TrainingSession(train_data,
                              test_data,
                              preview_data,
                              device=device,
                              net=net,
                              net_config=net_config,
                              user_config=user_config)
    session.run_epochs(num_epochs=100)


if __name__ == '__main__':
    main()
