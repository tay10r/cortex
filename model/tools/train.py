from pathlib import Path
from random import Random
import torch
from torch.nn import functional as F

from loguru import logger

from scopenet.config import open_config, Config
from scopenet.net import Net
from scopenet.dataset import Dataset
from scopenet.train import TrainingSession
from scopenet.ssim import SSIMLoss

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
    config: Config = open_config()
    device = torch.device(config.device)
    logger.info(f'Torch Device: {device.type}:{device.index}')
    train_data = Dataset(Path(config.dataset_root) /
                         'train', transform=_Transform(seed=0))
    test_data = Dataset(Path(config.dataset_root) / 'test')
    net = Net(in_channels=3)
    net.to(device)
    session = TrainingSession(train_data,
                              test_data,
                              device=device,
                              model_name='scopenet',
                              batch_size=config.batch_size)
    loss_fn = SSIMLoss()
    for i in range(100):
        session.run_epoch(net, loss_fn)


if __name__ == '__main__':
    main()
