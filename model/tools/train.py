from pathlib import Path
import torch
from torch.nn import functional as F

from loguru import logger

from scopenet.config import open_config, Config
from scopenet.net import Net
from scopenet.dataset import Dataset
from scopenet.train import TrainingSession
from scopenet.ssim import SSIMLoss


def main():
    config: Config = open_config()
    device = torch.device(config.device)
    logger.info(f'Torch Device: {device.type}:{device.index}')
    train_data = Dataset(Path(config.dataset_root) / 'train')
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
