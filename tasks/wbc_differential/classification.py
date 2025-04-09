from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torchvision.datasets import ImageFolder

from train import Mode


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=0)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.norm(self.conv(x)))


class WBCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBlock(3, 16),          # 256 -> 254
            _ConvBlock(16, 32, stride=2),   # 254 -> 252/2 -> 126
            _ConvBlock(32, 64, stride=2),  # 126 -> 124/2 -> 62
            _ConvBlock(64, 128, stride=2),  # 62 -> 60/2 -> 30
            _ConvBlock(128, 16, stride=2),  # 30 -> 28/2 -> 14
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ClassificationMode(Mode):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0
        self.loss_sum = 0.0

    def load_datasets(self, root: Path) -> tuple[Dataset, Dataset]:
        assert isinstance(root, Path)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        train_ds = ImageFolder(str(root / 'train' / 'color'),
                               transform=transform)
        val_ds = ImageFolder(str(root / 'test' / 'color'),
                             transform=transform)
        return train_ds, val_ds

    def open_model(self) -> nn.Module:
        return WBCClassifier()

    def loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(predicted, target)

    def reset_metrics(self):
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0

    def update_metrics(self, predicted: Tensor, target: Tensor):
        self.correct += (predicted.argmax(dim=1) == target).sum().item()
        self.total += target.size(0)
        self.loss_sum += F.cross_entropy(predicted, target).item()

    def get_metrics(self) -> dict[str, float]:
        return {
            'accuracy': self.correct / self.total,
            'loss': self.loss_sum / self.total
        }
