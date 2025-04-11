from pathlib import Path

from torch import nn, Tensor, concat
from torch.nn import functional as F
from torch.utils.data import Dataset as DatasetBase
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as FT

from train import Mode


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.norm(self.conv(x)))


class WBCLocalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _ConvBlock(3, 16),          # 512 -> 510
            _ConvBlock(16, 128, stride=2),   # 510 -> 508/2 -> 254
            _ConvBlock(128, 256, stride=2),   # 254 -> 252/2 -> 126
            _ConvBlock(256, 512, stride=2),  # 126 -> 124/2 -> 62
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Dataset(DatasetBase):
    def __init__(self,
                 path: Path,
                 image_size: tuple[int, int],
                 transform: nn.Module | None):
        super().__init__()
        self.images = self.__load_images(path / 'color', '.jpg', image_size)
        self.masks = self.__load_images(path / 'mask', '.png', image_size)
        for i in range(len(self.masks)):
            self.masks[i] = resize_target(self.masks[i])
        self.transform = transform

    def __len__(self) -> int:
        assert len(self.images) == len(self.masks)
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.transform is not None:
            return self.transform(self.images[idx], self.masks[idx])
        return self.images[idx], self.masks[idx]

    @staticmethod
    def __load_images(path: Path, ext: str, size: tuple[int, int]) -> list[Tensor]:
        paths: list[str] = []
        for entry in path.glob(f'*/*{ext}'):
            paths.append(str(entry))
        paths.sort()
        images: list[Tensor] = []
        for entry in paths:
            img = read_image(entry)
            img = FT.resize(img, size)
            img = FT.to_dtype(img, scale=True)
            images.append(img)
        return images


def resize_target(mask: Tensor) -> Tensor:
    return FT.resize(mask, (64, 64))
    """
    This functions resizes the mask.
    How the mask is resized depends on the convolution operators in the network.
    If the network is updated, this function has to be updated as well.
    """
    # conv2d(kernel_size=3, stride=1)
    mask = mask[:, 1:-1, 1:-1]
    # conv2d(kernel_size=3, stride=2)
    mask = mask[:, 1:-1, 1:-1]
    mask = FT.resize(mask, (mask.shape[1] // 2, mask.shape[2] // 2))
    # conv2d(kernel_size=3, stride=2)
    mask = mask[:, 1:-1, 1:-1]
    mask = FT.resize(mask, (mask.shape[1] // 2, mask.shape[2] // 2))
    # conv2d(kernel_size=3, stride=2)
    mask = mask[:, 1:-1, 1:-1]
    mask = FT.resize(mask, (mask.shape[1] // 2, mask.shape[2] // 2))
    # conv2d(kernel_size=3, stride=1)
    mask = mask[:, 1:-1, 1:-1]
    return mask


class LocalizationMode(Mode):
    def __init__(self):
        self.loss_sum = 0.0
        self.total = 0
        self.example: Tensor | None = None

    def load_datasets(self, root: Path) -> tuple[Dataset, Dataset]:
        train_transform = transforms.Compose([
            transforms.Identity()
        ])
        image_size = (512, 512)
        train_ds = Dataset(root / 'train',
                           image_size,
                           transform=train_transform)
        val_ds = Dataset(root / 'test',
                         image_size,
                         transform=None)
        return train_ds, val_ds

    def open_model(self) -> nn.Module:
        return WBCLocalizer()

    def loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(predicted, target)

    def reset_metrics(self):
        self.loss_sum = 0.0
        self.total = 0
        self.example = None

    def update_metrics(self, predicted: Tensor, target: Tensor):
        self.loss_sum += F.binary_cross_entropy(predicted, target).item()
        self.total += target.shape[0]
        if self.example is None:
            self.example = make_grid(concat((predicted, target)),
                                     nrow=predicted.shape[0])

    def get_metrics(self) -> dict[str, float]:
        return {
            'loss': self.loss_sum / self.total,
            'example': self.example
        }
