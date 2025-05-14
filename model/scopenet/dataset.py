from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset as DatasetBase
from torchvision.transforms.v2 import functional as FT
from torchvision.transforms import v2 as transforms
import cv2


@dataclass
class _Sample:
    x_path: Path
    y_path: Path
    x: Tensor | None = None
    y: Tensor | None = None


class Dataset(DatasetBase):
    def __init__(self, root: str | Path, transform: transforms.Transform | None = None):
        if isinstance(root, str):
            root = Path(root)
        paths: list[Path] = []
        for entry in root.glob('*.png'):
            paths.append(entry)
        paths.sort()
        self.__samples: list[_Sample] = []
        for i in range(len(paths) // 2):
            self.__samples.append(_Sample(x_path=paths[i * 2 + 0],
                                          y_path=paths[i * 2 + 1]))

    def __len__(self) -> tuple[int]:
        return len(self.__samples)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        s: _Sample = self.__samples[index]
        if s.x is None:
            s.x = Dataset.__image_to_tensor(
                cv2.imread(s.x_path, cv2.IMREAD_UNCHANGED))
        if s.y is None:
            s.y = Dataset.__image_to_tensor(
                cv2.imread(s.y_path, cv2.IMREAD_UNCHANGED))
        return s.x, s.y

    @staticmethod
    def __image_to_tensor(img: np.ndarray) -> Tensor:
        return FT.to_image(img) * (1.0 / 65535.0)
