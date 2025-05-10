from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset as DatasetBase
from torchvision.transforms.v2 import functional as FT
from torchvision.transforms import v2 as transforms
from PIL import Image

from scopenet.zstack import ZStack


class _Transform(transforms.Transform):
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return x, target


class Dataset(DatasetBase):
    def __init__(self, root: str | Path, transform: transforms.Transform | None = _Transform()):
        if isinstance(root, str):
            root = Path(root)
        self.__transform = transform
        self.__stacks: list[ZStack] = []
        self.__stack_length: None | int = None
        for entry in root.glob('*'):
            if entry.is_dir():
                stack = ZStack(entry)
                if self.__stack_length is None:
                    self.__stack_length = len(stack)
                elif self.__stack_length != len(stack):
                    raise RuntimeError(
                        f'Stack length established as {self.__stack_length} but encountered stack with a length of {len(stack)}')
                self.__stacks.append(stack)

    def __len__(self) -> tuple[int]:
        if self.__stack_length is None:
            return 0
        return self.__stack_length * len(self.__stacks)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        stack_index = index // self.__stack_length
        slice_index = index % self.__stack_length
        stack: ZStack = self.__stacks[stack_index]
        img = Dataset.__image_to_tensor(stack[slice_index])
        target = Dataset.__image_to_tensor(stack[stack.get_best_focus_index()])
        if self.__transform is not None:
            return self.__transform(img, target)
        return img, target

    @staticmethod
    def __image_to_tensor(img: Image.Image) -> Tensor:
        return FT.to_image(img) * (1.0 / 255.0)
