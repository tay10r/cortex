from pathlib import Path
import io
import math

from PIL import Image


class ZStack:
    def __init__(self, root: str | Path):
        if isinstance(root, str):
            root = Path(root)
        self.__buffers: list[bytes] = []
        self.__paths: list[Path] = []
        self.__best_index = 0
        for path in root.glob('*'):
            if path.suffix in ['.png', '.jpeg', '.jpg', '.bmp']:
                self.__paths.append(path)
                self.__buffers.append(b'')
        self.__paths.sort()
        for i in range(len(self.__paths)):
            args = str(self.__paths[i].stem).split('-')
            if 'best' in args:
                self.__best_index = i
                break

    def get_slice_focus(self, index: int) -> float:
        delta = self.__best_index - index
        sigma = 5
        return math.exp(-(delta * delta) / 2 * (sigma * sigma))

    def get_best_focus_index(self) -> int:
        return self.__best_index

    def __getitem__(self, index: int) -> Image:
        if len(self.__buffers[index]) == 0:
            with open(self.__paths[index], 'rb') as f:
                self.__buffers[index] = f.read()
        stream = io.BytesIO(self.__buffers[index])
        return Image.open(stream)

    def __len__(self) -> int:
        return len(self.__paths)
