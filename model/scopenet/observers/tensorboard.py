from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from scopenet.observer import Observer


class TensorboardLogger(Observer):
    def __init__(self, hparams: dict, metrics: list):
        self.writer = SummaryWriter()
        self.hparams = hparams
        self.metrics: dict[str, float] = {}
        for metric in metrics:
            self.metrics[metric] = 0.0

    def on_scalar_metric(self, name: str, value: float, epoch: int):
        self.writer.add_scalar(name, value, epoch)
        if name in self.metrics:
            self.metrics[name] = value

    def on_preview(self, image: Tensor, epoch):
        self.writer.add_image('Preview', image, epoch)

    def on_model_update(self, net: nn.Module, epoch: int):
        pass

    def on_complete(self):
        self.writer.add_hparams(self.hparams, self.metrics)
