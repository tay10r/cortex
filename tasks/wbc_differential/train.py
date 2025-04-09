from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
from loguru import logger


class Tracker:

    @abstractmethod
    def log_param(self, name: str, value: float | int | str):
        raise NotImplementedError()

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def log_model(self, model: nn.Module):
        raise NotImplementedError()

    @abstractmethod
    def on_exit(self):
        raise NotImplementedError()


class TensorboardTracker(Tracker):
    def __init__(self, mode_name: str, start_timestamp: str):
        self.writer = SummaryWriter(log_dir=str(Path('runs') / mode_name / start_timestamp),
                                    flush_secs=5)

    def log_param(self, name: str, value: float | int | str):
        pass

    def log_metrics(self, metrics: dict[str, float], epoch: int):
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, epoch)

    def log_model(self, model: nn.Module):
        pass

    def on_exit(self):
        self.writer.flush()


class MLFlowTracker(Tracker):
    def __init__(self, mode_name: str, start_timestamp: str):
        self.run = mlflow.start_run(run_name=f'{mode_name}-{start_timestamp}')

    def log_param(self, name: str, value: float | int | str):
        mlflow.log_param(name, value)

    def log_metrics(self, metrics: dict[str, float], epoch: int):
        for tag, value in metrics.items():
            mlflow.log_metric(tag, value, step=epoch)

    def log_model(self, model: nn.Module):
        mlflow.pytorch.log_model(model, 'models')

    def on_exit(self):
        mlflow.end_run()


class ConsoleTracker(Tracker):

    def log_param(self, name: str, value: float | int | str):
        logger.info(f'Param: {name}={value}')

    def log_metrics(self, metrics: dict[str, float], epoch: int):
        values = ''
        for tag, value in metrics.items():
            values += f' {tag}={value}'
        logger.info(f'[{epoch}]:{values}')

    def log_model(self, model: nn.Module):
        logger.info('Model updated.')

    def on_exit(self):
        pass


class TrackerFactory:
    def __init__(self, mode_name: str):
        self.start_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.mode_name = mode_name

    def make(self, tracker_name: str) -> Tracker:
        match tracker_name:
            case 'console':
                return ConsoleTracker()
            case 'mlflow':
                return MLFlowTracker(self.mode_name, self.start_timestamp)
            case 'tensorboard':
                return TensorboardTracker(self.mode_name, self.start_timestamp)
        raise RuntimeError(f'Unknown tracker "{tracker_name}"')


class Mode:
    def __init__(self):
        pass

    @abstractmethod
    def load_datasets(self, root: Path) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        raise NotImplementedError()

    @abstractmethod
    def open_model(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def reset_metrics(self):
        raise NotImplementedError()

    @abstractmethod
    def loss(self, predicted: Tensor, target: Tensor):
        raise NotImplementedError()

    @abstractmethod
    def update_metrics(self, predicted: Tensor, target: Tensor):
        raise NotImplementedError()

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        raise NotImplementedError()


def save_onnx(model: nn.Module, filename: str, shape: list[int]):
    dummy_input = torch.randn(*shape)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )


def train(model: nn.Module,
          model_filename: str,
          train_loader: DataLoader,
          val_loader: DataLoader,
          learning_rate: float,
          epochs: int,
          device: str,
          mode: Mode,
          trackers: list[Tracker]):

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    for tracker in trackers:
        tracker.log_param('epochs', epochs)
        tracker.log_param('lr', learning_rate)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = math.inf
    input_shape: list[int] | None = None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            if input_shape is None:
                input_shape = x.shape
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            logits: Tensor = model(x)
            loss: Tensor = mode.loss(predicted=logits, target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = evaluate(model, val_loader, device, mode)
        if loss < best_loss:
            best_loss = loss
            model.to('cpu')
            for tracker in trackers:
                tracker.log_model(model)
            save_onnx(model, model_filename, input_shape)
            model.to(device)

        metrics = mode.get_metrics()
        for tracker in trackers:
            tracker.log_metrics(metrics, epoch)

    for tracker in trackers:
        tracker.on_exit()


def evaluate(model: nn.Module, val_loader: DataLoader, device: str, mode: Mode) -> float:
    model.eval()
    mode.reset_metrics()
    with torch.no_grad():
        for x, y in val_loader:
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            logits: Tensor = model(x)
            mode.update_metrics(predicted=logits, target=y)
    metrics = mode.get_metrics()
    loss: float = metrics['loss']
    return loss
