from typing import Callable
import math
from collections import deque
import torch
from torch import Tensor, nn, optim, concat
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb


class TrainingSession:
    def __init__(self,
                 train_data: Dataset,
                 test_data: Dataset,
                 device: torch.device,
                 model_name: str,
                 model_version_major: int = 1,
                 model_version_minor: int = 0,
                 batch_size: int = 16,
                 in_jupyter: bool = False,
                 log_test_output: bool = True):
        self.__device = device
        self.__train_data = DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.__test_data = DataLoader(test_data,
                                      batch_size=batch_size,
                                      shuffle=False)
        self.__epoch = 0
        self.__optimizer: optim.Optimizer | None = None
        self.__best_loss = math.inf
        self.__model_name = model_name
        self.__model_version = f'v{model_version_major}.{model_version_minor}'
        self.__in_jupyter = in_jupyter
        self.__log_test_output = log_test_output

    def run_epochs(self, net: nn.Module, loss_fn: Callable[[Tensor, Tensor], Tensor], num_epochs: int):
        for _ in range(num_epochs):
            self.run_epoch(net, loss_fn)

    def run_epoch(self, net: nn.Module, loss_fn: Callable[[Tensor, Tensor], Tensor]):
        if self.__optimizer is None:
            self.__optimizer = optim.AdamW(net.parameters())
        self._run_train_epoch(net, loss_fn)
        test_loss: float = self._run_test_epoch(net, loss_fn)
        if test_loss < self.__best_loss:
            self.__best_loss = test_loss
            self._save_onnx(net)
        self.__epoch += 1

    def _save_onnx(self, net: nn.Module):
        filename = f'{self.__model_name}-{self.__model_version}.onnx'
        net.to('cpu')
        x = torch.randn((1, 3, 512, 512))
        torch.onnx.export(net,
                          x,
                          filename,
                          input_names=['image'],
                          output_names=['reconstructed'],
                          dynamic_axes={
                              'image': {0: 'batch_size'},
                              'reconstructed': {0: 'batch_size'}
                          },
                          opset_version=11)
        net.to(self.__device)

    def _run_test_epoch(self, net: nn.Module, loss_fn: Callable[[Tensor, Tensor], Tensor]) -> float:
        net.eval()
        loss_list = deque(maxlen=1000)
        loss_sum = 0.0
        loader = None
        if self.__in_jupyter:
            loader = tqdm_nb(self.__test_data)
        else:
            loader = tqdm(self.__test_data)
        k = 0
        for sample in loader:
            x, target = sample
            x: Tensor = x.to(self.__device)
            target: Tensor = target.to(self.__device)
            predicted: Tensor = net(x)
            if self.__log_test_output:
                self._log_output(x, predicted, target, k)
            loss = loss_fn(predicted, target)
            loss_sum += loss.item()
            loss_list.append(loss.item())
            avg_loss = sum(loss_list) / len(loss_list)
            loader.set_description(
                f'Epoch [{self.__epoch}]:  Test Loss: {avg_loss:04}')
            k += 1
        loss_avg = loss_sum / len(loader)
        return loss_avg

    def _log_output(self, input: Tensor, predicted: Tensor, target: Tensor, index: int):
        if index != 4:
            return
        filename = f'{self.__epoch}_{index}.png'
        batch_size = target.shape[0]
        g = make_grid(concat((input, predicted, target), dim=0),
                      nrow=batch_size)
        save_image(g, filename)

    def _run_train_epoch(self, net: nn.Module, loss_fn: Callable[[Tensor, Tensor], Tensor]):
        net.train()
        loss_list = deque(maxlen=1000)
        if self.__in_jupyter:
            loader = tqdm_nb(self.__train_data)
        else:
            loader = tqdm(self.__train_data)
        for sample in loader:
            x, target = sample
            x: Tensor = x.to(self.__device)
            target: Tensor = target.to(self.__device)
            predicted: Tensor = net(x)
            loss = loss_fn(predicted, target)
            loss_list.append(loss.item())
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            avg_loss = sum(loss_list) / len(loss_list)
            loader.set_description(
                f'Epoch [{self.__epoch}]: Train Loss: {avg_loss:04}')
