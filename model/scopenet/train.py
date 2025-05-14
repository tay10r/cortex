import math
from dataclasses import asdict

import torch
from torch import Tensor, nn, optim, concat, unsqueeze
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

from scopenet.observer import Observer
from scopenet.config import UserConfig, NetConfig
from scopenet.observers.tensorboard import TensorboardLogger
from scopenet.loss import loss_functions, PatchDiscriminator
from scopenet.optim import optimizers


class TrainingSession:
    """
    This class is responsible for training the model, given a user-defined configuration.
    The functionality can be extended by adding what are called "observers" to the training
    session. Observers will be called under certain conditions, such as to report the test loss
    or to report a preview of the model output.
    """

    def __init__(self,
                 train_data: Dataset,
                 test_data: Dataset,
                 preview_data: Dataset,
                 device: torch.device,
                 net: nn.Module,
                 net_config: NetConfig,
                 user_config: UserConfig):
        self.__device = device
        self.__train_data = DataLoader(train_data,
                                       batch_size=net_config.batch_size,
                                       shuffle=True)
        self.__test_data = DataLoader(test_data,
                                      batch_size=net_config.batch_size,
                                      shuffle=False)
        self.__preview_data = preview_data
        self.__epoch = 0
        self.__optimizer: optim.Optimizer = optimizers[net_config.optimizer](net.parameters(),
                                                                             net_config.lr)
        self.__lr_schedule = StepLR(self.__optimizer,
                                    step_size=1,
                                    gamma=net_config.lr_gamma)
        self.__best_loss = math.inf
        self.__net = net
        self.__observers: list[Observer] = []
        self.__loss = loss_functions[net_config.loss]
        if net_config.adversarial:
            self.__disc = PatchDiscriminator(in_channels=3).to(self.__device)
            self.__disc_optim = optim.Adam(self.__disc.parameters(),
                                           lr=net_config.lr * 0.25)
        else:
            self.__disc = None
        if user_config.use_tensorboard:
            hparams = asdict(net_config)
            self.__observers.append(TensorboardLogger(hparams, ['loss/test']))

    def add_observer(self, observer: Observer):
        self.__observers.append(observer)

    def run_epochs(self, num_epochs: int):
        for _ in range(num_epochs):
            self._run_epoch()
        self._complete()

    def _run_epoch(self):
        self._run_train_epoch()
        test_loss: float = self._run_test_epoch()
        if test_loss < self.__best_loss:
            self.__best_loss = test_loss
            for observer in self.__observers:
                observer.on_model_update(self.__net, self.__epoch)
        self._run_preview()
        self.__epoch += 1

    def _complete(self):
        for observer in self.__observers:
            observer.on_complete()

    def _run_preview(self):
        results = torch.zeros((0, 3, 256, 256))
        for x, _ in self.__preview_data:
            results = concat((results, unsqueeze(x, dim=0)))
        for x, _ in self.__preview_data:
            y: Tensor = self.__net(unsqueeze(x, dim=0).to(self.__device))
            results = concat((results, y.cpu()))
        for _, target in self.__preview_data:
            results = concat((results, unsqueeze(target, dim=0)))
        g = make_grid(results, nrow=len(self.__preview_data))
        for observer in self.__observers:
            observer.on_preview(g, self.__epoch)

    def _run_test_epoch(self) -> float:
        self.__net.eval()
        loss_sum = 0.0
        for sample in self.__test_data:
            x, target = sample
            x: Tensor = x.to(self.__device)
            target: Tensor = target.to(self.__device)
            predicted: Tensor = self.__net(x)
            loss: Tensor = self.__loss(predicted, target)
            loss_sum += loss.item()
        loss_avg = loss_sum / len(self.__test_data)
        for observer in self.__observers:
            observer.on_scalar_metric('loss/test', loss_avg, self.__epoch)
        return loss_avg

    def _run_train_epoch(self):
        self.__net.train()
        if self.__disc is not None:
            self.__disc.train()
        loss_sum = 0.0
        for sample in self.__train_data:
            x, target = sample
            x: Tensor = x.to(self.__device)
            target: Tensor = target.to(self.__device)
            output: Tensor = self.__net(x)
            loss: Tensor = self.__loss(output, target)

            # if adversarial training is enabled,
            # train discriminator and generator
            if self.__disc is not None:
                # train discriminator
                pred_real = self.__disc(target)
                pred_fake = self.__disc(output.detach())
                loss_d = F.binary_cross_entropy_with_logits(pred_real,
                                                            torch.ones_like(pred_real))
                loss_d += F.binary_cross_entropy_with_logits(pred_fake,
                                                             torch.zeros_like(pred_fake))
                self.__disc_optim.zero_grad()
                loss_d.backward()
                self.__disc_optim.step()
                # train generator
                pred_fake = self.__disc(output)
                adv_loss = F.binary_cross_entropy_with_logits(pred_fake,
                                                              torch.ones_like(pred_fake))
                loss += adv_loss * 0.01

            loss_sum += loss.item()
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
        loss_avg = loss_sum / len(self.__train_data)
        self.__lr_schedule.step()
        for observer in self.__observers:
            observer.on_scalar_metric('loss/train', loss_avg, self.__epoch)
            observer.on_scalar_metric('lr',
                                      self.__lr_schedule.get_last_lr()[0],
                                      self.__epoch)
