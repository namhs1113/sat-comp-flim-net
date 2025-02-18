
import os
import torch
import numpy as np
from functools import wraps
from abc import ABC, abstractmethod

from typing import Optional
from torch import Tensor
import torch.nn.functional as F
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from util import misc


def make_path(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)


def tensorboard(operate):
    @wraps(operate)
    def _impl(self):
        self.tensorboard = SummaryWriter(os.path.join(self.config.model_path, "logs"))
        operate(self)
        self.tensorboard.close()

    return _impl


def weighted_l1_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    weight_input = torch.mul(input, weight)
    weight_target = torch.mul(target, weight)
    return F.l1_loss(weight_input, weight_target,
                     size_average=size_average, reduce=reduce, reduction=reduction)


def weighted_l2_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    weight_input = torch.mul(input, weight)
    weight_target = torch.mul(target, weight)
    return F.mse_loss(weight_input, weight_target,
                      size_average=size_average, reduce=reduce, reduction=reduction)


class BaseSolver(ABC):
    def __init__(self, config):
        # Get configuration parameters
        self.config = config
        self.config_cur = config

        # Get data loader
        self.image_loader, self.num_images, self.num_steps = dict(), dict(), dict()

        # Model, optimizer, lr_scheduler, criterion
        self.models, self.optimizers, self.lr_schedulers = dict(), dict(), dict()

        # Criterion, loss, metric
        self.criteria = dict()
        self.loss, self.metric = dict(), dict()

        # Training status
        self.phase_types = list()
        self.batch_size = self.config.batch_size
        self.lr = {_type: self.config.lr_opt[_type]["init"] for _type in self.config.lr_opt}
        self.complete_epochs = 0
        self.best_metric, self.best_epoch = 1e10, 0

        # Model and loss types
        self.model_types, self.optimizer_types = list(), list()
        self.loss_types, self.metric_types = list(), list()

        # Tensorboard
        self.tensorboard = None

        # CPU or CUDA
        self.device_ids = self.config.device_ids
        self.device = torch.device("cuda:%d" % self.device_ids[0] if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True if torch.cuda.is_available() else False

    @abstractmethod
    def get_data_loader(self, image_loader):
        pass

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, epoch, valid_metric):

        is_best_epoch = False
        if valid_metric < self.best_metric:
            self.best_metric = valid_metric
            self.best_epoch = epoch + 1
            is_best_epoch = True

        checkpoint = {"config": self.config,
                      "lr": self.lr,
                      "model_types": self.model_types,
                      "optimizer_types": self.optimizer_types,
                      "loss_types": self.loss_types,
                      "complete_epochs": epoch,
                      "best_metric": self.best_metric,
                      "best_epoch": self.best_epoch}
        model_state_dicts = {"model_%s_state_dict" % model_type:
                             self.models[model_type].state_dict() for model_type in self.model_types}
        optimizer_state_dicts = \
            {"optimizer_%s_state_dict" % optimizer_type:
             self.optimizers[optimizer_type].state_dict() for optimizer_type in self.optimizer_types}
        lr_scheduler_state_dicts = \
            {"lr_scheduler_%s_state_dict" % optimizer_type:
             self.lr_schedulers[optimizer_type].state_dict() for optimizer_type in self.optimizer_types}
        checkpoint = dict(checkpoint, **model_state_dicts)
        checkpoint = dict(checkpoint, **optimizer_state_dicts)
        checkpoint = dict(checkpoint, **lr_scheduler_state_dicts)

        torch.save(checkpoint, os.path.join(self.config.model_path,
                                            self.config.model_name.replace(".pth",
                                                                           "_best.pth" if is_best_epoch else ".pth")))
        if is_best_epoch:
            print("Best model (%.3f) is saved to %s" % (self.best_metric, self.config.model_path))

    def load_model(self, best=True, model_only=False):
        model_full_path = os.path.join(self.config.model_path,
                                       self.config.model_name.replace(".pth",
                                                                      ".pth" if not best else "_best.pth"))
        if os.path.isfile(model_full_path):
            checkpoint = torch.load(model_full_path)

            if not model_only:
                self.config = checkpoint["config"]
                self.lr = checkpoint["lr"]
                self.model_types = checkpoint["model_types"]
                self.optimizer_types = checkpoint["optimizer_types"]
                self.loss_types = checkpoint["loss_types"]
                self.complete_epochs = checkpoint["complete_epochs"]
                self.best_metric = checkpoint["best_metric"]
                self.best_epoch = checkpoint["best_epoch"]

            self.build_model()
            self.load_model_state_dict(checkpoint)
        else:
            self.build_model()

    def load_model_state_dict(self, checkpoint):
        for _type in self.model_types:
            self.models[_type].load_state_dict(checkpoint["model_%s_state_dict" % _type])
        for _type in self.optimizer_types:
            self.optimizers[_type].load_state_dict(checkpoint["optimizer_%s_state_dict" % _type])
            self.lr_schedulers[_type].load_state_dict(checkpoint["lr_scheduler_%s_state_dict" % _type])

    def set_train(self, is_train=True):
        for model_type in self.model_types:
            if is_train:
                self.models[model_type].train(True)
            else:
                self.models[model_type].eval()

    def print_info(self, phase="train", print_func=None, epoch=0, step=0, time=0):
        # Assert
        assert(phase in self.phase_types)

        # Print process information
        total_epoch = self.config.num_epochs
        total_step = self.num_steps[phase]
        eta_sec = int(time / (step + 1) * (total_step - step - 1))
        eta_min = int(eta_sec / 60)
        eta_sec = eta_sec % 60
        ela_min = int(time / 60)
        ela_sec = time % 60

        prefix = "\033[91m" + "[Epoch %4d / %4d]\033[0m lr " % (epoch, total_epoch)
        for _type in self.lr:
            prefix = prefix + "%s %.1e " % (_type[0], self.lr[_type])

        suffix = "(%02d:%02d>%02d:%02d) " % (eta_min, eta_sec, ela_min, ela_sec) if phase == "train" else ""
        suffix += "\033[31m" + ("[%s] " % phase) + "\033[0m"
        for loss_type in self.loss_types:
            suffix += "%s: %.5f / " % (loss_type, misc.average(self.loss[loss_type][phase]))
        for metric_type in self.metric_types:
            suffix += "%s: %.5f / " % (metric_type, misc.average(self.metric[metric_type][phase]))
        if print_func is not None:
            print_func(step + 1, total_step, prefix=prefix, suffix=suffix, dec=1, bar_len=20)
        else:
            print(suffix, end="")

        if (step + 1 == total_step) or (phase == "valid"):
            print("")

    def log_to_tensorboard(self, epoch, elapsed_time=None):
        if elapsed_time is not None:
            self.tensorboard.add_scalar("elapsed_time", elapsed_time, epoch)
        self.tensorboard.add_scalars("learning_rate", {_type: self.lr[_type] for _type in self.lr}, epoch)
        for loss_type in self.loss_types:
            self.tensorboard.add_scalars("%s" % loss_type, {phase: misc.average(self.loss[loss_type][phase])
                                                            for phase in self.phase_types}, epoch)
        for metric_type in self.metric_types:
            self.tensorboard.add_scalars("%s" % metric_type, {phase: misc.average(self.metric[metric_type][phase])
                                                              for phase in self.phase_types}, epoch)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def d_backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def g_backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *arg, **kwargs):
        pass

    @abstractmethod
    def calculate_metric(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
