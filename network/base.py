
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np


def init_weights(net, init_type="kaiming", init_gain=0.02):
    """ Initialize network weights.
    Parameters:
        net (network): network to be initialized
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print("[%s] Initialize network with %s." % (net.__name__(), init_type))
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, device_ids=None):
    """ Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network): the network to be initialized
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
        device_ids (int list): which GPUs the network runs on: e.g., 0, 1, 2
    Return an initialized network.
    """
    if torch.cuda.is_available():
        if device_ids is None:
            device_ids = []
        net.to(device_ids[0])
        if len(device_ids) > 1:
            net = nn.DataParallel(net, device_ids=device_ids)  # multi-GPUs
    else:
        net.to("cpu")
    # init_weights(net, init_type=init_type, init_gain=init_gain)  # Weight initialization

    return net


def set_requires_grad(nets, requires_grad=False):
    """ Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list): a list of networks
        requires_grad (bool): whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class GANLoss(nn.Module):
    """ Define various GAN objectives. """
    def __init__(self, gan_mode):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str): the type of GAN objectives (vanilla, lsgan, wgangp, hinge)
        Note: the discriminator should not return the sigmoid output.
        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        elif gan_mode in ["hinge"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def __call__(self, prediction, is_target_real):
        """ Calculate loss given dicriminator's output and ground truth labels.
        Parameters:
            prediction (tensor): discriminator's output
            is_target_real (bool): real or fake?
        Returns:
            the calculated loss
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            label = torch.ones if is_target_real else torch.zeros
            target_tensor = label(prediction.shape, dtype=prediction.dtype).to(prediction.device)
            loss = self.loss(prediction, target_tensor)
            loss = loss.mean()
        elif self.gan_mode in ["wgangp"]:
            loss = -prediction.mean() if is_target_real else prediction.mean()
        elif self.gan_mode in ["hinge"]:
            label = 1.0 if is_target_real else -1.0
            loss = F.relu(1.0 - label * prediction).mean()
        else:
            raise NotImplementedError("gan mode is not supported")

        return loss