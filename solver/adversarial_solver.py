
import os
import csv
import numpy as np
import time

import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from .base_solver import BaseSolver, tensorboard, weighted_l1_loss, make_path
from dataset import data_loader
from network import base, rdn
from util import misc, metric


class AdversarialSolver(BaseSolver):
    def __init__(self, config):
        super(AdversarialSolver, self).__init__(config)

        # Member variables for data
        self.inputs, self.targets, self.masks = None, None, None
        self.weights, self.outputs = None, None
        self.phase_types = ["train", "valid"] if self.config.phase == "train" else ["test"]
        self.model_types = ["gen", "dsc"]
        self.optimizer_types = self.model_types
        self.loss_types = ["d", "d_real", "d_fake", "g", "l1", "lb"]
        self.metric_types = ["mse"]

        # Get data loader & build new model or load existing one
        self.load_model(best=False)  # if self.config.phase == "train" else True)
        self.get_data_loader(data_loader.get_loader)

    def get_data_loader(self, image_loader):
        # Get data loader
        for phase in self.phase_types:
            self.image_loader[phase] = image_loader(dataset_path=self.config_cur.dataset_path,
                                                    phase=phase,
                                                    shuffle=True if phase == "train" else False,
                                                    batch_size=self.batch_size if phase != "test" else 128,
                                                    num_workers=self.config.num_workers)
            self.num_images[phase] = int(self.image_loader[phase].dataset.__len__())
            self.num_steps[phase] = int(np.ceil(self.num_images[phase] / self.batch_size))

    def build_model(self):
        # Build model
        self.models["gen"] = rdn.RDN(n_in_ch=self.config.num_input_ch,
                                     n_out_ch=self.config.num_output_ch,
                                     act=nn.LeakyReLU(0.2), k_size=3,
                                     G0=32, G=16, D=8, C=3)
        self.models["dsc"] = rdn.SN_PatchGAN_D(in_channels=3,
                                               num_features=64)

        # Build optimizer & learning rate scheduler
        for model_type in self.model_types:
            self.optimizers[model_type] = optim.Adam(self.models[model_type].parameters(),
                                                     lr=self.config.lr_opt[model_type]["init"],
                                                     betas=(0.5, 0.999),
                                                     weight_decay=self.config.l2_penalty)
            self.lr_schedulers[model_type] = optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[model_type],
                                                     mode="min",
                                                     factor=self.config.lr_opt[model_type]["gamma"],
                                                     patience=self.config.lr_opt[model_type]["patience"],
                                                     min_lr=self.config.lr_opt[model_type]["term"],
                                                     verbose=True)

        # Build criterion
        self.criteria["l1"] = weighted_l1_loss
        self.criteria["hinge"] = base.GANLoss("hinge")
        self.criteria["wgangp"] = base.GANLoss("wgangp")

        # Model initialization
        for model_type in self.model_types:
            self.models[model_type] = base.init_net(self.models[model_type],
                                                    init_type="kaiming", init_gain=0.02,
                                                    device_ids=self.device_ids)

    def forward(self, inputs, targets, masks):
        # Image to device
        self.inputs = inputs.to(self.device)  # n2hw
        self.targets = targets.to(self.device) if targets is not None else None  # n1hw
        self.masks = masks.to(self.device) if masks is not None else None  # n1hw

        # Generation (forward)
        self.outputs = self.models["gen"](self.inputs)

    def d_backward(self, phase="train"):
        # Backward to calculate the gradient by discriminator
        # Discrimination of real case
        output_real = self.models["dsc"](torch.cat((self.inputs, self.targets), dim=1))

        # Discrimination of fake case
        output_fake = self.models["dsc"](torch.cat((self.inputs, self.outputs), dim=1))

        # Discriminator loss defition
        d_loss_real = self.criteria["hinge"](output_real, True)  # validity map
        d_loss_fake = self.criteria["hinge"](output_fake, False)  # validity map

        # Loss integration and gradient calculation (backward)
        loss = self.config.adv_weight * (d_loss_real + d_loss_fake) / 2
        if phase == "train":
            loss.backward(retain_graph=True)

        self.loss["d"][phase].append(loss.detach())
        self.loss["d_real"][phase].append(d_loss_real.detach())
        self.loss["d_fake"][phase].append(d_loss_fake.detach())

    def g_backward(self, phase="train"):
        # Backward to calculate the gradient by generator
        # Discrimination of fake case (fooling discriminator)
        output_fake = self.models["dsc"](torch.cat((self.inputs, self.outputs), dim=1))

        # Generator loss definition
        g_loss = self.config.adv_weight * self.criteria["wgangp"](output_fake, True)

        # L1 similarity loss definition (for refine final output)
        sat_mask = self.inputs[:, 1:2]
        weights = sat_mask * (self.config.satcomp_weight - 1) + 1
        weights[self.masks == 1] = 0
        l1_loss = self.criteria["l1"](self.outputs, self.targets, weights)

        # Constraint for original saturation points
        input = self.inputs[:, 0:1]
        weights = (sat_mask - self.masks) == 1.0
        comp_ps = (self.outputs[weights] - input[weights]).mean()
        lb_constraint = torch.exp(-comp_ps / 0.5).to(self.device)

        # Loss integration and gradient calculation (backward)
        loss = g_loss + self.config.l1_weight * l1_loss + self.config.lb_weight * lb_constraint
        if phase == "train":
            loss.backward()

        self.loss["g"][phase].append(g_loss.detach())
        self.loss["l1"][phase].append(l1_loss.detach())
        self.loss["lb"][phase].append(lb_constraint.detach())

    def optimize(self, optimizer_type, backward):
        """ Optimize and update weights according to the calculated gradients. """
        assert (optimizer_type in self.optimizer_types)

        for model_type in self.model_types:
            base.set_requires_grad(self.models[model_type], optimizer_type == model_type)

        self.optimizers[optimizer_type].zero_grad()
        backward()
        self.optimizers[optimizer_type].step()

    def evaluate(self):
        pass

    def calculate_metric(self, phase="train"):
        assert (phase in self.phase_types)
        weights = self.inputs[:, 1:2] - self.masks
        self.metric["mse"][phase].append(metric.get_mse(self.outputs, self.targets, weights))

    @tensorboard
    def train(self):
        for epoch in range(self.complete_epochs, self.config.num_epochs):
            # ============================= Training ============================= #
            # ==================================================================== #
            # Training status parameters
            t0 = time.time()
            self.loss = {loss_type: {"train": list(), "valid": list()} for loss_type in self.loss_types}
            self.metric = {metric_type: {"train": list(), "valid": list()} for metric_type in self.metric_types}

            # Image generating for training process
            self.set_train(is_train=True)
            for i, (inputs, targets, masks, _, _) in enumerate(self.image_loader["train"]):

                # Forward
                self.forward(inputs, targets, masks)

                # Backward & Optimize
                self.optimize("dsc", self.d_backward)
                self.optimize("gen", self.g_backward)

                # Calculate evaluation metrics
                self.calculate_metric()

                # Print training info
                self.print_info(phase="train", print_func=misc.print_progress_bar,
                                epoch=epoch + 1, step=i, time=time.time() - t0)

            # ============================ Validation ============================ #
            # ==================================================================== #
            # Image generating for validation process
            with torch.no_grad():
                self.set_train(is_train=False)
                for i, (inputs, targets, masks, _, _) in enumerate(self.image_loader["valid"]):
                    # Forward
                    self.forward(inputs, targets, masks)

                    # Backward
                    self.d_backward(phase="valid")
                    self.g_backward(phase="valid")

                    # Calculate evaluation metrics
                    self.calculate_metric(phase="valid")

            # Print validation info
            self.print_info(phase="valid")

            # Tensorboard logs
            self.log_to_tensorboard(epoch + 1, elapsed_time=time.time() - t0)

            # ============================ Model Save ============================ #
            # ==================================================================== #
            # Best validation metric logging & model save
            valid_metric = misc.average(self.metric["mse"]["valid"]).item()
            self.save_model(epoch, valid_metric)

            # Learning rate adjustment
            early_stop = True
            for _type in self.optimizer_types:
                self.lr_schedulers[_type].step(valid_metric)
                self.lr[_type] = self.lr_schedulers[_type].optimizer.param_groups[0]["lr"]
                early_stop &= (self.lr[_type] == self.config.lr_opt[_type]["term"])
                early_stop &= (self.lr_schedulers[_type].num_bad_epochs == self.config.lr_opt[_type]["patience"])

            # Check early stopping
            if early_stop:
                print("The model seems to be sufficiently converged."
                      "The training session shall be early stopped.")
                break

    def test(self):
        # Make dir first
        test_data_name = "solution"
        model_path = self.config.model_path.split('/')[-2]
        make_path(test_data_name)
        make_path(test_data_name + "/" + model_path)
        make_path(test_data_name + "/" + model_path + "/output")

        # =============================== Test =============================== #
        # ==================================================================== #
        # Image generating for test process
        with torch.no_grad():
            self.set_train(is_train=False)
            for i, (inputs, data_paths) in enumerate(self.image_loader["test"]):
                print(data_paths[0] + f" and others... ")

                # Forward
                self.forward(inputs, None, None)

                # Save result
                for b, (input, output) in enumerate(zip(self.inputs, self.outputs)):
                    dname = data_paths[b].split('/')[-1]
                    output.detach().cpu().numpy().tofile(test_data_name + "/" + model_path + "/output/" + dname)

        # # =============================== Test =============================== #
        # # ==================================================================== #
        # # Image generating for test process  # i b c s
        # etimes = list()
        # with torch.no_grad():
        #     result = list()
        #     self.set_train(is_train=False)
        #     for i, (inputs, targets, masks, data_paths, roi_ranges) in enumerate(self.image_loader["test"]):
        #         print(data_paths[0] + f" and others... ", end="")
        #
        #         # Forward
        #         t0 = time.time()
        #         self.forward(inputs, targets, masks)
        #         etime = (time.time() - t0) / inputs.size(0) / inputs.size(2) * 1000.0 * 1000.0
        #         print(f"(avg etime: {etime:.3f} usec/pulse)")
        #         etimes.append(etime)
        #
        #         # Find saturation ratio & RMSE
        #         for b, (input, target, output, roi_range) in enumerate(zip(self.inputs, self.targets, self.outputs, roi_ranges)):
        #             roi_range = [range(int(start), int(end)) for start, end in roi_range]
        #             sat = ((target[0] - input[0]) / input[0]) * input[1]
        #             diff = ((output[0] - target[0]) * input[1]) ** 2
        #             sat[torch.isnan(sat)] = 0
        #             diff[torch.isnan(diff)] = 0
        #
        #             for c, roi in enumerate(roi_range):
        #                 ns = torch.sum(input[1][:, roi], dim=1)
        #
        #                 sat_ratio = torch.sum(sat[:, roi], dim=1) / ns * 100.0
        #                 sat_ratio[torch.isnan(sat_ratio)] = 0
        #
        #                 rmse = torch.sqrt(torch.sum(diff[:, roi], dim=1) / ns)
        #                 rmse[torch.isnan(rmse)] = 0
        #
        #                 rmse = rmse[sat_ratio != 0]
        #                 sat_ratio = sat_ratio[sat_ratio != 0]
        #
        #                 result.extend([[dname, c, sat.item(), r.item()] for sat, r in zip(sat_ratio, rmse)])
        #
        #     with open(self.config.model_path.split('/')[-2] + ".csv", mode="w", newline="") as file:
        #         writer = csv.writer(file)
        #         writer.writerows(result)
        #
        #     etimes.pop(0)
        #     print(sum(etimes) / len(etimes))
