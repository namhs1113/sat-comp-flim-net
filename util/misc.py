
import os
import torch
import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib import cm
from torchvision.utils import make_grid
from torchvision import transforms as T
import datetime

from collections import OrderedDict


def identity(x):
    return x


def average(_list: list):
    if len(_list) != 0:
        return sum(_list) / len(_list)
    else:
        return 0


def print_progress_bar(it, total, prefix="", suffix="", dec=1, bar_len=50, fill="█"):
    """
    Call in a loop to create terminal progress bar
    @params:
        it      - Required  : current iteration (Int)
        total   - Required  : total iterations (Int)
        prefix  - Optional  : prefix string (Str)
        suffix  - Optional  : suffix string (Str)
        dec     - Optional  : positive number of decimals in percent complete (Int)
        bar_len - Optional  : character length of bar (Int)
        fill    - Optional  : bar fill character (Str)
    """
    percent = ("%" + str(dec + 3) + "." + str(dec) + "f") % (100.0 * (it / float(total)))
    filled_len = int(bar_len * it // total)
    bar = fill * filled_len + "-" * (bar_len - filled_len)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="")  # if it != total else "\n")


def get_grid_image(image_list, nrow=4, transform=None):
    # image_list: list of tensor of RGB image
    if transform is None:
        transform = identity
    idx = np.random.choice(image_list[0].size(0), nrow)
    grid_list = list()
    for image in image_list:
        grid_list.append(transform(make_grid(image[idx].detach().cpu(), nrow=4, padding=5)))
    grid_image = torch.cat([grid for grid in grid_list], dim=1)
    return grid_image


def clip(image):
    image[image > 1] = 1
    image[image < 0] = 0
    return image


def normalize(img, vmin, vmax):
    ind = (img - vmin) / (vmax - vmin)
    ind[ind > 1] = 1
    ind[ind < 0] = 0
    return ind


def pseudocolor(tensors, cmap_name, vmin, vmax):
    tensors = tensors.detach().cpu()
    tensors1 = list()
    for tensor in tensors:
        index = normalize(tensor, vmin, vmax)
        index = (255.0 * index).int().view(-1)

        cmap = cm.get_cmap(cmap_name, 256)
        cmap = cmap([c for c in range(256)])
        cmap = cmap[:, 0:3]
        cmap[0, :] = 0.0

        tensors1.append(cmap[index, :].reshape(tensor.size() + (3,)))
    tensors1 = np.array(tensors1)
    return torch.from_numpy(tensors1).permute((0, 3, 1, 2)).float()


def inv_transform(image):
    # Transfrom
    inv_tf = T.Compose([
                        T.Normalize([-0.48501961, -0.45795686, -0.40760392], [1, 1, 1]),
                        T.Lambda(clip),
                       ])
    return inv_tf(image)


def rgb_transform(image):
    # Transfrom
    tf = T.Compose([
                    T.Normalize([0.48501961, 0.45795686, 0.40760392], [1, 1, 1]),
                   ])
    return tf(image)


def imshow_with_struts(images, masks, preds, prob_vis=False,
                       target_struts_props=None, output_struts_props=None,
                       visualize=True, save=False, savepath=None):
    # Assert
    if save:
        assert savepath is not None

    # Struts visualization helper function
    def strut_vis(ax, struts_prop):
        for strut_prop in struts_prop:
            y, x = strut_prop["y"], strut_prop["x"]
            if strut_prop["is_true"]:
                ax.plot(x, y, '.b', markersize=1)
            else:
                ax.plot(x, y, '.r', markersize=1)

    # Strut visualization
    if not prob_vis:
        masks = masks.argmax(axis=1)
        preds = preds.argmax(axis=1)
    else:
        masks = masks[:, 1, :, :]
        preds = preds[:, 1, :, :]

    # Visualization
    n_batch = images.shape[0]
    fig, axes = plt.subplots(n_batch, 3)

    for i in range(n_batch):
        axes[i, 0].imshow(images[i, 0], vmin=0, vmax=10, cmap="gray")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(masks[i], vmin=0, vmax=1, cmap="viridis")
        if target_struts_props is not None:
            strut_vis(axes[i, 1], target_struts_props[i])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        axes[i, 2].imshow(preds[i], vmin=0, vmax=1, cmap="viridis")
        if output_struts_props is not None:
            strut_vis(axes[i, 2], output_struts_props[i])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

    fig.set_dpi(450)
    fig.set_size_inches(4.8, 1.6 * n_batch)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01, wspace=0.01)

    if visualize:
        plt.show()

    if save:
        dt = datetime.datetime.now()
        plt.savefig(os.path.join(savepath, "%4d%02d%02d-%02d%02d%02d.jpg"
                                 % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)), dpi=450)
        plt.close()


def imshow(images, masks, preds, idx=0):

    # Image reconstruction
    idx = idx if idx >= 0 else 0
    idx = idx if idx < images.size(0) else images.size(0) - 1
    img_ch = images.size(1)
    num_classes = masks.size(1)

    image = images[idx, :, :, :].numpy().transpose((1, 2, 0))
    mask = masks[idx, :, :, :].numpy().transpose((1, 2, 0)).argmax(axis=2) if masks is not None else []
    pred = preds[idx, :, :, :].numpy().transpose((1, 2, 0)).argmax(axis=2)

    if img_ch == 1:
        image = image.reshape(images.size(2), images.size(3))
    if masks is []:
        mask = np.zeros(pred.shape, dtype=np.uint8)

    mask_c = np.zeros(mask.shape + (3,), dtype=np.uint8)
    pred_c = np.zeros(pred.shape + (3,), dtype=np.uint8)

    # Coloring
    if num_classes == 2:
        color = (255, 255, 0)  # random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors = [color, (255 - color[0], 255 - color[1], 255 - color[2])]
    elif num_classes == 5:
        colors = [(64, 64, 64), (147, 100, 141), (76, 195, 217), (255, 198, 93), (241, 103, 69)]
    else:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]

    for c in range(num_classes):
        for i in range(3):
            mask_c[:, :, i] += ((mask[:, :] == c) * (colors[c][i])).astype("uint8")
            pred_c[:, :, i] += ((pred[:, :] == c) * (colors[c][i])).astype("uint8")

    # Visualization
    fig, axes = plt.subplots(1, 3)
    axes = axes.flat

    axes[0].imshow(image, vmin=-0.5, vmax=0.5, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(mask_c, vmin=0, vmax=255)
    axes[1].set_title("Target Image")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(pred_c, vmin=0, vmax=255)
    axes[2].set_title("Predicted Image")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.100, wspace=0.112)

    plt.show(block=False)
    plt.pause(2)

    return fig


def weight_visualizer(model, images, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = module.__class__.__name__ # str(module.__class__).split(".")[-1].split("'")[0]
            print(class_name)
            module_idx = len(summary)

            if class_name == "Conv2d":
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["weight"] = module.weight.cpu()
                summary[m_key]["input_channel"] = module.weight.size(1)
                summary[m_key]["output_channel"] = module.weight.size(0)

                weight_mat = summary[m_key]["weight"].reshape(-1, 1, 3, 3)
                weight_grid = make_grid(weight_mat,
                                        nrow=int(np.sqrt(weight_mat.size(0))), padding=1).permute(1, 2, 0)

                plt.imsave(m_key + ".bmp", weight_grid[:, :, 0], cmap="bwr", vmin=-0.5, vmax=0.5, format="bmp")

                output_image = output.permute(2, 3, 1, 0).cpu()
                plt.imshow(output_image[:, :, 0, 0], cmap="viridis")
                plt.show()
                # layer_weight = make_grid(self.model.enc_conv1.conv[3].weight[10, :, :, :].detach().cpu(),
                #                          nrow=8, padding=1).permute(1, 2, 0)
                # plt.imshow(layer_weight[:, :, 0],
                #            cmap='bwr',
                #            vmin=-1,
                #            vmax=1)
                # plt.show()

                # summary[m_key]["input_shape"][0] = batch_size
                # if isinstance(output, (list, tuple)):
                #     summary[m_key]["output_shape"] = [
                #         [-1] + list(o.size())[1:] for o in output
                #     ]
                # else:
                #     summary[m_key]["output_shape"] = list(output.size())
                #     summary[m_key]["output_shape"][0] = batch_size

            # params = 0
            # if hasattr(module, "weight") and hasattr(module.weight, "size"):
            #     params += torch.prod(torch.LongTensor(list(module.weight.size())))
            #     summary[m_key]["trainable"] = module.weight.requires_grad
            # if hasattr(module, "bias") and hasattr(module.bias, "size"):
            #     params += torch.prod(torch.LongTensor(list(module.bias.size())))
            # summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    # if isinstance(input_size, tuple):
    #     input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [images]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    with torch.no_grad():
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # print("----------------------------------------------------------------")
    # line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    # print(line_new)
    # print("================================================================")
    # total_params = 0
    # total_output = 0
    # trainable_params = 0
    # for layer in summary:
    #     # input_shape, output_shape, trainable, nb_params
    #     line_new = "{:>20}  {:>25} {:>15}".format(
    #         layer,
    #         str(summary[layer]["output_shape"]),
    #         "{0:,}".format(summary[layer]["nb_params"]),
    #     )
    #     total_params += summary[layer]["nb_params"]
    #     total_output += np.prod(summary[layer]["output_shape"])
    #     if "trainable" in summary[layer]:
    #         if summary[layer]["trainable"] == True:
    #             trainable_params += summary[layer]["nb_params"]
    #     print(line_new)
    #
    # # assume 4 bytes/number (float on cuda).
    # total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    # total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    # total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    # total_size = total_params_size + total_output_size + total_input_size
    #
    # print("================================================================")
    # print("Total params: {0:,}".format(total_params))
    # print("Trainable params: {0:,}".format(trainable_params))
    # print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    # print("----------------------------------------------------------------")
    # print("Input size (MB): %0.2f" % total_input_size)
    # print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    # print("Params size (MB): %0.2f" % total_params_size)
    # print("Estimated Total Size (MB): %0.2f" % total_size)
    # print("----------------------------------------------------------------")
    # # return summary


class Circularize:
    def __init__(self, radius, nalines):

        self.radius = radius  # height
        self.nalines = nalines  # width

        xx, yy = np.meshgrid(np.arange(2 * self.radius), np.arange(2 * self.radius))
        xx, yy = self.radius - xx, yy - self.radius

        rho = np.sqrt(xx ** 2 + yy ** 2) * (self.radius - 1) / self.radius
        rho[rho >= radius - 1] = radius - 1
        theta = (np.arctan2(yy, xx) + np.pi) * (self.nalines - 1) / (2 * np.pi)

        # Nearest neighborhood interpolation
        self.rho_nn = np.round(rho).astype(np.int)
        self.theta_nn = np.round(theta).astype(np.int)

        # Bilinear interpolation
        self.rho_bl = [np.floor(rho).astype(np.int), np.ceil(rho).astype(np.int)]
        self.theta_bl = [np.floor(theta).astype(np.int), np.ceil(theta).astype(np.int)]
        w_r0, w_t0 = rho - self.rho_bl[0], theta - self.theta_bl[0]
        w_r1, w_t1 = 1 - w_r0, 1 - w_t0

        self.weight = [ [ np.multiply(w_r1, w_t1), np.multiply(w_r1, w_t0) ],
                        [ np.multiply(w_r0, w_t1), np.multiply(w_r0, w_t0)] ]

    def __call__(self, rect, method='bilinear'):

        rect = np.array(rect).astype(np.float64)

        circ = None
        if method == 'nearest':
            circ = rect[self.rho_nn, self.theta_nn]
        elif method == 'bilinear':
            circ = np.multiply(self.weight[0][0], rect[self.rho_bl[0], self.theta_bl[0]]) + \
                   np.multiply(self.weight[0][1], rect[self.rho_bl[0], self.theta_bl[1]]) + \
                   np.multiply(self.weight[1][0], rect[self.rho_bl[1], self.theta_bl[0]]) + \
                   np.multiply(self.weight[1][1], rect[self.rho_bl[1], self.theta_bl[1]])

        circ = circ.astype(np.uint8)

        return circ

