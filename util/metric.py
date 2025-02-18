
import math
import torch
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
import numpy as np
from torch.nn import functional as F

import matplotlib
import matplotlib.pyplot as plt

eps = 1e-8
threshold = 0.5


def tversky_index(output, target, ch_weight=None, alpha=0.5):
    """
    Call to calculate similiarity metrics for train, validation and test
    @ inputs:
        output : predicted model result (N x C x H x W)
        target : one-hot formatted labels (N x C x H x W)
    @ outputs:
        dice similiarity = 2 * inter / (inter + union + e)
        jaccard similiarity = inter / (union + e)
    """
    # fig, axes = plt.subplots(1, 2)
    # axes = axes.flatten()
    # axes[0].imshow(output[0][2].detach().cpu().numpy(), vmin=0, vmax=1)d
    # plt.showdsd
    tversky = 0
    tensor_dim = len(output.shape)

    if ch_weight is None:
        output1 = output
        target1 = target

        true_pos = torch.sum(output1 * target1.float())
        false_neg = torch.sum(output1 * (1 - target1.float()))
        false_pos = torch.sum((1 - output1) * target1.float())

        tversky = true_pos / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + eps)
    else:
        for ch, weight in enumerate(ch_weight):
            if tensor_dim == 4:
                output1 = output[:, ch, :, :]
                target1 = target[:, ch, :, :]
            else:
                output1 = output[ch, :, :]
                target1 = target[ch, :, :]

            true_pos = torch.sum(output1 * target1.float())
            false_neg = torch.sum(output1 * (1 - target1.float()))
            false_pos = torch.sum((1 - output1) * target1.float())

            tversky += weight * true_pos / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + eps)

        tversky = tversky / sum(ch_weight)

    # intersection = torch.sum(output1 * target1.float())
    # union = torch.sum(output1) + torch.sum(target1.float()) - intersection
    # dice = 2 * intersection / (union + intersection + eps)
    # jaccard = intersection / (union + eps)
    return tversky  #, jaccard


def dice_coef(output, target, ch=None):
    """
    Call to calculate similiarity metrics for train, validation and test
    @ inputs:
        output : predicted model result (N x C x H x W)
        target : one-hot formatted labels (N x C x H x W)
    @ outputs:
        dice similiarity = 2 * inter / (inter + union + e)
        jaccard similiarity = inter / (union + e)
    """
    # fig, axes = plt.subplots(1, 2)
    # axes = axes.flatten()
    # axes[0].imshow(output[0][2].detach().cpu().numpy(), vmin=0, vmax=1)d
    # plt.showdsd
    if ch is not None:
        output1 = output[:, ch, :, :]
        target1 = target[:, ch, :, :]
    else:
        output1 = output
        target1 = target

    intersection = torch.sum(output1 * target1.float())
    union = torch.sum(output1) + torch.sum(target1.float()) - intersection
    dice = 2 * intersection / (union + intersection + eps)
    # jaccard = intersection / (union + eps)

    return dice  #, jaccard


class focal_tversky_loss:

    def __init__(self, ch_weight=None, alpha=0.5, gamma=1):
        self.ch_weight = ch_weight
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, output, target):
        tversky_loss = torch.sub(1, tversky_index(output, target, ch_weight=self.ch_weight, alpha=self.alpha))
        return torch.pow(tversky_loss, self.gamma)


def get_strut_props(images):
    """
    Call to obtain property of struts in the image batch
    @ inputs:
        images: list of images (ndarray) with strut probability (N x H x W)
    @ outputs:
        strut property
    """
    def _get_regions(bin_image):
        res = dict()
        res["label_image"] = label(bin_image)
        erode_image = binary_erosion(bin_image)
        res["label_contour"] = label(bin_image)
        res["label_contour"][erode_image == True] = 0
        res["regions"] = regionprops(res["label_image"])
        return res

    def _get_props(image, _struts_props):

        bin_image = image > 0
        region_info = _get_regions(bin_image)

        props = list()
        for r in region_info["regions"]:
            prop = dict()
            prop["y"], prop["x"] = r.centroid
            prop["area"] = r.area
            prop["label"] = r.label
            prop["contour"] = np.transpose(np.array(np.where(region_info["label_contour"] == prop["label"])))
            prop["is_true"] = False
            prop["label_cnt"] = 0
            prop["dist"] = 100000.0
            prop["overlap"] = 0.0

            props.append(prop)

        properties = dict()
        properties["label_image"] = region_info["label_image"].astype(np.uint8)
        properties["label_contour"] = region_info["label_contour"].astype(np.uint8)
        properties["props"] = props

        _struts_props.append(properties)

    struts_props = list()
    for img_batch in images:
        _get_props(img_batch, struts_props)

    return struts_props


def get_accuracy(target_struts_props, output_struts_props, dist_thres=10, overlap_thres=0.75):
    """
    Call to calculate precision and recall for stent strut detection
    @ inputs:
        target_struts_props: list (frame) / list(struts) / dict(strut props)
        output_struts_props: list (frame) / list(struts) / dict(strut props)
    @ outputs:
        f_score: 2 * r * p / (r + p)
        recall: tp / (tp + fn)
        precision: tp / (tp + fp)
    """
    def _set_true_struts(refr_struts_props, comp_struts_props):
        for refr_struts_prop, comp_struts_prop in zip(refr_struts_props, comp_struts_props):
            # Decision by distance
            for ir, refr_strut_prop in enumerate(refr_struts_prop["props"]):
                for ic, comp_strut_prop in enumerate(comp_struts_prop["props"]):
                    if (not refr_strut_prop["is_true"]) or (not comp_strut_prop["is_true"]):
                        dist = math.sqrt((comp_strut_prop["y"] - refr_strut_prop["y"]) ** 2
                                         + (comp_strut_prop["x"] - refr_strut_prop["x"]) ** 2)
                        refr_strut_prop["dist"] = dist
                        comp_strut_prop["dist"] = dist
                        if dist <= dist_thres:
                            refr_strut_prop["is_true"] = True
                            refr_strut_prop["label_cnt"] = [ic, ]
                            comp_strut_prop["is_true"] = True
                            comp_strut_prop["label_cnt"] = [ir, ]

            # Decision by overlap ratio
            for refr_strut_prop in refr_struts_prop["props"]:
                if not refr_strut_prop["is_true"]:
                    overlap = 0
                    if len(comp_struts_prop) > 0:
                        intersection = np.sum((refr_struts_prop["label_image"] == refr_strut_prop["label"])
                                              & (comp_struts_prop["label_image"] > 0))
                        overlap = intersection / refr_strut_prop["area"]
                    refr_strut_prop["overlap"] = overlap
                    if overlap > overlap_thres:
                        cnt_list = np.sort(np.unique((refr_struts_prop["label_image"] == refr_strut_prop["label"])
                                                     * comp_struts_prop["label_image"]))
                        refr_strut_prop["is_true"] = True
                        refr_strut_prop["label_cnt"] = cnt_list[1:].tolist()

            for comp_strut_prop in comp_struts_prop["props"]:
                if not comp_strut_prop["is_true"]:
                    overlap = 0
                    if len(refr_struts_prop) > 0:
                        intersection = np.sum((comp_struts_prop["label_image"] == comp_strut_prop["label"])
                                              & (refr_struts_prop["label_image"] > 0))
                        overlap = intersection / comp_strut_prop["area"]
                    comp_strut_prop["overlap"] = overlap
                    if overlap > overlap_thres:
                        cnt_list = np.sort(np.unique((comp_struts_prop["label_image"] == comp_strut_prop["label"])
                                                     * refr_struts_prop["label_image"]))
                        comp_strut_prop["is_true"] = True
                        comp_strut_prop["label_cnt"] = cnt_list[1:].tolist()

    def _get_metric(struts_props):
        true, false = 0, 0
        for struts_prop in struts_props:
            # true0, false0 = 0, 0
            for strut_prop in struts_prop["props"]:
                if strut_prop["is_true"]:
                    # true0 += 1
                    true += 1
                else:
                    # false0 += 1
                    false += 1
            # print(true0, false0)
        return true / (true + false + eps)

    # False negative & false positive decision
    _set_true_struts(target_struts_props, output_struts_props)

    # Recall & precision calculation
    recall = _get_metric(target_struts_props)
    precision = _get_metric(output_struts_props)
    f_score = 2 * recall * precision / (recall + precision + eps)

    return f_score, precision, recall


def get_distance(target_struts_props, output_struts_props):

    def _calc_distance(refr_struts_props, comp_struts_props):

        for i, (refr_struts_prop, comp_struts_prop) in enumerate(zip(refr_struts_props, comp_struts_props)):

            for refr_strut_prop in refr_struts_prop["props"]:
                if refr_strut_prop["is_true"]:
                    cnt_list = refr_strut_prop["label_cnt"]
                    # if len(cnt_list) == 0:
                    #     continue
                    refr_ps = refr_strut_prop["contour"]
                    if any([cnt >= len(comp_struts_prop["props"]) for cnt in cnt_list]):
                        continue
                    comp_ps = np.concatenate([comp_struts_prop["props"][_i]["contour"] for _i in cnt_list])

                    dist = np.ndarray((refr_ps.shape[0], comp_ps.shape[0]))
                    for ir, refr_p in enumerate(refr_ps):
                        for ic, comp_p in enumerate(comp_ps):
                            dist[ir, ic] = math.sqrt((refr_p[0] - comp_p[0]) ** 2 + (refr_p[1] - comp_p[1]) ** 2)
                    refr_dist = np.min(dist, axis=1)
                    comp_dist = np.min(dist, axis=0)

                    refr_strut_prop["hausdorff"] = np.max((np.max(refr_dist), np.max(comp_dist)))
                    refr_strut_prop["avg_surface"] = np.mean(np.concatenate((refr_dist, comp_dist)))

            for comp_strut_prop in comp_struts_prop["props"]:
                if comp_strut_prop["is_true"]:
                    cnt_list = comp_strut_prop["label_cnt"]
                    # if len(cnt_list) == 0:
                    #     continue
                    comp_ps = comp_strut_prop["contour"]
                    if any([cnt >= len(refr_struts_prop["props"]) for cnt in cnt_list]):
                        continue
                    refr_ps = np.concatenate([refr_struts_prop["props"][_i]["contour"] for _i in cnt_list])

                    dist = np.ndarray((comp_ps.shape[0], refr_ps.shape[0]))
                    for ic, comp_p in enumerate(comp_ps):
                        for ir, refr_p in enumerate(refr_ps):
                            dist[ic, ir] = math.sqrt((comp_p[0] - refr_p[0]) ** 2 + (comp_p[1] - refr_p[1]) ** 2)
                    comp_dist = np.min(dist, axis=1)
                    refr_dist = np.min(dist, axis=0)

                    comp_strut_prop["hausdorff"] = np.max((np.max(comp_dist), np.max(refr_dist)))
                    comp_strut_prop["avg_surface"] = np.mean(np.concatenate((comp_dist, refr_dist)))

    def _get_mean_distance(refr_struts_props, comp_struts_props):

        _distance = dict()
        _distance["hd_r"], _distance["hd_c"] = [], []
        _distance["asd_r"], _distance["asd_c"] = [], []

        for refr_struts_prop, comp_struts_prop in zip(refr_struts_props, comp_struts_props):

            for t_prop in refr_struts_prop["props"]:
                if t_prop["is_true"]:
                    if "hausdorff" in t_prop:
                        _distance["hd_r"].append(t_prop["hausdorff"])
                    if "avg_surface" in t_prop:
                        _distance["asd_r"].append(t_prop["avg_surface"])

            for o_prop in comp_struts_prop["props"]:
                if o_prop["is_true"]:
                    if "hausdorff" in o_prop:
                        _distance["hd_c"].append(o_prop["hausdorff"])
                    if "avg_surface" in o_prop:
                        _distance["asd_c"].append(o_prop["avg_surface"])

        _distance["hd_r"] = [np.mean(np.array(_distance["hd_r"])), np.median(np.array(_distance["hd_r"]))]
        _distance["hd_c"] = [np.mean(np.array(_distance["hd_c"])), np.median(np.array(_distance["hd_c"]))]
        _distance["asd_r"] = [np.mean(np.array(_distance["asd_r"])), np.median(np.array(_distance["asd_r"]))]
        _distance["asd_c"] = [np.mean(np.array(_distance["asd_c"])), np.median(np.array(_distance["asd_c"]))]

        return _distance

    _calc_distance(target_struts_props, output_struts_props)
    distance = _get_mean_distance(target_struts_props, output_struts_props)

    return distance


def get_struts_metric(output, target):
    """
    Call to calculate metrics for train, validation and test
    @ inputs:
        output : predicted model result (N x C x H x W)
        target : one-hot formatted labels (N x C x H x W)
    @ outputs (type: python dictionary):
        tot_metric : metrics for total class prediction
        cls_metric : metrics for per-class prediction

        in 'metric' dict...
        - sensitivity = tp / (tp + fn + e)
        - specificity = tn / (tn + fp + e)
        - precision = tp / (tp + fp + e)
        - recall = sensitivity
        - accuracy = (tp + tn) / (tp + tn + fp + fn + e)
        - jaccard similiarity = inter / (union + e)
        - dice coefficient = 2 * inter / (inter + union + e)
        - f1 score = dice coefficient
    """
    tot_tp, tot_tn, tot_fp, tot_fn = 0.0, 0.0, 0.0, 0.0
    tot_it, tot_un = 0.0, 0.0

    cls_metric = list()
    for c in range(output.size(1)):
        cls_output = output[:, c, :, :] > threshold
        cls_target = target[:, c, :, :] == 1

        cls_tp = float(torch.sum(((cls_output == 1) + (cls_target == 1)) == 2))  # true positive
        cls_tn = float(torch.sum(((cls_output == 0) + (cls_target == 0)) == 2))  # true negative
        cls_fp = float(torch.sum(((cls_output == 1) + (cls_target == 0)) == 2))  # false positive
        cls_fn = float(torch.sum(((cls_output == 0) + (cls_target == 1)) == 2))  # false negative

        cls_it = float(torch.sum((cls_output + cls_target) == 2))  # intersection
        cls_un = float(torch.sum((cls_output + cls_target) >= 1))  # union

        tot_tp += cls_tp
        tot_tn += cls_tn
        tot_fp += cls_fp
        tot_fn += cls_fn

        tot_it += cls_it
        tot_un += cls_un

        metric = dict()

        metric["sensitivity"] = cls_tp / (cls_tp + cls_fn + eps)  # sensitivity
        metric["specificity"] = cls_tn / (cls_tn + cls_fp + eps)  # specificity
        metric["precision"] = cls_tp / (cls_tp + cls_fp + eps)  # precision
        metric["recall"] = metric["sensitivity"]  # recall

        metric["accuracy"] = (cls_tp + cls_tn) / (cls_tp + cls_tn + cls_fp + cls_fn + eps)  # accuracy
        metric["jaccard"] = cls_it / (cls_un + eps)  # Jaccard similarity
        metric["dice_coef"] = 2 * cls_it / (cls_it + cls_un + eps)  # dice coefficient
        metric["f1_score"] = metric["dice_coef"]  # f1 score

        cls_metric.append(metric)

    tot_metric = dict()

    tot_metric["sensitivity"] = tot_tp / (tot_tp + tot_fn + eps)  # sensitivity
    tot_metric["specificity"] = tot_tn / (tot_tn + tot_fp + eps)  # specificity
    tot_metric["precision"] = tot_tp / (tot_tp + tot_fp + eps)  # precision
    tot_metric["recall"] = tot_metric["sensitivity"]  # recall

    tot_metric["accuracy"] = (tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn + eps)  # accuracy
    tot_metric["jaccard"] = tot_it / (tot_un + eps)  # Jaccard similarity
    tot_metric["dice_coef"] = 2 * tot_it / (tot_it + tot_un + eps)  # dice coefficient
    tot_metric["f1_score"] = tot_metric["dice_coef"]

    return tot_metric, cls_metric


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3, device_ids=[]):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, bias=False)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    # Reflection padding
    padding = torch.nn.ReflectionPad2d(int(mean))

    return gaussian_filter, padding


def get_ssim(outputs, targets, r=5, s=2):
    """ Get structur similarity of given tensors
    Parameters:
        outputs (tensor): given tensors (output image) (4D: N X C X H X W)
        targets (tensor): given tensors (target image) (4D: N X C X H X W)
        r (int): radius of gaussian filter
        s (int): sigma of gaussian filter
    Returns:
        ssim (Tensor): list of calculated ssim (numel: N)
    """
    # Basic parameters
    l_max = 1
    ch = outputs.size(1)
    c = [(0.01*l_max) ** 2, (0.03*l_max) ** 2, ((0.03*l_max) ** 2) / 2]
    outputs = outputs.detach()  #.cpu()
    targets = targets.detach()  #.cpu()

    # Gaussian kernel initialization
    gaussian_filter, padding = get_gaussian_kernel(kernel_size=r, sigma=s, channels=ch)
    gaussian_filter = gaussian_filter.to(outputs.device)

    # SSIM calculation
    mu_o = gaussian_filter(padding(outputs))
    mu_t = gaussian_filter(padding(targets))

    mu_ot = mu_o * mu_t
    mu_o2 = mu_o ** 2
    mu_t2 = mu_t ** 2

    sigma_o2 = gaussian_filter(padding(outputs ** 2)) - mu_o2
    sigma_o = torch.sqrt(sigma_o2)
    sigma_t2 = gaussian_filter(padding(targets ** 2)) - mu_t2
    sigma_t = torch.sqrt(sigma_t2)
    sigma_ot = gaussian_filter(padding(outputs * targets)) - mu_ot

    lumi = (2 * mu_o * mu_t + c[0]) / (mu_o2 + mu_t2 + c[0] + eps)
    cont = (2 * sigma_o * sigma_t + c[1]) / (sigma_o2 + sigma_t2 + c[1] + eps)
    stru = (sigma_ot + c[2]) / (sigma_o * sigma_t + c[2] + eps)

    ssim = lumi * cont * stru
    ssim[ssim != ssim] = 0
    ssim = torch.mean(ssim, dim=[1, 2, 3])

    return ssim


def get_psnr(outputs, targets):
    """ Get peak signal-to-ratio of given tensors
    Parameters:
        outputs (tensor): given tensors (output image) (4D: N X C X H X W)
        targets (tensor): given tensors (target image) (4D: N X C X H X W)
    Returns:
        psnr (Tensor): list of calculated psnr (numel: N)
    """
    # Basic parameters
    l_max = 1
    outputs = outputs.detach()  #.cpu()
    targets = targets.detach()  #.cpu()

    # PSNR calculation
    mse = torch.mean((outputs - targets) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10((l_max ** 2) / (mse + eps))

    return psnr


def gradient(img):

    left = img
    right = F.pad(img, [0, 1, 0, 0])[:, :, :, 1:]
    top = img
    bottom = F.pad(img, [0, 0, 0, 1])[:, :, 1:, :]

    gx, gy = right - left, bottom - top
    gx[:, :, :, -1] = 0
    gy[:, :, -1, :] = 0

    return torch.pow(gx, 2) + torch.pow(gy, 2)


def axial_gradient(img):

    top = img
    bottom = F.pad(img, [0, 0, 0, 1])[:, :, 1:, :]

    gy = bottom - top
    gy[:, :, -1, :] = 0

    return torch.abs(gy)


def get_mse(outputs, targets, weights=None):
    if weights is not None:
        outputs = outputs * weights
        targets = targets * weights

    mse = torch.sum(torch.square(torch.abs(outputs - targets))).detach()
    if weights is None:
        mse = mse / outputs.reshape(-1).size()[0]
    else:
        mse = mse / torch.sum(weights).detach()

    return mse


# def get_struts_contour(img):
#
# def get_false_struts(output, target, thres=threshold, dist_thres=20, overlap_thres=0.5):
#     """
#     Call to obtain false struts for performance validation of stent strut detection
#     @ inputs:
#         output : predicted model result (N x C x H x W)
#         target : one-hot formatted labels (N x C x H x W)
#     @ outputs:
#         false_negatives
#         false_positives
#     """
#     def _get_regions(bin_image):
#         res = dict()
#         res["label_image"] = label(bin_image)
#         res["regions"] = regionprops(res["label_image"])
#         return res
#
#     def _get_strut_info(region, label_image):
#         dict_strut = dict()
#         dict_strut["y"], dict_strut["x"] = region.centroid
#         dict_strut["area"] = region.area
#         dict_strut["label"] = int(region.label)
#         dict_strut["region"] = label_image == dict_strut["label"]
#         return dict_strut
#
#     def _get_false_struts(refr_res, comp_res):
#         # Reference strut definition
#         n_refr_struts = len(refr_res["regions"])
#         refr_struts = dict()
#         refr_struts["x"] = [0,] * n_refr_struts
#         refr_struts["y"] = [0,] * n_refr_struts
#         refr_struts["is_false"] = [1,] * n_refr_struts
#         refr_struts["dist"] = [0,] * n_refr_struts
#         refr_struts["overlap_refr"] = [0,] * n_refr_struts
#         refr_struts["overlap_comp"] = [0,] * n_refr_struts
#
#         # False decision
#         for rfr in refr_res["regions"]:
#             # A reference strut
#             refr_strut_info = _get_strut_info(rfr, refr_res["label_image"])
#
#             n_comp_struts = len(comp_res["regions"])
#             if n_comp_struts != 0:
#                 dist, overlap_refr, overlap_comp = list(), list(), list()
#                 for i, pr in enumerate(comp_res["regions"]):
#                     # A comparison strut
#                     comp_strut_info = _get_strut_info(pr, comp_res["label_image"])
#
#                     # distance condition
#                     dist.append(math.sqrt(math.pow(comp_strut_info["y"] - refr_strut_info["y"], 2)
#                                           + math.pow(comp_strut_info["x"] - refr_strut_info["x"], 2)))
#                     # overlap condition
#                     intersection = np.sum(comp_strut_info["region"] & refr_strut_info["region"])
#                     overlap_refr.append(intersection / refr_strut_info["area"])
#                     overlap_comp.append(intersection / comp_strut_info["area"])
#
#                 # decision
#                 refr_struts["x"][refr_strut_info["label"] - 1] = refr_strut_info["x"]
#                 refr_struts["y"][refr_strut_info["label"] - 1] = refr_strut_info["y"]
#                 if min(dist) <= dist_thres or sum(overlap_refr) > overlap_thres or sum(overlap_comp) > overlap_thres:
#                     refr_struts["is_false"][refr_strut_info["label"] - 1] = 0
#                     refr_struts["dist"][refr_strut_info["label"] - 1] = np.min(dist)
#                     refr_struts["overlap_refr"][refr_strut_info["label"] - 1] = sum(overlap_refr)
#                     refr_struts["overlap_comp"][refr_strut_info["label"] - 1] = sum(overlap_comp)
#
#         return refr_struts
#
#     # Batch loop
#     false_negatives, false_positives = list(), list()
#     for i in range(output.shape[0]):
#         # Labeling
#         res_output = _get_regions(output[i, 1, :, :].cpu().numpy() > thres)
#         res_target = _get_regions(target[i, 1, :, :].cpu().numpy() > thres)
#
#         # False negative & false positive decision
#         false_negatives.append(_get_false_struts(res_target, res_output))
#         false_positives.append(_get_false_struts(res_output, res_target))
#
#     return false_negatives, false_positives
#
#
# def get_accuracy(false_negatives, false_positives):
#     """
#     Call to calculate precision and recall for stent strut detection
#     @ inputs:
#         false_negatives: list for batch
#         false_positives: list for batch
#     @ outputs:
#         f_score: 2 * r * p / (r + p)
#         recall: tp / (tp + fn)
#         precision: tp / (tp + fp)
#     """
#     tp, fn = 0, 0
#     for iter0 in false_negatives:
#         for iter1 in iter0:
#             tp = tp + len(iter1["is_false"]) - sum(iter1["is_false"])
#             fn = fn + sum(iter1["is_false"])
#     recall = tp / (tp + fn + eps)
#
#     tp, fp = 0, 0
#     for iter0 in false_positives:
#         for iter1 in iter0:
#             tp = tp + len(iter1["is_false"]) - sum(iter1["is_false"])
#             fp = fp + sum(iter1["is_false"])
#     precision = tp / (tp + fp + eps)
#
#     f_score = 2 * recall * precision / (recall + precision + eps)
#
#     return f_score, recall, precision
