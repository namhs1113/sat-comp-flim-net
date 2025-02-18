
import os
import torch
import numpy as np
from skimage.transform import resize
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

eps = 1e-12


class ImageFolder(data.Dataset):
    def __init__(self, root, phase):
        """Initializes image paths and preprocessing module."""

        # Parameters setting
        assert phase in ["train", "valid", "test"]
        self.phase = phase

        # Path setting
        assert root[-1] == "/", "Last character should be /."
        self.root = root
        data_paths = os.listdir(os.path.join(self.root, self.phase))
        data_paths.sort()

        self.data_paths = []
        for data_path in data_paths:
            self.data_paths.append(os.path.join(self.root, self.phase, data_path))

    def __getitem__(self, index):
        """Reads an image from a file, preprocesses it and returns."""

        # Specify data path
        data_path = self.data_paths[index]

        # Load ROI data
        n_rois = np.fromfile(data_path, count=1, dtype=np.int32)[0]
        rois = np.fromfile(data_path, offset=4, count=n_rois, dtype=np.int32)

        # Load pulse data
        scan = int(data_path.split("_t")[1][:-6])
        aline, aline0 = 256 if n_rois == 5 else 500, 256
        if self.phase == "test":
            np.random.seed(index * 100 + 0)
        ra = 0 if n_rois == 5 else np.random.randint(0, aline - aline0)  # Random index
        pulse_temp = np.fromfile(data_path,
                                 offset=4 * (n_rois + 1 + ra * scan),
                                 count=scan*aline0,
                                 dtype=np.float32)

        # Reshape pulse
        pulse_temp = np.reshape(pulse_temp, [aline0, scan, 1])

        # Zero padding & random resizing/drifting in time domain
        scan0 = 224
        scanr = scan
        sp = 0

        pulse0 = np.zeros((aline0, scan0, 1), dtype=pulse_temp.dtype)
        pulse0[:, sp:sp+scanr] = pulse_temp

        # Random horizontal fliping for data augmentation
        if self.phase == "test":
            np.random.seed(index * 100 + 1)
        if np.random.random() < 0.5:
            pulse0 = np.flip(pulse0, axis=0)

        # Find saturation indices in the original pulse data
        sat0 = pulse0 < -0.5  # saturation mask in the original data
        pulse0[sat0] = -pulse0[sat0]  # original data

        # Determine pulse ranges
        roi_range = list()
        if n_rois == 8:  # System 1 dataset
            rois = np.reshape(rois, [4, 2])
            rois = rois[:, 0]
            roi_range.append(range(rois[0], scan0))  # irf range
            roi_range.append(range(rois[1], rois[0] - 1))  # ch1 range
            roi_range.append(range(rois[2], rois[1] - 1))  # ch2 range
            roi_range.append(range(rois[3], rois[2] - 1))  # ch3 range
        elif n_rois == 5:  # System 2 dataset
            roi_range.append(range(rois[0], rois[1] - 1))  # irf range
            roi_range.append(range(rois[1], rois[2] - 1))  # ch1 range
            roi_range.append(range(rois[2], rois[3] - 1))  # ch2 range
            roi_range.append(range(rois[3], rois[4] - 1))  # ch3 range

        # Channel-wise random adjustment
        pulse1 = pulse0.copy()

        # IRF random processing
        pulse0_irf = pulse0[:, roi_range[0]]
        M_pulse0_irf = np.max(pulse0_irf)

        if self.phase == "test":
            np.random.seed(index * 100 + 2)
        M_new = np.random.random() * 0.45 + 0.5

        if self.phase == "test":
            np.random.seed(index * 100 + 3)
        if np.random.random() > 0.8:
            M_new *= 1.66
        pulse1_irf = pulse0_irf / M_pulse0_irf * M_new

        pulse1[:, roi_range[0]] = pulse1_irf

        # Emission band random processing
        pulse1_em = [None,] * 3
        for ch in range(3):
            pulse0_em_ch = pulse0[:, roi_range[ch + 1]]

            M_pulse0_em_ch = np.max(pulse0_em_ch)
            pulse1_em[ch] = pulse0_em_ch
            if M_pulse0_em_ch > 0.15:

                if self.phase == "test":
                    np.random.seed(index * 100 + (ch + 1) * 10 + 4)
                if np.random.random() > 0.6:
                    pulse1_em[ch] /= M_pulse0_em_ch

                if self.phase == "test":
                    np.random.seed(index * 100 + (ch + 1) * 10 + 5)
                if np.random.random() > 0.5:
                    pulse1_em[ch] *= 2.1

            if self.phase == "test":
                np.random.seed(index * 100 + (ch + 1) * 10 + 6)
            mul_factor = np.random.randn() * 1.3 + 1.8
            mul_factor = np.clip(mul_factor, a_min=0.5, a_max=3.0)

            pulse1_em[ch] *= mul_factor

            pulse1[:, roi_range[ch + 1]] = pulse1_em[ch]

        # Pulse incorporation & generate saturation mask
        pulse2 = pulse1.copy()

        # Find random normalization factor -> random saturation level
        if self.phase == "test":
            np.random.seed(index * 100 + 5)
        sat_lev = np.random.randn() * 0.0074 + 0.94
        sat1 = pulse1 >= sat_lev  # virtual saturation mask
        pulse1[sat1] = sat_lev

        # Unsqueezing to introduce the channel dimension (HW -> (C=1)HW)
        sat0 = sat0.astype(np.float32)
        sat1 = sat1.astype(np.float32)

        # Make input and target tensor
        input = np.concatenate((pulse1, sat1), axis=2)
        input[np.isnan(input)] = 0
        target = pulse2
        target[np.isnan(target)] = 0
        mask = (sat0 == 1).astype(np.float32)

        # ToTensor
        input = T.ToTensor()(input)
        target = T.ToTensor()(target)
        mask = T.ToTensor()(mask)
        roi_range = torch.tensor([[r.start, r.stop] for r in roi_range])

        return input, target, mask, data_path, roi_range

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)


class TestFolder(data.Dataset):
    def __init__(self, root):
        """Initializes image paths and preprocessing module."""

        # Path setting
        assert root[-1] == "/", "Last character should be /."
        self.root = root
        data_paths = os.listdir(os.path.join(self.root, "test"))
        data_paths.sort()

        self.data_paths = []
        for data_path in data_paths:
            self.data_paths.append(os.path.join(self.root, "test", data_path))

    def __getitem__(self, index):
        """Reads an image from a file, preprocesses it and returns."""

        # Specify data path
        data_path = self.data_paths[index]

        # Load pulse data
        pulse_temp = np.fromfile(data_path, dtype=np.float32)

        # Reshape pulse
        input = np.reshape(pulse_temp, [2, 256, 224])
        input = np.transpose(input, (1, 2, 0))

        # ToTensor
        input = T.ToTensor()(input)
        return input, data_path

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)


def get_loader(dataset_path, phase="train", shuffle=True, batch_size=1, num_workers=2):
    """Builds and returns Dataloader."""
    if phase != "test":
        dataset = ImageFolder(root=dataset_path,
                              phase=phase)
    else:
        dataset = TestFolder(root=dataset_path)

    data_loader = data.DataLoader(dataset=dataset,
                                  shuffle=shuffle,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  num_workers=num_workers)

    return data_loader
