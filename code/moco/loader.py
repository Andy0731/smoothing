# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import torch
import numpy as np


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianNoise(torch.nn.Module):
    """Adding random Gaussian noise to image.
   
    Args:
        noise_sd (float): standard deviation of Gaussian noise

    Returns:
        x + noise
    """

    def __init__(self, noise_sd=0.0):
        super().__init__()
        self.noise_sd = noise_sd

    def forward(self, x):
        return x + torch.randn_like(x) * self.noise_sd
    
    
class GaussianNoiseNEP(torch.nn.Module):
    """Adding random Gaussian noise to image.
   
    Args:
        noise_sd (float): standard deviation of Gaussian noise

    Returns:
        x + noise
    """

    def __init__(self, noise_sd_list=[0.0, 0.25, 0.5, 1.0]):
        super().__init__()
        self.noise_sd_list = np.array(noise_sd_list)

    def forward(self, x):
        noise_sd = np.random.choice(self.noise_sd_list, size=(1))
        x = x + torch.randn_like(x) * noise_sd[0]
        return x