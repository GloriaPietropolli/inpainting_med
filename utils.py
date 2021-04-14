import random
import torch
import torchvision.transforms as transforms
import numpy as np


def generate_input_mask(shape, hole_size, hole_area=None, number_holes=1):
    """
    shape = (B, C, D, H, W), C=1 bc it iis only necessary one channel
    hole_size = (hole_min_d, hole_max_d, hole_min_w, hole_max_d, hole_minh, hole_max_h)
    hole_area = (left_corner, depth, width, height)
    number_holes = holes considered
    output = masked tensor of shape (N, C, D, H, W) with holes (denoted with channel value 1)
    """
    mask = torch.zeros(shape)  # complete tensor, both covered and not
    mask_batch_size, _, mask_d, mask_h, mask_w = mask.shape

    for i in range(mask_batch_size):
        for _ in range(number_holes):
            pass  # later implement what happens if more than a hole is considered


