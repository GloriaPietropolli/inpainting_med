import random
import torch
import torchvision.transforms as transforms
import numpy as np


def generate_input_mask(shape, hole_size, hole_area=None, number_holes=1):
    """
    shape = (B, C, D, H, W), C=1 bc it iis only necessary one channel
    hole_size = (hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w)
    hole_area = (left_corner, depth, width, height)
    number_holes = holes considered
    output = masked tensor of shape (N, C, D, H, W) with holes (denoted with channel value 1)
    """
    mask = torch.zeros(shape)  # complete tensor, both covered and not
    mask_batch_size, _, mask_d, mask_h, mask_w = mask.shape

    for i in range(mask_batch_size):
        for _ in range(number_holes):
            pass  # later implement what happens if more than a hole is considered
        hole_d = random.randint(hole_size[0], hole_size[1])
        hole_h = random.randint(hole_size[2], hole_size[3])
        hole_w = random.randint(hole_size[4], hole_size[5])
        if hole_area:
            pass  # later implement what happens if the area where to put the hole is fixed
        offset_x = random.randint(0, mask_w - hole_w)
        offset_y = random.randint(0, mask_h - hole_h)
        offset_z = random.randint(0, mask_d - hole_d)

        mask[i, :, offset_z:offset_z + hole_d, offset_y:offset_y+hole_h, offset_x:offset_x+hole_w] = 1.0
        return mask

