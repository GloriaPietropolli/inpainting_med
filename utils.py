import random
import torch
import torchvision.transforms as transforms
import numpy as np


def generate_input_mask(shape, hole_size, hole_area=None, number_holes=1):
    """
    shape = (B, C, D, H, W), C=1 bc it iis only necessary one channel
    hole_size = (hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w)
    hole_area = (left_corner, width, height, depth)
                contains the area where the holes are generated
                used as input value for the LOCAL DISCRIMINATOR
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
            area_x_min, area_y_min, area_z_min = hole_area[0]
            area_w, area_h, area_d = hole_area[1]
            offset_x = random.randint(area_x_min, area_x_min + area_w - hole_w)
            offset_y = random.randint(area_y_min, area_y_min + area_h - hole_h)
            offset_z = random.randint(area_z_min, area_z_min + area_d - hole_d)
        else:
            offset_x = random.randint(0, mask_w - hole_w)
            offset_y = random.randint(0, mask_h - hole_h)
            offset_z = random.randint(0, mask_d - hole_d)

        mask[i, :, offset_z:offset_z + hole_d, offset_y:offset_y + hole_h, offset_x:offset_x + hole_w] = 1.0
        return mask


def generate_hole_area(size, mask_size):
    """
    total_size = total size of the area (W,H,D)
    mask_size = size of the input mask
    """
    mask_w, mask_h, mask_d = mask_size
    area_w, area_h, area_d = size
    offset_x = random.randint(0, mask_w - area_w)
    offset_y = random.randint(0, mask_h - area_h)
    offset_z = random.randint(0, mask_d - area_d)
    return ((offset_x, offset_y, offset_z), (area_w, area_h, area_d))


def crop(x, area):
    """
    x = torch.tensor (N,C,D,H,W)
    area = ((left corner of the area to be cropped), (depth, width, height to be crop))
    """
    x_left_corner, y_left_corner, z_left_corner = area[0]
    w, h, d = area[1]
    cropped_tensor = x[:, :, z_left_corner:z_left_corner + d, y_left_corner:y_left_corner + h,
                       x_left_corner:x_left_corner + w]
    return cropped_tensor
