import torch


def make_float_mask(weight, mask):
    weight[weight == 1] = 2
    weight[weight == 0] = 1
    weight[weight == 2] = 0

    training_mask = mask + weight
    training_mask[training_mask == 2] = 1  # if I have pixel where I have both mask and weight

    sum_along_channel_mask = torch.sum(training_mask, 1)
    training_mask[:, 0:1, :, :] = sum_along_channel_mask
    training_mask = training_mask[:, 0:1, :, :]

    training_mask[:, 0, :, :][training_mask[:, 0, :, :] == 0] = 0
    training_mask[:, 0, :, :][training_mask[:, 0, :, :] != 0] = 1

    return training_mask
