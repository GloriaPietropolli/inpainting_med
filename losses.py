from torch.nn.functional import mse_loss
import torch


def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)


def completion_float_loss(training_x, output, mask):
    mask[mask == 0] = 2
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    return mse_loss(training_x * mask, output * mask)


def completion_sat_loss(input, output, mask, weight):
    masked_input, masked_output = input * mask, output * mask
    weighted_input, weighted_output = masked_input * weight, masked_output * weight
    weighted_input_fl = weighted_input[:, :, 0, :, :]
    weighted_output_fl = weighted_input[:, :, 0, :, :]
    return mse_loss(weighted_output_fl, weighted_input_fl)
