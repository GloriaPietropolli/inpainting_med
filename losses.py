""""
Implementation of the losses for the different train of the model
"""

from torch.nn.functional import mse_loss


def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)


def completion_float_loss(training_x, output, mask):
    """
    compute the loss only where we have the float information
    """
    mask[mask == 0] = 2
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    return mse_loss(training_x * mask, output * mask)


def completion_sat_loss(training_x, output, mask):
    """
    compute the loss only where we have the sat information
    """
    mask[mask == 0] = 2
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    return mse_loss(training_x * mask, output*mask)
