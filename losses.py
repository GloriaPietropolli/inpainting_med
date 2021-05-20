from torch.nn.functional import mse_loss


def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)


def completion_float_loss(training_x, output, mask):
    mask[mask == 0] = 2
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    return mse_loss(training_x * mask, output * mask)


def completion_sat_loss(training_x, output, mask):
    mask[mask == 0] = 2
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    masked_input, masked_output = training_x * mask, output*mask
    return mse_loss(masked_input, masked_output)
