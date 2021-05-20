import torch


def make_float_mask(weight, mask):
    """"
    adding to the original mask the complementary of the weight of the float (total mask for the training)
    """
    weight[weight == 1] = 2
    weight[weight == 0] = 1
    weight[weight == 2] = 0

    training_mask = mask + weight
    training_mask[training_mask == 2] = 1  # if I have pixel where I have both mask and weight

    sum_along_channel_mask = torch.sum(training_mask, 1)
    training_mask[:, 0:1, :, :] = sum_along_channel_mask  # NON CONSIDERI UNA DIM???
    training_mask = training_mask[:, 0:1, :, :]

    training_mask[:, 0, :, :][training_mask[:, 0, :, :] == 0] = 0
    training_mask[:, 0, :, :][training_mask[:, 0, :, :] != 0] = 1

    return training_mask


def make_sat_mask(weight, mask):
    """"
    adding to the original mask the complementary of the weight of the float (total mask for the training)
    """
    weight[weight == 1] = 2
    weight[weight == 0] = 1
    weight[weight == 2] = 0

    training_mask = mask + weight
    training_mask[training_mask == 2] = 1  # if I have pixel where I have both mask and weight

    training_mask = training_mask[:, 3:, :, :, :]

    return training_mask


def masked_sat_tensor(tensor, mean_value_pixel, mask):
    tensor_masked_02 = tensor[:, 0:3, :, :, :].float()
    tensor_masked_02[tensor_masked_02 == 0] = 1.0
    tensor_masked_02 = tensor_masked_02 * mean_value_pixel[:, 0:3, :, :, :]

    tensor_masked_3 = tensor[:, 3:, :, :, :] - tensor[:, 3:, :, :, :] * mask + mean_value_pixel[:, -1:,
                                                                                           :, :, :] * tensor[:,
                                                                                                      3:, :, :, :]
    training_x_masked = torch.cat((tensor_masked_02, tensor_masked_3), dim=1)
    return tensor

