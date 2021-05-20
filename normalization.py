"""
Normalization function that must be applied before proceeding with the training
bc the values of the unknown we want to estimate are way higher than 1
"""
from hyperparameter import *
from plot_tensor import *
from mean_pixel_value import MV_pixel, std_pixel


def Normalization(list_tensor):
    mean_value_pixel = MV_pixel(list_tensor)
    mean_tensor = torch.tensor(mean_value_pixel.reshape(1, number_channel, 1, 1, 1))
    std_value_pixel = std_pixel(list_tensor)
    std_tensor = torch.tensor(std_value_pixel.reshape(1, number_channel, 1, 1, 1))
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor
        tensor = tensor[:, :, :-1, :, 1:-1]
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list, mean_tensor, std_tensor


def Normalization_Float(list_tensor, mean_tensor, std_tensor):
    """
    normalization routine for FLOAT data
    """
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor
        tensor = tensor[:, :, :-1, :, 1:-1]
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list


def Normalization_Sat(list_tensor, mean_tensor, std_tensor):
    """
    normalization routine for FLOAT data
    """
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor
        tensor = tensor[:, :, :-1, :, 1:-1]
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list
