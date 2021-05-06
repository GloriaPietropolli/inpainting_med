from get_dataset import *
from hyperparameter import *
from plot_tensor import *
from mean_pixel_value import MV_pixel, std_pixel

mean_value_pixel = MV_pixel(list_model_tensor)
mean_tensor = torch.tensor(mean_value_pixel.reshape(1, number_channel, 1, 1, 1))
std_value_pixel = std_pixel(list_model_tensor)
std_tensor = torch.tensor(std_value_pixel.reshape(1, number_channel, 1, 1, 1))


def Normalization(list_tensor):
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor
        tensor = tensor[:, :, :-1, :, :]
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list

