import torch
import os

path_directory = os.getcwd()
directory_tensor = path_directory + '/tensor/'


def get_list_float_tensor():
    """
    created a list containing the my_tensor representing the float information uploaded
    """
    list_float_tensor = []
    directory_float = directory_tensor + 'float/'
    list_ptFIles = os.listdir(directory_float)
    for ptFiles in list_ptFIles:
        my_tensor = torch.load(directory_float + ptFiles)
        list_float_tensor.append(my_tensor)
    return list_float_tensor


list_float_tensor = get_list_float_tensor()
