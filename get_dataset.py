import torch
import os
from hyperparameter import *

path_directory = os.getcwd()
directory_tensor = path_directory + '/tensor/'


def get_list_float_tensor():
    """
    created a list containing the my_tensor representing the FLOAT information uploaded
    """
    list_float_tensor = []
    directory_float = directory_tensor + 'float/'
    list_ptFIles = os.listdir(directory_float)
    for ptFiles in list_ptFIles:
        my_tensor = torch.load(directory_float + ptFiles)
        list_float_tensor.append(my_tensor)
    return list_float_tensor


def get_list_model_tensor():
    """
    created a list containing the my_tensor representing the MODEL information uploaded
    """
    model_tensor = []
    directory_float = directory_tensor + 'model2015/'
    list_ptFIles = os.listdir(directory_float)
    for ptFiles in list_ptFIles:
        my_tensor = torch.load(directory_float + ptFiles)
        model_tensor.append(my_tensor)
    return model_tensor


def get_list_sat_tensor():
    """
    created a list containing the my_tensor representing the MODEL information uploaded
    """
    sat_tensor = []
    directory_sat = directory_tensor + 'sat/'
    list_ptFIles = os.listdir(directory_sat)
    for ptFiles in list_ptFIles:
        my_tensor = torch.load(directory_sat + ptFiles)
        sat_tensor.append(my_tensor[:, :, :-1, :, :])
    return sat_tensor


directory_weight = path_directory + '/weight_tensor/'


def get_list_float_weight_tensor():
    """
    created a list containing the my_tensor representing the FLOAT information uploaded
    """
    list_float_tensor = []
    directory_float = directory_weight + 'float/'
    list_ptFIles = os.listdir(directory_float)
    for ptFiles in list_ptFIles:
        my_tensor = torch.load(directory_float + ptFiles)
        list_float_tensor.append(my_tensor)
    return list_float_tensor


if kindof == 'float':
    list_float_tensor = get_list_float_tensor()
if kindof == 'model2015':
    list_model_tensor = get_list_model_tensor()
if kindof == 'sat':
    list_sat_tensor = get_list_sat_tensor()
