"""
Routine for the creation of the parallelepiped that composed the training set
"""
import torch
import numpy as np

constant_latitude = 111  # 1° of latitude corresponds to 111 km
constant_longitude = 111  # 1° of latitude corresponds to 111 km


def create_box(batch, number_channel, lat, lon, depth, resolution):
    """
    Function that creates the tensor that will be filled with data
    batch = batch size/ batch number ?
    number_channel = number of channel (i.e. unknowns we want to predict)
    lat = (lat_min, lat_max)
    lon = (lon_min, lon_max)
    depth = (depth_min, depth_max) in km
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
    output = tensor zeros (MB, C, D, H, W)
    """
    lat_min, lat_max = lat
    lon_min, lon_max = lon
    depth_min, depth_max = depth
    w_res, h_res, d_res = resolution
    w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1)
    h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d = np.int((depth_max - depth_min) / d_res + 1)
    empty_parallelepiped = torch.zeros(batch, number_channel, d, h, w)
    return empty_parallelepiped
