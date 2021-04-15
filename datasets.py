"""
Routine for the creation of the parallelepiped that composed the training set
"""
import torch
import numpy as np
import netCDF4 as nc
import pandas as pd
from index_float import list_float_total

constant_latitude = 111  # 1° of latitude corresponds to 111 km
constant_longitude = 111  # 1° of latitude corresponds to 111 km
float_path = "../FLOAT_BIO/"


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


def insert_model_values():
    pass


def insert_sat_values():
    pass


def insert_float_values(parallelepiped=None):
    """
    Function that update the parallelepiped updating the voxel where the float info is avaiable
    """
    data = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 0]
    list_data = []
    for i in data:
        list_data.append(i)
    for float_number in list_float_total:
        for i in range(np.size(list_data)):
            if float_number == list_data[i][0:7]:
                path_current_float = float_path + str(float_number) + "/" + list_data[i]
                ds = nc.Dataset(path_current_float)
    return

# box = create_box(1, 5, (36, 44), (2, 9), (0, 0.6), (12, 12, 0.01))
# print(box.size())
insert_float_values()
