"""
Routine for the creation of the parallelepiped that composed the training set
"""
import torch
import netCDF4 as nc
import numpy as np
import pandas as pd

from index_float import list_float_total

constant_latitude = 111  # 1° of latitude corresponds to 111 km
constant_longitude = 111  # 1° of latitude corresponds to 111 km
float_path = "../FLOAT_BIO/"


def read_date_time(date_time):
    """
    Take as input a date-time in format UTF-8 and decode it in a format considering only year and month
    year + 0.01 * week
    """
    date_time_decoded = ''
    for i in range(0, 14):
        new_digit = date_time[i].decode('UTF-8')
        date_time_decoded += new_digit
    year = np.int(date_time_decoded[0:4])
    month = np.int(date_time_decoded[4:6])
    day = np.int(date_time_decoded[6:8])
    week = np.int(month * 4 + day / 7)
    date_time_decoded = year + 0.01 * week
    return date_time_decoded


def create_list_date_time(years_consider):
    """
    Creation of a list containing date_time reference for training dataset
    years_consider = (first year considered, last year considered)
    interval_of_time = intervals within measurement are aggregated
    """
    first_year_considered, last_year_considered = years_consider
    total_list = []
    for year in np.arange(first_year_considered, last_year_considered):
        lists = np.arange(year, year + 0.53, 0.01)
        for i in range(len(lists)):
            lists[i] = round(lists[i], 2)
        lists = lists.tolist()
        total_list = total_list + lists
    return total_list


list_data_time = create_list_date_time((2015, 2022))


def create_box(batch, number_channel, lat, lon, depth, resolution):
    """
    Function that creates the EMPTY tensor that will be filled with data
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


def find_index(lat, lat_limits, lat_size):
    """
    Function that given a latitude/longitude/depth as input return the index where to place it in the tensor
    lat = latitude considered
    lat_limits = (lat_min, lat_max)
    lat_size = dimension of latitude dmensin in the tensor
    """
    lat_min, lat_max = lat_limits
    lat_res = (lat_max - lat_min) / lat_size
    lat_index = np.int((lat - lat_min) / lat_res)
    return lat_index


def insert_model_values():
    pass


def insert_sat_values():
    pass


def insert_float_values(lat_limits, lon_limits, depth_limits, resolution):
    """
    Function that update the parallelepiped updating the voxel where the float info is available
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    parallelepiped = parallelepiped to fill with float values
    index_parallelepiped = represents the index in the input parallelepiped list, i.e. the week considered
                           for example index_parallelepiped = 1 means we are considering first week of 2015
    """
    lat_min, lat_max = lat_limits
    lon_min, lon_max = lon_limits
    depth_min, depth_max = depth_limits
    w_res, h_res, d_res = resolution

    w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1)
    h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d = np.int((depth_max - depth_min) / d_res + 1)

    list_data = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()

    for float_number in list_float_total:
        for i in range(np.size(list_data)):  # indexing on list_data
            if float_number == list_data[i][0:7]:
                path_current_float = float_path + "data/" + list_data[i]
                ds = nc.Dataset(path_current_float)

                var_list = []
                for var in ds.variables:
                    var_list.append(var)

                time = read_date_time(ds['DATE_CREATION'][:].data)  # 2015.22

                lat = float(ds['LATITUDE'][:].data)  # single value
                lon = float(ds['LONGITUDE'][:].data)  # single value

                lat_index = find_index(lat, lat_limits, w)
                lon_index = find_index(lon, lon_limits, h)

                depth_list = ds['PRES'][:].data[0]  # list of value

                temp = ds['TEMP'][:].data[0]  # list of value
                salinity = ds['PSAL'][:].data[0]  # list of value
                if 'DOXY' not in var_list:
                    continue
                doxy = ds['DOXY'][:].data[0]  # list of value

                if lat_max > lat > lat_min:
                    if lon_max > lon > lon_min:
                        for depth in depth_list:
                            if depth_max > depth > depth_min:
                                index = list_data_time.index(time)  # index input tens considered, i.e. the one to upd
                                select_parallelepiped = list_parallelepiped[index]  # parall we are modifying

                                depth_index = find_index(depth, depth_limits, d)
                                channel_index = np.where(depth_list == depth)[0][0]

                                select_parallelepiped[:, 0, depth_index, lon_index, lat_index] = torch.tensor(
                                    temp[channel_index])  # update first channel
                                select_parallelepiped[:, 1, depth_index, lon_index, lat_index] = torch.tensor(
                                    salinity[channel_index])  # update second channel
                                select_parallelepiped[:, 2, depth_index, lon_index, lat_index] = torch.tensor(
                                    doxy[channel_index])  # update third channel
    return


# box = create_box(1, 5, (36, 44), (2, 9), (0, 0.6), (12, 12, 0.01))
batch = 1
number_channel = 3  # 1: temp, 2:salinity, 3:doxy
lat = (36, 44)
lon = (2, 9)
depth = (1, 100)
resolution = (12, 12, 1)
list_parallelepiped = [create_box(batch, number_channel, lat, lon, depth, resolution)] * len(list_data_time)
insert_float_values(lat, lon, depth, resolution)
