"""
Routine for the creation of the parallelepiped that composed the training set
"""
import torch
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from plot_tensor import plot_routine

constant_latitude = 111  # 1° of latitude corresponds to 111 km
constant_longitude = 111  # 1° of latitude corresponds to 111 km
float_path = "../FLOAT_BIO/"


def read_date_time_float(date_time):
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


def read_date_time_sat(date_time):
    """
    Take as input a date-time in str format and decode it in a format considering only year and month
    year + 0.01 * week
    """
    year = np.int(date_time[0:4])
    month = np.int(date_time[4:6])
    day = np.int(date_time[6:8])
    week = np.int(month * 4 + day / 7)
    date_time_decoded = year + 0.01 * week
    return date_time_decoded


def to_depth(press, latitude):
    """
    convert press input in depth one
    press = pressure in decibars
    lat = latitude in deg
    depth = depth in metres
    """
    x = np.sin(latitude / 57.29578)
    x = x * x
    gr = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * press
    depth = (((-1.82e-15 * press + 2.279e-10) * press - 2.2512e-5) * press + 9.72659) * press / gr
    return depth


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


def insert_model_values(year, lat_limits, lon_limits, depth_limits, resolution):
    """
        function that update the parallelepiped updating all the voxel with MODEL information
        year = folder of the year we are considering
        lat_limits = (lat_min, lat_max)
        lon_limits = (lon_min, lon_max)
        depth_limits = (depth_min, depth_max) in km
        resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
        """
    lat_min, lat_max = lat_limits
    lon_min, lon_max = lon_limits
    depth_min, depth_max = depth_limits
    w_res, h_res, d_res = resolution

    w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1)
    h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d = np.int((depth_max - depth_min) / d_res + 1)

    path_model = os.getcwd() + "/MODEL/" + str(year) + '/phys/'
    model_files = os.listdir(path_model)
    for model_file in model_files:
        file = path_model + model_file
        ds = nc.Dataset(file)

        time = model_file[4:12]
        time = read_date_time_sat(time)
        index = list_data_time.index(time)  # index input tens considered, i.e. the one to upd
        select_parallelepiped = list_parallelepiped[index]  # parall we are modifying

        latitude_list = ds['nav_lat'][:].data
        longitude_list = ds['nav_lon'][:].data
        depth_list = ds['deptht'][:].data

        temp_tens = torch.tensor(ds['votemper'][:].data)[0, :, :, :]  # tensor indexes as temp(depth, y, x)
        salinity_tens = torch.tensor(ds['vosaline'][:].data)[0, :, :, :]

        for i in range(len(latitude_list)):  # indexing over the latitude (3rd component of the tensor)
            for j in range(len(longitude_list)):  # indexing over the longitude (2nd component of the tensor)
                for k in range(len(depth_list)):  # indexing over the depth (1st component of the tensor)
                    latitude = latitude_list[i]
                    longitude = longitude_list[j]
                    depth = depth_list[k]

                    temp = float(temp_tens[k, i, j].item())
                    salinity = float(salinity_tens[k, i, j].item())

                    if lat_max > latitude > lat_min:
                        if lon_max > longitude > lon_min:
                            if depth_max > depth > depth_min:
                                latitude_index = find_index(latitude, lat_limits, w)
                                longitude_index = find_index(longitude, lon_limits, h)
                                depth_index = find_index(depth, depth_limits, d)

                                if -3 < temp < 40:
                                    select_parallelepiped[0, 0, depth_index, longitude_index, latitude_index] = temp
                                if 2 < salinity < 41:
                                    select_parallelepiped[0, 1, depth_index, longitude_index, latitude_index] = salinity

    return


def insert_sat_values(lat_limits, lon_limits, depth_limits, resolution):
    """
    function that update the parallelepiped updating the voxel on the surfaces
    the only information provided is the 'CHL' ones
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
    """
    lat_min, lat_max = lat_limits
    lon_min, lon_max = lon_limits
    depth_min, depth_max = depth_limits
    w_res, h_res, d_res = resolution

    w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1)
    h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d = np.int((depth_max - depth_min) / d_res + 1)

    path_sat = os.getcwd() + "/WEEKLY_1_1km/"
    sat_measurement = os.listdir(path_sat)

    for sat_file in sat_measurement:
        file = path_sat + sat_file
        ds = nc.Dataset(file)

        data_time = sat_file[0:8]
        data_time = read_date_time_sat(data_time)
        if data_time < 2015.01:
            continue
        index = list_data_time.index(data_time)  # index input tens considered, i.e. the one to upd
        select_parallelepiped = list_parallelepiped[index]  # parall we are modifying

        latitude_list = ds['lat'][:].data
        longitude_list = ds['lon'][:].data

        depth = float(ds['depth'][:].data)
        depth_index = find_index(depth, depth_limits, d)  # 0 bc we are on the surfaces

        matrix_chl = ds['CHL'][0::].data[0]

        for i in range(len(latitude_list)):
            for j in range(len(longitude_list)):
                lat = latitude_list[i]
                lon = longitude_list[j]
                chl = matrix_chl[i, j]
                if chl == -999:
                    continue
                if lat_max > lat > lat_min:
                    if lon_max > lon > lon_min:
                        lat_index = find_index(lat, lat_limits, w)
                        lon_index = find_index(lon, lon_limits, h)
                        select_parallelepiped[0, 3, depth_index, lon_index, lat_index] = float(chl)
    return


def insert_float_values(lat_limits, lon_limits, depth_limits, resolution):
    """
    Function that update the parallelepiped updating the voxel where the float info is available
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
    """
    set_measurement = set()
    lat_min, lat_max = lat_limits
    lon_min, lon_max = lon_limits
    depth_min, depth_max = depth_limits
    w_res, h_res, d_res = resolution

    w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1)
    h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d = np.int((depth_max - depth_min) / d_res + 1)

    list_data = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
    list_datetime = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()  # at
    # each element of list_data corresponds one element of list_datetime

    for i in range(np.size(list_data)):  # indexing on list_data and list_datetime also
        path_current_float = float_path + "data/" + list_data[i]
        ds = nc.Dataset(path_current_float)

        var_list = []
        for var in ds.variables:
            var_list.append(var)

        # time = read_date_time_float(ds['DATE_CREATION'][:].data)  # 2015.22
        datetime = list_datetime[i]
        time = read_date_time_sat(datetime)
        if not 2020 > time > 2015:
            print('time out of range', time)
            continue
        index = list_data_time.index(time)  # index input tens considered, i.e. the one to upd
        select_parallelepiped = list_parallelepiped[index]  # parall we are modifying

        lat = float(ds['LATITUDE'][:].data)  # single value
        lon = float(ds['LONGITUDE'][:].data)  # single value

        lat_index = find_index(lat, lat_limits, w)
        lon_index = find_index(lon, lon_limits, h)

        pres_list = ds['PRES'][:].data[0]  # list of value
        depth_list = []
        for pres in pres_list:
            depth_list.append(to_depth(pres, lat))

        temp = ds['TEMP'][:].data[0]  # list of value
        salinity = ds['PSAL'][:].data[0]  # list of value
        if 'DOXY' not in var_list:
            continue
        doxy = ds['DOXY'][:].data[0]  # list of value

        if lat_max > lat > lat_min:
            if lon_max > lon > lon_min:
                for depth in depth_list:
                    if depth_max > depth > depth_min:
                        depth_index = find_index(depth, depth_limits, d)
                        channel_index = np.where(depth_list == depth)[0][0]

                        temp_v, salinity_v, doxy_v = temp[channel_index], salinity[channel_index], doxy[
                            channel_index]

                        if not -3 < temp_v < 40:
                            print('invalid temperature found', temp_v)
                        else:
                            select_parallelepiped[0, 0, depth_index, lon_index, lat_index] = float(temp_v)  # update
                            # first channel

                        if not 2 < salinity_v < 41:
                            print('invalid psal found', salinity_v)
                        else:
                            select_parallelepiped[0, 1, depth_index, lon_index, lat_index] = float(
                                salinity_v)  # update second channel

                        if not -5 < doxy_v < 600:
                            print('invalid doxy found', doxy_v)
                            select_parallelepiped[0, 2, depth_index, lon_index, lat_index] = float(doxy_v)  # update
                            # third channel

                        set_measurement.add(time)

    return set_measurement


def save_result(tensor_list, dir):
    if dir == 'empty':
        path = os.getcwd() + '/result/empty'
    if dir == 'float':
        path = os.getcwd() + '/result/float'
    if dir == 'sat':
        path = os.getcwd() + '/result/sat'
    if dir == 'mod':
        path = os.getcwd() + '/result/mod'
    else:
        print('not saving ')
        # return
    for i in range(len(tensor_list)):
        date_time = list_data_time[i]
        np.savetxt('tens_' + str(date_time) + '.csv', tensor_list[i], delimiter=',')


# box = create_box(1, 5, (36, 44), (2, 9), (0, 0.6), (12, 12, 0.01))
batch = 1
number_channel = 4  # 1: temp, 2:salinity, 3:doxy, 4: chla
latitude_interval = (36, 44)
longitude_interval = (2, 9)
depth_interval = (1, 200)
resolution = (12, 12, 50)
list_data_time = create_list_date_time((2015, 2022))
list_parallelepiped = [
    create_box(batch, number_channel, latitude_interval, longitude_interval, depth_interval, resolution) for i in
    range(len(list_data_time))]

# set_measurement = insert_float_values(latitude_interval, longitude_interval, depth_interval, resolution)
# insert_sat_values(latitude_interval, longitude_interval, depth_interval, resolution)
# insert_model_values(2015, latitude_interval, longitude_interval, depth_interval, resolution)
channels = [0]
plot_routine('model2015', list_parallelepiped, list_data_time, channels)

# save_result(list_parallelepiped, 'empty')
