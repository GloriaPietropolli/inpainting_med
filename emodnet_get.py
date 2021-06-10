import torch
import netCDF4 as nc
import os
import numpy as np
from datetime import date, timedelta

path = os.getcwd() + '/emodnet/'
path_data = path + 'data_from_Mediterranean.nc'
ds = nc.Dataset(path_data)

datetime = ds['date_time'][:].data
latitude = ds['latitude'][:].data
longitude = ds['longitude'][:].data
temperature = ds['var2'][:].data
depth = ds['var1'][:].data
doxy = ds['var4'][:].data
chl = ds['var12'][:].data
psal = ds['var3'][:].data
typez = ds['metavar3'][:].data

n_samples, n_stations = 8167, 60876
n_input, n_output = 8, 0
absence_flag = -10 ** 10
start = date(1911, 1, 1)
observation_2015 = []


def preparation_data_single_station2(i, param):  # i=number of stations
    if typez[i] == b'B':  # se il dato è di tipo bottiglia
        data_single_station = torch.zeros(n_samples, n_input + n_output)

        days = datetime[i]
        delta = timedelta(days)
        offset = start + delta

        if offset.year != 2015:
            print('year ' + str(offset.year) + ' out of range')
            return None
        else:
            year = offset.year
            month = offset.month
            month = month - 1
            day = offset.day
            week = np.int(month * 4 + day / 7)
            date_time = year + 0.01 * week
            data_single_station[:, 0] = date_time * torch.ones(n_samples)

        if not 36 < latitude[i] < 44:
            print('latitude ' + str(latitude[i]) + ' out of range')
            return None
        else:
            data_single_station[:, 1] = latitude[i] * torch.ones(n_samples)

        if not 2 < longitude[i] < 9:
            print('latitude ' + str(longitude[i]) + ' out of range')
            return None
        else:
            data_single_station[:, 2] = longitude[i] * torch.ones(n_samples)

        data_single_station[:, 3] = torch.from_numpy(temperature[i, :])
        data_single_station[:, 4] = torch.from_numpy(depth[i, :])
        data_single_station[:, 5] = torch.from_numpy(doxy[i, :])
        data_single_station[:, 6] = torch.from_numpy(chl[i, :])
        data_single_station[:, 7] = torch.from_numpy(psal[i, :])

        if 'temperature' in param:
            data_single_station = data_single_station[data_single_station[:, 3] > absence_flag]

        data_single_station = data_single_station[data_single_station[:, 4] > absence_flag]

        if 'psal' in param:
            data_single_station = data_single_station[data_single_station[:, 7] > absence_flag]
        if 'doxy' in param:
            data_single_station = data_single_station[data_single_station[:, 5] > absence_flag]
        if 'chl' in param:
            data_single_station = data_single_station[data_single_station[:, 6] > absence_flag]

        return data_single_station


def merge_data_stations(n_stations_considered, param):
    my_data = torch.zeros(1, n_input + n_output)
    for i in range(0, n_stations_considered):
        data_to_add = preparation_data_single_station2(i, param)
        if data_to_add is not None:
            my_data = torch.cat((my_data, data_to_add), 0)
    return my_data


variable = ['temperature', 'doxy', 'psal']
emodnet = merge_data_stations(n_stations, variable)
# print(emodnet.shape)
torch.save(emodnet, path + 'emodnet2015.pt')
print(emodnet.shape)