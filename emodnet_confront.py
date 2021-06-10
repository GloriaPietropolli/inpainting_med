import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from get_dataset import get_list_model_tensor
from completion import CompletionN
from utils import generate_input_mask
from normalization import Normalization
from mean_pixel_value import MV_pixel
from make_datasets import find_index
from hyperparameter import latitude_interval, longitude_interval, depth_interval, resolution

variable = 'temperature'
snaperiod = 25

constant_latitude = 111  # 1° of latitude corresponds to 111 km
constant_longitude = 111  # 1° of latitude corresponds to 111 km
lat_min, lat_max = latitude_interval
lon_min, lon_max = longitude_interval
depth_min, depth_max = depth_interval
w_res, h_res, d_res = resolution

w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1)
w_m = w - 2
h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
d = np.int((depth_max - depth_min) / d_res + 1) - 2
d_m = d - 2

latitude_interval_m = (lat_min + (lat_max - lat_min)/w, lat_max - (lat_max - lat_min)/w)
depth_interval_m = (depth_min + (depth_max - depth_min)/d, depth_max - (depth_max - depth_min)/d)

hole_min_d, hole_max_d = 10, 20
hole_min_h, hole_max_h = 30, 50
hole_min_w, hole_max_w = 30, 50

mvp_dataset = get_list_model_tensor()
mvp_dataset, mean_model, std_model = Normalization(mvp_dataset)
mean_value_pixel = MV_pixel(mvp_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 4, 1, 1, 1))

epoch_model, lr_model = 501, 0.01
epoch_float, lr_float = 50, 0.0001

model_considered = 'model2015_c/model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(lr_model)
path_emodnet = os.getcwd() + '/emodnet/' + 'emodnet2015_' + variable + '.pt'
path_model = os.getcwd() + '/model/' + model_considered + '.pt '
path_model_float = os.getcwd() + '/result2/model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(lr_model) + '/' + str(epoch_float) + '/' + str(lr_float) + '/model.pt'

emodnet = torch.load(path_emodnet)

model = CompletionN()
model.load_state_dict(torch.load(path_model))  # network trained only with model information
model.eval()

model_float = CompletionN()
model_float.load_state_dict(torch.load(path_model_float))  # network adjusted with float information
model_float.eval()

win_m, win_f, win_d = 0, 0, 0
for i in range(emodnet.shape[0]):  # for every sample considered
    datetime = round(emodnet[i, 0].item(), 2)
    data_tensor = os.getcwd() + '/tensor/model2015_n/datetime_' + str(datetime) + '.pt'
    # get the data_tensor correspondent to the datetime of emodnet sample to feed the nn (NORMALIZED!!)
    if not os.path.exists(data_tensor):
        continue

    data_tensor = torch.load(data_tensor)

    # TEST ON THE MODEL'S MODEL AND THE FLOAT MODEL WITH SAME HOLE
    with torch.no_grad():
        training_mask = generate_input_mask(
            shape=(data_tensor.shape[0], 1, data_tensor.shape[2], data_tensor.shape[3], data_tensor.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
        data_tensor_mask = data_tensor - data_tensor * training_mask + mean_value_pixel * training_mask
        input = torch.cat((data_tensor_mask, training_mask), dim=1)

        model_result = model(input.float())
        float_result = model_float(input.float())

    emodnet_result = emodnet[i, :]
    emodnet_lat = emodnet_result[1].item()
    emodnet_lon = emodnet_result[2].item()
    emodnet_depth = emodnet_result[4].item()
    emodnet_temp = emodnet_result[3].item()

    mean_temp = mean_model[0, 0, 0, 0, 0]
    std_temp = std_model[0, 0, 0, 0, 0]
    # emodnet_temp = (emodnet_temp - mean_temp)/std_temp

    emodnet_lat_index = find_index(emodnet_lat, latitude_interval_m, w_m)
    if emodnet_lat_index >= 73:
        continue
    emodnet_lon_index = find_index(emodnet_lon, longitude_interval, h)
    emodnet_depth_index = find_index(emodnet_depth, depth_interval_m, d_m)
    if emodnet_depth_index >= 29:
        continue

    data_lat_index = find_index(emodnet_lat, latitude_interval, w)
    data_lon_index = find_index(emodnet_lon, longitude_interval, h)
    data_depth_index = find_index(emodnet_depth, depth_interval, d)

    temp_data = data_tensor[:, 0, data_depth_index, data_lon_index, data_lat_index]
    temp_model = model_result[:, 0, emodnet_depth_index, emodnet_lon_index, emodnet_lat_index]
    temp_float = float_result[:, 0, emodnet_depth_index, emodnet_lon_index, emodnet_lat_index]

    temp_data = temp_data * std_temp + mean_temp
    temp_model = temp_model * std_temp + mean_temp
    temp_float = temp_float * std_temp + mean_temp

    diff_d = np.abs(emodnet_temp - temp_data)
    diff_m = np.abs(emodnet_temp - temp_model)
    diff_f = np.abs(emodnet_temp - temp_float)

    if diff_f <= diff_m and diff_f <= diff_d:
        win_f = win_f + 1
    if diff_m < diff_f and diff_m < diff_d:
        win_m = win_m + 1
    if diff_d < diff_f and diff_d < diff_m:
        win_d = win_d + 1

    if i % snaperiod == 0:
        plt.bar(np.arange(3), height=[win_d, win_m, win_f])
        plt.xticks(np.arange(3), ['data', 'model', 'float'])
        plt.savefig(os.getcwd() + '/emodnet/comparison/comparison_' + str(i) + '.png')
        plt.close()

    # print('\ndata temp    : ', temp_data.item())
    # print('model temp   : ', temp_model.item())
    # print('model float  : ', temp_float.item())
    # print('emodnet temp : ', emodnet_temp)


