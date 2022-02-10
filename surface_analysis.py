import os
import random
import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from get_dataset import get_list_model_tensor
from completion import CompletionN
from utils import generate_input_mask
from normalization import Normalization
from mean_pixel_value import MV_pixel
from make_datasets import find_index
from hyperparameter import latitude_interval, longitude_interval, depth_interval, resolution

sns.set(context='notebook', style='whitegrid')

epoch_float, lr_float = 25, 0.0001

name_model = "model_PHASE1_completion_epoch_1000_lrc_0.01"
# name_model = "model_completion_epoch_1000_1000_350_lrc_0.01_lrd_0.01"
# name_model = "model_completion_epoch_500_500_200_lrc_0.01_lrd_0.01+epoch_250_lr_0.001"

model_considered = 'model2015/' + name_model
path_model = os.getcwd() + '/model/' + model_considered + '.pt'
path_model_float = os.getcwd() + '/result2/' + name_model + '/' + str(epoch_float) + '/' + str(lr_float) + '/model.pt'
if not os.path.exists(path_model_float):
    flag_float = False
else:
    flag_float = True

dict_channel = {'temperature': 0, 'salinity': 1, 'oxygen': 2, 'chla': 3}
dict_channel = {'temperature': 0}

for variable in list(dict_channel.keys()):
    snaperiod = 25

    constant_latitude = 111  # 1° of latitude corresponds to 111 km
    constant_longitude = 111  # 1° of latitude corresponds to 111 km
    lat_min, lat_max = latitude_interval
    lon_min, lon_max = longitude_interval
    depth_min, depth_max = depth_interval
    w_res, h_res, d_res = resolution

    w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1) - 2
    h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d_d = np.int((depth_max - depth_min) / d_res + 1) - 1
    d = d_d - 1

    latitude_interval = (lat_min + (lat_max - lat_min) / w, lat_max - (lat_max - lat_min) / w)
    depth_interval = (depth_min + (depth_max - depth_min) / d, depth_max - (depth_max - depth_min) / d)
    depth_interval_d = (depth_min, depth_max - (depth_max - depth_min) / d_d)

    hole_min_d, hole_max_d = 10, 20
    hole_min_h, hole_max_h = 30, 50
    hole_min_w, hole_max_w = 30, 50

    mvp_dataset = get_list_model_tensor()
    mvp_dataset, mean_model, std_model = Normalization(mvp_dataset)
    mean_value_pixel = MV_pixel(mvp_dataset)  # compute the mean of the channel of the training set
    mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 4, 1, 1, 1))

    model = CompletionN()
    model.load_state_dict(torch.load(path_model))  # network trained only with model information
    model.eval()

    if flag_float:
        model_float = CompletionN()
        model_float.load_state_dict(torch.load(path_model_float))  # network adjusted with float information
        model_float.eval()

    path_fig = os.getcwd() + '/analysis_result/surface_time_series/' + name_model[23:] + '/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    if flag_float:
        path_fig = path_fig + str(epoch_float) + '_' + str(lr_float) + '/'
        if not os.path.exists(path_fig):
            os.mkdir(path_fig)
    path_fig = path_fig + variable
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)

    months = ["0" + str(month) for month in range(1, 10)] + [str(month) for month in range(10, 52)]

    means_phys, means_mod = [], []
    if flag_float:
        means_flo = []
    std_phys, std_mod = [], []
    if flag_float:
        std_flo = []

    for month in months:  # iteration among months
        if month[-1] == "0":
            month = month[:-1]
        datetime = "2015." + month
        data_tensor = os.getcwd() + '/tensor/model2015_n/datetime_' + str(
            datetime) + '.pt'  # get the data_tensor correspondent to the datetime of emodnet sample to feed the nn (
        # NORMALIZED!!)
        data_tensor = torch.load(data_tensor)

        # TEST ON THE MODEL'S MODEL AND THE FLOAT MODEL WITH SAME HOLE
        with torch.no_grad():
            training_mask = generate_input_mask(
                shape=(data_tensor.shape[0], 1, data_tensor.shape[2], data_tensor.shape[3], data_tensor.shape[4]),
                hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
            data_tensor_mask = data_tensor - data_tensor * training_mask + mean_value_pixel * training_mask
            input = torch.cat((data_tensor_mask, training_mask), dim=1)

            model_result = model(input.float())
            if flag_float:
                float_result = model_float(input.float())

        mean_unkn = mean_model[0, dict_channel[variable], 0, 0, 0]
        std_unkn = std_model[0, dict_channel[variable], 0, 0, 0]

        depth_index = 0  # here I consider only surface data

        unkn_phys = data_tensor[:, dict_channel[variable], depth_index, :, :]
        unkn_model = model_result[:, dict_channel[variable], depth_index, :, :]
        if flag_float:
            unkn_float = float_result[:, dict_channel[variable], depth_index, :, :]

        unkn_phys = unkn_phys * std_unkn + mean_unkn
        unkn_model = unkn_model * std_unkn + mean_unkn
        if flag_float:
            unkn_float = unkn_float * std_unkn + mean_unkn

        means_phys.append(torch.mean(unkn_phys))
        means_mod.append(torch.mean(unkn_model))
        if flag_float:
            means_flo.append(torch.mean(unkn_float))

        std_phys.append(torch.std(unkn_phys))
        std_mod.append(torch.std(unkn_model))
        if flag_float:
            std_flo.append(torch.std(unkn_float))

    if flag_float:
        zip_result = zip(means_mod, std_mod, means_flo, std_flo, means_phys, std_phys)
        if variable == "oxygen":
            zip_result = [x for x in zip_result if x[4] > 50]
        if variable == "salinity":
            zip_result = [x for x in zip_result if x[4] > 10]
        if variable == "temperature":
            zip_result = [x for x in zip_result if x[4] > 8 or x[0] > 8]
        means_mod, std_mod, means_flo, std_flo, means_phys, std_phys = zip(*zip_result)
    else:
        zip_result = zip(means_mod, std_mod, means_phys, std_phys)
        if variable == "oxygen":
            zip_result = [x for x in zip_result if x[2] > 50]
        if variable == "salinity":
            zip_result = [x for x in zip_result if x[2] > 10]
        if variable == "temperature":
            zip_result = [x for x in zip_result if x[2] > 8 or x[0] > 8]
        means_mod, std_mod, means_phys, std_phys = zip(*zip_result)

    plt.plot(means_phys, color="slategray", linestyle='--', marker='v', alpha=0.8)
    plt.plot(means_mod, color="deeppink", linestyle='--', marker='o', alpha=0.8)
    if flag_float:
        plt.plot(means_flo, color="purple", linestyle='--', marker='*', alpha=0.8)
        plt.legend(["physical model", "CNN + GAN model", "CNN + GAN + float"])
    else:
        plt.legend(["physical model", "CNN"])
    plt.ylabel(variable)
    plt.suptitle("MEAN")
    plt.title("Surface time series of the " + variable)
    plt.savefig(path_fig + '/' + variable + '_ts_mean.png')
    plt.show()
    plt.close()

    plt.plot(std_phys, color="slategray", linestyle='--', marker='v', alpha=0.8)
    plt.plot(std_mod, color="deeppink", linestyle='--', marker='o', alpha=0.8)
    if flag_float:
        plt.plot(std_flo, color="purple", linestyle='--', marker='*', alpha=0.8)
        plt.legend(["physical model", "CNN + GAN model", "CNN + GAN + float"])
    else:
        plt.legend(["physical model", "CNN"])
    plt.ylabel(variable)
    plt.suptitle("STD")
    plt.title("Surface time series of the " + variable)
    plt.savefig(path_fig + '/' + variable + '_ts_std.png')
    plt.show()
    plt.close()

    plt.plot(means_phys, color="slategray", linestyle='--', marker='h', alpha=0.8)
    plt.fill_between(range(len(means_phys)),
                     np.array(means_phys) - np.array(std_phys) / 2,
                     np.array(means_phys) + np.array(std_phys) / 2,
                     color="slategray",
                     alpha=0.2
                     )
    plt.plot(means_mod, color="deeppink", linestyle='--', marker='o', alpha=0.8)
    plt.fill_between(range(len(means_mod)),
                     np.array(means_mod) - np.array(std_mod) / 2,
                     np.array(means_mod) + np.array(std_mod) / 2,
                     color="deeppink",
                     alpha=0.2
                     )
    if flag_float:
        plt.plot(means_flo, color="purple", linestyle='--', marker='*', alpha=0.8)
        plt.fill_between(range(len(means_flo)),
                         np.array(means_flo) - np.array(std_flo) / 2,
                         np.array(means_flo) + np.array(std_flo) / 2,
                         color="purple",
                         alpha=0.2
                         )
        plt.legend(["physical model", "CNN + GAN model", "CNN + GAN + float"])
    else:
        plt.legend(["physical model", "CNN"])
    plt.ylabel(variable)
    plt.suptitle("MEAN+STD")
    plt.title("Surface time series of the " + variable)
    plt.savefig(path_fig + '/' + variable + '_ts_mean+std.png')
    plt.show()
    plt.close()

