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

name_model = "model_completion_epoch_500_500_200_lrc_0.01_lrd_0.01"
model_considered = 'model2015/' + name_model
path_model = os.getcwd() + '/model/' + model_considered + '.pt'
path_model_float = os.getcwd() + '/result2/' + name_model + '/' + str(epoch_float) + '/' + str(lr_float) + '/model.pt'

dict_channel = {'temperature': 0, 'salinity': 1, 'oxygen': 2, 'chla': 3}

for variable in list(dict_channel.keys()):
    snaperiod = 25
    print("plotting " + variable + " vertical profile")

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

    model_float = CompletionN()
    model_float.load_state_dict(torch.load(path_model_float))  # network adjusted with float information
    model_float.eval()

    path_fig = os.getcwd() + '/analysis_result/profile/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = os.getcwd() + '/analysis_result/profile/' + name_model[23:] + '/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = path_fig + str(epoch_float) + '_' + str(lr_float) + '/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = path_fig + variable
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    if not os.path.exists(path_fig + '/mean/'):
        os.mkdir(path_fig + '/mean/')
    if not os.path.exists(path_fig + '/std/'):
        os.mkdir(path_fig + '/std/')

    months = ["0" + str(month) for month in range(1, 10)] + [str(month) for month in range(10, 52)]

    for month in months:  # iteration among months
        if month[-1] == "0":
            month = month[:-1]
            # month = "12"
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
            float_result = model_float(input.float())

        mean_unkn = mean_model[0, dict_channel[variable], 0, 0, 0]
        std_unkn = std_model[0, dict_channel[variable], 0, 0, 0]

        means_phys, means_mod, means_flo = [], [], []
        std_phys, std_mod, std_flo = [], [], []

        for depth_index in range(0, d):  # iteration among depth
            unkn_phys = data_tensor[:, dict_channel[variable], depth_index, :, :]
            unkn_model = model_result[:, dict_channel[variable], depth_index, :, :]
            unkn_float = float_result[:, dict_channel[variable], depth_index, :, :]

            unkn_phys = unkn_phys * std_unkn + mean_unkn
            unkn_model = unkn_model * std_unkn + mean_unkn
            unkn_float = unkn_float * std_unkn + mean_unkn

            means_phys.append(torch.mean(unkn_phys))
            means_mod.append(torch.mean(unkn_model))
            means_flo.append(torch.mean(unkn_float))

            std_phys.append(torch.std(unkn_phys))
            std_mod.append(torch.std(unkn_model))
            std_flo.append(torch.std(unkn_float))

        zip_result = zip(means_mod, std_mod, means_flo, std_flo, means_phys, std_phys)
        zip_result = [x for x in zip_result if x[0] > 5]
        if zip_result:
            means_mod, std_mod, means_flo, std_flo, means_phys, std_phys = zip(*zip_result)
        else:
            continue

        plt.plot(means_phys, color="slategray", linestyle='--', marker='v', alpha=0.8, label="physical model")
        plt.plot(means_mod, color="deeppink", linestyle='--', marker='o', alpha=0.8, label="CNN + GAN model")
        plt.plot(means_flo, color="purple", linestyle='--', marker='*', alpha=0.8, label="CNN + GAN + float")
        plt.legend()
        plt.ylabel(variable)
        plt.xlabel("depth")
        plt.suptitle("Week " + str(month))
        plt.title("Profile of the mean of " + variable)
        plt.savefig(path_fig + '/mean/' + variable + '_pro_mean_' + month + '.png')
        plt.close()

        plt.plot(std_phys, color="slategray", linestyle='--', marker='v', alpha=0.8, label="physical model")
        plt.plot(std_mod, color="deeppink", linestyle='--', marker='o', alpha=0.8, label="CNN + GAN model")
        plt.plot(std_flo, color="purple", linestyle='--', marker='*', alpha=0.8, label="CNN + GAN + float")
        plt.legend()
        plt.ylabel(variable)
        plt.xlabel("depth")
        plt.suptitle("Week " + str(month))
        plt.title("Profile of the std of " + variable)
        plt.savefig(path_fig + '/std/' + variable + '_pro_std_' + month + '.png')
        plt.close()
