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

variable = 'temperature'
print("start...")
dict_channel = {'temperature': 0, 'salinity': 1, 'oxygen': 2, 'chla': 3}

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

epoch_float, lr_float = 25, 0.0001

# model_considered = 'model2015_c/model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(lr_model)
where = 'model2015/'

if where == 'model2015/':
    name_model = 'model_PHASE1_completion_epoch_200_lrc_0.01'
    # name_model = 'model_completion_epoch_250_150_150_lrc_0.01_lrd_0.01'
    model_considered = where + name_model
    path_emodnet = os.getcwd() + '/emodnet/' + 'emodnet2015.pt'
    path_model = os.getcwd() + '/model/' + model_considered + '.pt'
    path_model_float = os.getcwd() + '/result2/' + name_model + '/' + str(epoch_float) + '/' + str(lr_float) + '/model.pt'

if where == 'model2015_c/':
    epoch_model, lr_model = 501, 0.01
    name_model = 'model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(lr_model)
    model_considered = 'model2015_c/model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(lr_model)
    path_emodnet = os.getcwd() + '/emodnet/' + 'emodnet2015.pt'
    path_model = os.getcwd() + '/model/' + model_considered + '.pt '
    path_model_float = os.getcwd() + '/result2/model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(
    lr_model) + '/' + str(epoch_float) + '/' + str(lr_float) + '/model.pt'

print("loading emodnet dataset...")
emodnet = torch.load(path_emodnet)
print(emodnet.shape)

model = CompletionN()
model.load_state_dict(torch.load(path_model))  # network trained only with model information
model.eval()

model_float = CompletionN()
model_float.load_state_dict(torch.load(path_model_float))  # network adjusted with float information
model_float.eval()


win2_m, win2_f = 0, 0
win3_m, win3_f, win3_d = 0, 0, 0
diff2_m_, diff2_f_ = [], []
diff3_m_, diff3_f_, diff3_d_ = [], [], []
data_, model_, float_, emodnet_ = [], [], [], []

path_fig = os.getcwd() + '/emodnet/' + variable + '_comparison_' + name_model + '_float_' + str(epoch_float) \
           + '_' + str(lr_float)
if not os.path.exists(path_fig):
    os.mkdir(path_fig)

path_fig2 = path_fig + '/2comp'
if not os.path.exists(path_fig2):
    os.mkdir(path_fig2)
path_ist2 = path_fig2 + '/inst'
if not os.path.exists(path_ist2):
    os.mkdir(path_ist2)
path_bp2 = path_fig2 + '/bp'
if not os.path.exists(path_bp2):
    os.mkdir(path_bp2)

path_fig3 = path_fig + '/3comp'
if not os.path.exists(path_fig3):
    os.mkdir(path_fig3)
path_ist3 = path_fig3 + '/inst'
if not os.path.exists(path_ist3):
    os.mkdir(path_ist3)
path_bp3 = path_fig3 + '/bp'
if not os.path.exists(path_bp3):
    os.mkdir(path_bp3)

counter = 0
f = open(path_fig + "/differences.txt", "w+")
while counter < 1000:  # for every sample considered
    i = random.randint(0, emodnet.shape[0])
    datetime = round(emodnet[i, 0].item(), 2)
    data_tensor = os.getcwd() + '/tensor/model2015_n/datetime_' + str(datetime) + '.pt'
    # get the data_tensor correspondent to the datetime of emodnet sample to feed the nn (NORMALIZED!!)

    if not os.path.exists(data_tensor):
        print(data_tensor)
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
    if variable == 'temperature':
        emodnet_unkn = emodnet_result[3].item()
    if variable == 'oxygen':
        emodnet_unkn = emodnet_result[-3].item()
    if variable == 'chla':
        emodnet_unkn = emodnet_result[-2].item()
    if variable == 'salinity':
        emodnet_unkn = emodnet_result[-1].item()

    mean_unkn = mean_model[0, dict_channel[variable], 0, 0, 0]
    std_unkn = std_model[0, dict_channel[variable], 0, 0, 0]

    emodnet_lat_index = find_index(emodnet_lat, latitude_interval, w)
    if emodnet_lat_index >= 73:
        continue

    emodnet_lon_index = find_index(emodnet_lon, longitude_interval, h)

    emodnet_depth_index_d = find_index(emodnet_depth, depth_interval_d, d_d)
    emodnet_depth_index = find_index(emodnet_depth, depth_interval, d)
    if emodnet_depth_index >= 29:
        continue

    unkn_data = data_tensor[:, dict_channel[variable], emodnet_depth_index_d, emodnet_lon_index, emodnet_lat_index]
    unkn_model = model_result[:, dict_channel[variable], emodnet_depth_index, emodnet_lon_index, emodnet_lat_index]
    unkn_float = float_result[:, dict_channel[variable], emodnet_depth_index, emodnet_lon_index, emodnet_lat_index]

    unkn_data = unkn_data * std_unkn + mean_unkn
    unkn_model = unkn_model * std_unkn + mean_unkn
    unkn_float = unkn_float * std_unkn + mean_unkn

    if unkn_data < 0 or unkn_float < 0:
        continue

    diff_d = np.abs(emodnet_unkn - unkn_data)
    diff_m = np.abs(emodnet_unkn - unkn_model)
    diff_f = np.abs(emodnet_unkn - unkn_float)

    diff2_f_.append(diff_f)
    diff2_m_.append(diff_m)

    diff3_f_.append(diff_f)
    diff3_m_.append(diff_m)
    diff3_d_.append(diff_d)

    if diff_f <= diff_m and diff_f <= diff_d:
        win3_f = win3_f + 1
    if diff_m < diff_f and diff_m < diff_d:
        win3_m = win3_m + 1
    if diff_d < diff_f and diff_d < diff_m:
        win3_d = win3_d + 1

    if diff_f <= diff_m:
        win2_f = win2_f + 1
    else:
        win2_m = win2_m + 1

    if unkn_data == 0:
        continue

    counter = counter + 1
    if counter % snaperiod == 0:
        '''
        plt.bar(np.arange(3), height=[win3_d, win3_m, win3_f])
        plt.xticks(np.arange(3), ['data', 'model', 'float'])
        path_fig3 = path_fig + '/3comp'
        if not os.path.exists(path_fig3):
            os.mkdir(path_fig3)
        plt.savefig(path_fig3 + '/comparison_' + str(i) + '.png')
        plt.close()
        '''

        plt.bar(np.arange(2), height=[win2_m, win2_f])
        plt.xticks(np.arange(2), ['model', 'float'])
        plt.savefig(path_ist2 + '/comparison_' + str(counter) + '.png')
        plt.close()

        sns.utils.axlabel(xlabel=None, ylabel='fitness', fontsize=10)
        box_plot = sns.boxplot(data=[diff2_m_, diff2_f_],
                               width=.8,
                               palette="muted",
                               # medianprops=dict(color="darkred", alpha=0.8),
                               showfliers=False)

        plt.xticks(plt.xticks()[0], ['MOD', 'FLO'])

        plt.savefig(path_bp2 + '/comparison_' + str(counter) + '.png')
        plt.close()

        sns.utils.axlabel(xlabel=None, ylabel='fitness', fontsize=10)
        box_plot = sns.boxplot(data=[diff3_m_, diff3_f_, diff3_d_],
                               width=.8,
                               palette="muted",
                               # medianprops=dict(color="darkred", alpha=0.8),
                               showfliers=False)

        plt.xticks(plt.xticks()[0], ['MOD-CNN', 'FLO', 'MOD-DET'])

        plt.savefig(path_bp3 + '/comparison_' + str(counter) + '.png')
        plt.close()

    data_.append(unkn_data.item())
    model_.append(unkn_model.item())
    float_.append(unkn_float.item())
    emodnet_.append(emodnet_unkn)

    print('\nEMODNET ' + str(variable) + '  : ', emodnet_unkn)
    # print('\ndata ' + str(variable) + '     : ', unkn_data.item())
    print('model ' + str(variable) + '    : ', unkn_model.item())
    print('float ' + str(variable) + '    : ', unkn_float.item())
    print('\n------------------------')

    f.write("---------------------\n")
    f.write(f"[EMODNET]: {emodnet_unkn:.5e} \n")
    f.write(f"[DATA]   : {unkn_data.item():.5e} \n")
    f.write(f"[MODEL]  : {unkn_model.item():.5e} \n")
    f.write(f"[FLOAT]  : {unkn_float.item():.5e} \n")

f.close()

'''
n_measurement = range(len(emodnet_))
plt.plot(n_measurement, emodnet_, 'r', linewidth=1.25, marker='.', label='EMODNET')
# plt.plot(n_measurement, data_, 'b', linewidth=1, linestyle='dashed', marker='.', label='Model')
plt.plot(n_measurement, model_, 'g', linewidth=1, linestyle='dotted', marker='.', label='Inpainting Model')
plt.plot(n_measurement, float_, 'm', linewidth=1, linestyle='dashdot', marker='.', label='Inpainting Float')
plt.title('Comparison between ' + variable + ' values')
plt.legend()
plt.savefig(path_fig + '/comparison.png')
plt.show()
plt.close()
'''