"""
Plotting the information contained in the tensors
"""
import matplotlib.pyplot as plt
import os
import torch

path = "fig/"
path_directory = os.getcwd()


def Plot_Tensor(kindof, tensor, data_time, channel):
    """
    Plotting the tensor's values at different levels of depth
    i.e. plotting along the component d (=depth) of tensor (bs, c,c d, h, w)
    tensor = tensor we want to plot (i.e. parallelepiped at a fixed date time)
    data_time = reference date time associated to the list of tensor
    channel = variable we want to plot
    """
    dict_channel = {0: 'temperature', 1: 'salinity', 2: 'oxygen', 3: 'chla'}
    directory = path_directory + '/fig/' + str(kindof) + '/' + str(channel) + '/' + str(data_time)
    if not os.path.exists(directory):
        os.mkdir(directory)

    number_fig = len(tensor[0, 0, :, 0, 0])  # number of levels of depth

    for i in range(number_fig):
        cmap = plt.get_cmap('Greens')
        plt.imshow(tensor[0, channel, i, :, :], cmap=cmap)
        plt.title('Section of ' + dict_channel[channel])
        plt.colorbar()
        plt.savefig(directory + "/profondity_level_" + str(i) + ".png")
        plt.close()


def plot_routine(kindof, list_parallelepiped, list_data_time, channels, year_interval):
    """
    measurement plot different for each kind of data (float/sat/tensor)
    kindof = requires a str (float, sat or tensor)
    list_parallelepiped = list of tensor we want to plot (i.e. parallelepiped at a fixed date time)
    list_data_time = list of reference date time associated to the list of tensor
    channels = list of variable we want to plot
    """
    year_min, year_max = year_interval
    for j in range(len(list_data_time)):
        time_considered = list_data_time[j]
        tensor_considered = list_parallelepiped[j]
        if year_min < time_considered < year_max:
            print('plotting tensor relative to time : ', time_considered)
            for channel in channels:
                Plot_Tensor(kindof, tensor_considered, time_considered, channel)


def Save_Tensor(kindof, tensor, data_time):
    """
    Saving the tensor's values at different levels of depth
    i.e. plotting along the component d (=depth) of tensor (bs, c,c d, h, w)
    tensor = tensor we want to plot (i.e. parallelepiped at a fixed date time)
    data_time = reference date time associated to the list of tensor
    channel = variable we want to plot
    """
    directory = path_directory + '/tensor/' + str(kindof)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(tensor, directory + "/datetime_" + str(data_time) + ".pt")


def save_routine(kindof, list_parallelepiped, list_data_time, year_interval):
    """
    measurement plot different for each kind of data (float/sat/tensor)
    kindof = requires a str (float, sat or tensor)
    list_parallelepiped = list of tensor we want to plot (i.e. parallelepiped at a fixed date time)
    list_data_time = list of reference date time associated to the list of tensor
    channels = list of variable we want to plot
    """
    year_min, year_max = year_interval
    for j in range(len(list_data_time)):
        time_considered = list_data_time[j]
        tensor_considered = list_parallelepiped[j]
        if year_min < time_considered < year_max:
            print('saving tensor relative to time : ', time_considered)
            Save_Tensor(kindof, tensor_considered, time_considered)


