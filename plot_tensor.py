"""
Plotting the information contained in the tensors
"""
import matplotlib.pyplot as plt
import os

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
    directory = path_directory + '/fig/' + str(kindof) + '/' + str(channel) + '/' + str(data_time)
    if not os.path.exists(directory):
        os.mkdir(directory)

    number_fig = len(tensor[0, 0, :, 0, 0])  # number of levels of depth

    for i in range(number_fig):
        cmap = plt.get_cmap('Greens')
        plt.imshow(tensor[0, channel, i, :, :], cmap=cmap)
        plt.savefig(directory + "/profondity_level_" + str(i) + ".png")
    plt.colorbar()


def plot_routine(kindof, list_parallelepiped, list_data_time, channels):
    """
    measurement plot different for each kind of data (float/sat/tensor)
    kindof = requires a str (float, sat or tensor)
    list_parallelepiped = list of tensor we want to plot (i.e. parallelepiped at a fixed date time)
    list_data_time = list of reference date time associated to the list of tensor
    channels = list of variable we want to plot
    """
    for j in range(len(list_data_time)):
        print('plotting tensor relative to time : ', list_data_time[j])
        for channel in channels:
            Plot_Tensor(kindof, list_parallelepiped[j], list_data_time[j], channel)

