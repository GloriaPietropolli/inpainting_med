"""
Plotting the information contained in the tensors
"""
import matplotlib.pyplot as plt

path = "fig/"


def Plot_Tensor(tensor, data_time, channel):
    """
    Plotting the tensor's values at different levels of depth
    i.e. plotting along the component d (=depth) of tensor (bs, c,c d, h, w)
    tensor = tensor we want to plot (i.e. parallelepiped at a fixed date time)
    channel = variable we want to plot
    """
    number_fig = len(tensor[0, 0, :, 0, 0])  # number of levels of depth

    for i in range(number_fig):
        plt.imshow(tensor[0, channel, i, :, :])
        plt.colorbar()
        plt.savefig(path + str(channel) + "/" + str(data_time) + "_profondity_level_" + str(i) + ".png")
    # plt.show()

