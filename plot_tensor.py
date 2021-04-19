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
    fig = plt.figure(figsize=(75, 75))
    columns = 5
    rows = number_fig / columns

    for i in range(number_fig):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(tensor[0, channel, i, :, :])
        plt.colorbar()
    plt.show()
    plt.savefig(path + str(data_time) + "channel" + str(channel) + "information.png")
