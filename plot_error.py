"""
Plotting the error during the different phase of the training
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


def Plot_Error(losses, flag, path):
    """
    plot of the losses
    flag : what phase of the training we are plotting
    path : where to save the plot
    """
    flag_dict = {'1c': 'model completion at phase 1',
                 '2d': 'model discriminator at phase 2',
                 '3c': 'model completion phase at 3',
                 '3d': 'model completion phase at 3',
                 'float': 'float completion',
                 'sat': 'sat completion'}

    if flag[-1] == 'd':
        losses = [losses[i] for i in range(0, len(losses), 50)]

    descr = flag_dict[flag]

    figure(figsize=(10, 6))

    label = 'losses'
    plt.plot(losses, 'orange')
    plt.plot(losses, 'm.', label=label)
    plt.xlabel('Number of epochs')
    plt.title('Losses of ' + descr)
    plt.legend()
    plt.savefig(path + "loss_" + str(flag) + ".png")
    plt.close()

    label = 'log losses '
    plt.plot(np.log(losses), 'orange')
    plt.plot(np.log(losses), 'm.', label=label)
    plt.xlabel('Number of epochs')
    plt.title('Logarithmic Losses of ' + descr)
    plt.legend()
    plt.savefig(path + "LOGloss_" + str(flag) + ".png")
    plt.close()
