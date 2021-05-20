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
    descr = flag_dict[flag]

    label = 'losses'

    figure(figsize=(10, 6))

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


def Plot_Adversarial_Error(losses_c, losses_d, path):
    """
    print the error of the completion network and of the discriminator network both in a normal scale and in a log scale
    """
    label_c = 'TEST completion network losses'
    label_d = 'discriminator network losses'
    plt.plot(losses_c, '-r', label=label_c)
    plt.plot(losses_d, '-g', label=label_d)
    plt.xlabel('number iteration phase3')
    plt.title('COMPLETION AND DISCRIMINATOR LOSS PHASE 3')
    plt.legend()
    plt.savefig(path + "_LOSS1C.png")
    plt.close()

    plt.plot(np.log(losses_c), '-r', label=label_c)
    plt.plot(np.log(losses_d), '-g', label=label_d)
    plt.xlabel('number iteration phase3')
    plt.title('COMPLETION AND DISCRIMINATOR LOG LOSS PHASE 3')
    plt.legend()
    plt.savefig(path + "_LOG_LOSS1C.png")
