"""
Plotting the error during the different phase of the training
"""
import matplotlib.pyplot as plt
import numpy as np


def Plot_Error(losses, flag, path):
    if flag == '1c':
        label = 'losses' + flag
        plt.plot(losses, '-r', label=label)
        plt.xlabel('number iteration phase1')
        plt.title('COMPLETION LOSS PHASE 1')
        plt.legend()
        plt.savefig(path + "_LOSS1C.png")
        plt.close()
    if flag == '2d':
        label = 'losses' + flag
        plt.plot(losses, '-r', label=label)
        plt.xlabel('number iteration phase2')
        plt.title('DISCRIMINATOR LOSS PHASE 2')
        plt.legend()
        plt.savefig(path + "_LOSS2D.png")
        plt.close()


def Plot_Adversarial_Error(losses_c, losses_d, path):
    """
    print the error of the completion network and of the discriminator network both in a normal scale and in a log scale
    """
    label_c = 'completion network losses'
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
