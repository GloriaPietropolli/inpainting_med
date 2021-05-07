"""
Plotting the error during the different phase of the training
"""
import matplotlib.pyplot as plt


def Plot_Error(losses, flag, path):
    if flag == '1c':
        label = 'losses' + flag
        plt.plot(losses, '-r', label=label)
        plt.xlabel('number iteration phase1')
        plt.title('COMPLETION LOSS PHASE 1')
        plt.savefig(path + "_LOSS1C.png")
    if flag == '2d':
        label = 'losses' + flag
        plt.plot(losses, '-r', label=label)
        plt.xlabel('number iteration phase2')
        plt.title('DISCRIMINATOR LOSS PHASE 2')
        plt.savefig(path + "_LOSS2D.png")
    if flag == '3c':
        label = 'losses' + flag
        plt.plot(losses, '-r', label=label)
        plt.xlabel('number iteration phase3')
        plt.title('COMPLETION LOSS PHASE 3')
        plt.savefig(path + "_LOSS3C.png")
    if flag == '3d':
        label = 'losses' + flag
        plt.plot(losses, '-r', label=label)
        plt.xlabel('number iteration phase3')
        plt.title('DISCRIMINATOR LOSS PHASE 1')
        plt.savefig(path + "_LOSS3D.png")
