import os
import torch
import matplotlib.pyplot as plt

analysis = 'model'

if analysis == 'float':
    epoch_model = 500
    lr_model = 0.01

    epoch_float = 50
    lr_float = 0.0001

    path = os.getcwd()
    model_considered = 'model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(lr_model) + '.'

    path_float = path + '/result2/' + model_considered + '/' + str(epoch_float) + '/' + str(lr_float) + '/tensor/'
    path_model = path + '/result/model2015_c/' + str(epoch_model) + '_epoch/' + str(lr_model) + '_lr/tensor/'

    diff = []
    name_float, name_model = os.listdir(path_float), os.listdir(path_model)
    name_float.sort()
    name_model.sort()

    for i in range(len(name_float)):
        float_n = name_float[i]
        model_n = 'epoch_' + str(epoch_model - 1) + '/tensor_phase1.pt'
        if not os.path.exists(path_model + model_n):
            continue
        float_t = torch.load(path_float + float_n)
        model_t = torch.load(path_model + model_n)
        print('calculating distance between float ' + str(float_n[:-3]) + ' and model epoch_' + str(epoch_model - 1) + '...')
        diff = float_t - model_t

        path_fig = path + '/result2/' + model_considered + '/' + str(epoch_float) + '/' + str(lr_float) + '/diff/epoch_' + str(i)
        if not os.path.exists(path_fig):
            os.mkdir(path_fig)
        number_fig = len(diff[0, 0, :, 0, 0])  # number of levels of depth

        for channel in [0, 1, 2, 3]:
            for i in range(number_fig):
                path_fig_channel = path_fig + '/' + str(channel)
                if not os.path.exists(path_fig_channel):
                    os.mkdir(path_fig_channel)
                cmap = plt.get_cmap('Greens')
                plt.imshow(diff[0, channel, i, :, :], cmap=cmap)
                plt.colorbar()
                plt.savefig(path_fig_channel + "/profondity_level_" + str(i) + ".png")
                plt.close()


if analysis == 'model':
    epoch_model = 501
    lr_model = 0.01

    path = os.getcwd()
    model_considered = 'model_completion_epoch_' + str(epoch_model) + '_lrc_' + str(lr_model) + '.'

    path_model = path + '/result/model2015_c/' + str(epoch_model) + '_epoch/' + str(lr_model) + '_lr/tensor/'
    models = os.listdir(path_model)

    for i in range(len(models)):
        epoch_directory = models[i]
        epoch = epoch_directory[-2:]

        model = torch.load(path_model + epoch_directory + '/tensor_phase1.pt')
        if not os.path.exists(model):
            continue
        data = torch.load(path_model + '/testing_x/original_tensor.pt')

        print('calculating distance among epoch ' + str(epoch) + ' ...')
        diff = data - model

        path_fig = path + '/result/model2015_c/' + str(epoch_model) + '_epoch/' + str(lr_model) + '_lr/diff/'
        if not os.path.exists(path_fig):
            os.mkdir(path_fig)
        path_fig_epoch = path_fig + '/epoch' + epoch
        if not os.path.exists(path_fig_epoch):
            os.mkdir(path_fig_epoch)

        number_fig = len(diff[0, 0, :, 0, 0])  # number of levels of depth

        for channel in [0, 1, 2, 3]:
            for i in range(number_fig):
                path_fig_channel = path_fig_epoch + '/' + str(channel)
                if not os.path.exists(path_fig_channel):
                    os.mkdir(path_fig_channel)
                cmap = plt.get_cmap('Greens')
                plt.imshow(diff[0, channel, i, :, :], cmap=cmap)
                plt.colorbar()
                plt.savefig(path_fig_channel + "/profondity_level_" + str(i) + ".png")
                plt.close()

