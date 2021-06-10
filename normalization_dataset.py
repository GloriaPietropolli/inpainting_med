import torch
import os
import matplotlib.pyplot as plt

from normalization import Normalization

path = os.getcwd() + '/tensor/model2015/'
tensor_ = os.listdir(path)

dataset = []
for ptFiles in tensor_:
    tensor = torch.load(path + ptFiles)
    dataset.append(tensor)

normalized_list, mean_tensor, std_tensor = Normalization(dataset)
path_norm = os.getcwd() + '/tensor/model2015_n/'
if not os.path.exists(path_norm):
    os.mkdir(path_norm)

for el in range(len(tensor_)):
    tensor_name = tensor_[el]
    tensor = torch.load(path + tensor_name)
    tensor = (tensor - mean_tensor) / std_tensor
    tensor = tensor[:, :, :-1, :, 1:-1]
    # tensor = tensor.float()
    torch.save(tensor, path_norm + tensor_name)

    # print normalized tensor
    path_fig = os.getcwd() + '/fig/model2015_n/'

    number_fig = len(tensor[0, 0, :, 0, 0])  # number of levels of depth

    for channel in [0, 1, 2, 3]:
        for i in range(number_fig):
            path_fig_channel = path_fig + '/' + str(channel)
            if not os.path.exists(path_fig_channel):
                os.mkdir(path_fig_channel)
            cmap = plt.get_cmap('Greens')
            plt.imshow(tensor[0, channel, i, :, :], cmap=cmap)
            plt.colorbar()
            plt.savefig(path_fig_channel + "/profondity_level_" + str(i) + ".png")
            plt.close()


