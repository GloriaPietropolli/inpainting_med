"""
Implementation of the training routine for the 3D CNN with GAN
- train_dataset : list/array of 5D (or 5D ?) tensor in form (bs, input_channels, D_in, H_in, W_in)
"""
import torch.nn as nn
from torch.optim import Adadelta
import matplotlib.pyplot as plt
from IPython import display
from discriminator import Discriminator
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import MV_pixel
from utils import generate_input_mask, generate_hole_area, crop
from normalization import Normalization
from plot_error import Plot_Error
from get_dataset import *

num_channel = number_channel  # 0,1,2,3

path = 'result/model2015_c'  # result directory

train_dataset = get_list_model_tensor()

index_testing = -1
train_dataset, _, _ = Normalization(train_dataset)
testing_x = train_dataset[index_testing]  # test on the last element of the list
train_dataset.pop(index_testing)

mean_value_pixel = MV_pixel(train_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, num_channel, 1, 1, 1))

# definitions of the hyperparameters
alpha = 4e-4
lr_c = 1e-3
alpha = torch.tensor(alpha)
num_test_completions = 0
epoch1 = 1500  # number of step for the first phase of training
snaperiod = 25
hole_min_d1, hole_max_d1 = 28, 29  # different hole size for the first training (no local discriminator here)
hole_min_h1, hole_max_h1 = 1, 50
hole_min_w1, hole_max_w1 = 1, 50
hole_min_d2, hole_max_d2 = 10, 20
hole_min_h2, hole_max_h2 = 30, 50
hole_min_w2, hole_max_w2 = 30, 50
snaperiod_hole = 2

# make directory
path_configuration = path + '/' + str(epoch1) + '_epoch'
if not os.path.exists(path_configuration):
    os.mkdir(path_configuration)
path_lr = path_configuration + '/' + str(lr_c) + '_lr'
if not os.path.exists(path_lr):
    os.mkdir(path_lr)

losses_1_c = []  # losses of the completion network during phase 1
losses_1_c_test = []  # losses of TEST of the completion network during phase 1


# PHASE 1
# COMPLETION NETWORK is trained with the MSE loss for T_c (=epoch1) iterations

model_completion = CompletionN()
optimizer_completion = Adadelta(model_completion.parameters(), lr=lr_c)
f = open(path_lr + "/phase1_losses.txt", "w+")
f_test = open(path_lr + "/phase1_TEST_losses.txt", "w+")
for ep in range(epoch1):
    for training_x in train_dataset:
        if ep % snaperiod_hole == 0:
            hole_min_d, hole_max_d = hole_min_d1, hole_max_d1
            hole_min_h, hole_max_h = hole_min_h1, hole_max_h1
            hole_min_w, hole_max_w = hole_min_w1, hole_max_w1
        else:
            hole_min_d, hole_max_d = hole_min_d2, hole_max_d2
            hole_min_h, hole_max_h = hole_min_h2, hole_max_h2
            hole_min_w, hole_max_w = hole_min_w2, hole_max_w2

        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask

        input = torch.cat((training_x_masked, mask), dim=1)
        output = model_completion(input.float())

        loss_completion = completion_network_loss(training_x, output, mask)  # MSE
        losses_1_c.append(loss_completion.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.6e}")
        display.clear_output(wait=True)
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.6e} \n")

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()

    # test
    if ep % snaperiod == 0:
        model_completion.eval()
        with torch.no_grad():
            # testing_x = random.choice(test_dataset)
            training_mask = generate_input_mask(
                shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
                hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
            # hole_area=generate_hole_area(ld_input_size,
            #                              (training_x.shape[2], training_x.shape[3], training_x.shape[4])))
            testing_x_mask = testing_x - testing_x * training_mask + mean_value_pixel * training_mask
            testing_input = torch.cat((testing_x_mask, training_mask), dim=1)
            testing_output = model_completion(testing_input.float())

            loss_1c_test = completion_network_loss(testing_x, testing_output, training_mask)
            losses_1_c_test.append(loss_1c_test)

            print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test.item():.6e}")
            display.clear_output(wait=True)
            f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test.item():.6e} \n")

            path_tensor = path_lr + '/tensor/'
            if not os.path.exists(path_tensor):
                os.mkdir(path_tensor)
            path_fig = path_lr + '/fig/'
            if not os.path.exists(path_fig):
                os.mkdir(path_fig)

            path_tensor_epoch = path_tensor + 'epoch_' + str(ep)
            if not os.path.exists(path_tensor_epoch):
                os.mkdir(path_tensor_epoch)
            torch.save(testing_output, path_tensor_epoch + "/tensor_phase1" + ".pt")

            path_fig_epoch = path_fig + 'epoch_' + str(ep)
            if not os.path.exists(path_fig_epoch):
                os.mkdir(path_fig_epoch)

            path_fig_original = path_fig + 'original_fig'
            if not os.path.exists(path_fig_original):
                os.mkdir(path_fig_original)

            number_fig = len(testing_output[0, 0, :, 0, 0])  # number of levels of depth

            for channel in [0, 1, 2, 3]:
                for i in range(number_fig):
                    path_fig_channel = path_fig_epoch + '/' + str(channel)
                    if not os.path.exists(path_fig_channel):
                        os.mkdir(path_fig_channel)
                    cmap = plt.get_cmap('Greens')
                    plt.imshow(testing_output[0, channel, i, :, :], cmap=cmap)
                    plt.colorbar()
                    plt.savefig(path_fig_channel + "/profondity_level_" + str(i) + ".png")
                    plt.close()

                    if ep == 0:
                        path_fig_channel = path_fig_original + '/' + str(channel)
                        if not os.path.exists(path_fig_channel):
                            os.mkdir(path_fig_channel)
                        plt.imshow(testing_x[0, channel, i, :, :], cmap=cmap)
                        plt.colorbar()
                        plt.savefig(path_fig_channel + "/profondity_level_original_" + str(i) + ".png")
                        plt.close()

f.close()
f_test.close()

Plot_Error(losses_1_c_test, '1c', path_lr + '/')  # plot of the error in phase1

path_model = 'model/model2015_c/model_completion_' + 'epoch_' + str(epoch1) + '_lrc_' + str(lr_c) + '.pt '
torch.save(model_completion.state_dict(), path_model)

f.close()

# printing specifics of the problem
print('epoch phase 1 : ', epoch1)
print('learning rate completion : ', lr_c)

# printing final loss training set
print('final loss of TRAINING completion network: ', losses_1_c[-1])
# printing final loss of testing set
print('final loss TEST : ', losses_1_c_test[-1].item())
