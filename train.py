"""
Implementation of the training routine for the 3D CNN with GAN

- train_dataset : list/array of 4D (or 5D ?) tensor in form (bs, input_channels, D_in, H_in, W_in)
"""
import random
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from IPython import display
from discriminator import Discriminator
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import MV_pixel
from utils import generate_input_mask, generate_hole_area, crop, sample_random_batch
from get_dataset import *

num_channel = 4  # 0,1,2,3

path = 'result/' + kindof  # result directory

if kindof == 'float':
    train_dataset = list_float_tensor
if kindof == 'model2015':
    train_dataset = list_model_tensor
if kindof == 'sat':
    train_dataset = list_sat_tensor

mean_value_pixel = MV_pixel(train_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, num_channel, 1, 1, 1))  # transform the mean_value_pixel
# (an array of length 3) into a tensor of the same shape as the input's ones

# definitions of the hyperparameters
alpha = 4e-4
lr_c = 1e-3
lr_d = 1e-3
alpha = torch.tensor(alpha)
num_test_completions = 10
epoch1 = 50  # number of step for the first phase of training
snaperiod_1 = 10
epoch2 = 50  # number of step for the second phase of training
epoch3 = 50  # number of step for the third phase of training
hole_min_d, hole_max_d = 1, 10
hole_min_h, hole_max_h = 1, 10
hole_min_w, hole_max_w = 1, 10
cn_input_size = (30, 65, 75)
ld_input_size = (15, 40, 40)

# PHASE 1
# COMPLETION NETWORK is trained with the MSE loss for T_c (=epoch1) iterations

model_completion = CompletionN()
optimizer_completion = Adam(model_completion.parameters(), lr=lr_c)
for ep in range(epoch1):
    for training_x in train_dataset:
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask  # mask the training tensor with
        # pixel containing the mean value
        input = torch.cat((training_x_masked, mask), dim=1)
        output = model_completion(input.float())

        output = output[:, :, :, :-1, :]
        output = torch.cat((output, torch.zeros(1, 4, 30, 65, 1)), -1)

        loss_completion = completion_network_loss(training_x, output, mask)  # MSE

        print(f"[PHASE1 : EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f}")
        display.clear_output(wait=True)

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()

        # test
        if ep % snaperiod_1 == 0:
            model_completion.eval()
            with torch.no_grad():
                testing_x = random.choice(train_dataset)
                training_mask = generate_input_mask(
                    shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
                    hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
                    hole_area=generate_hole_area(ld_input_size, (training_x.shape[2], training_x.shape[3], training_x.shape[4])))
                testing_x_mask = testing_x - testing_x * mask + mean_value_pixel * mask
                testing_input = torch.cat((testing_x_mask, training_mask), dim=1)
                testing_output = model_completion(testing_input.float())

                path_tensor_phase1 = path + '/phase1/tensor/'
                path_fig_phase1 = path + '/phase1/fig/'

                path_tensor_epoch = path_tensor_phase1 + 'epoch_' + str(ep)
                if not os.path.exists(path_tensor_epoch):
                    os.mkdir(path_tensor_epoch)
                torch.save(testing_output, path_tensor_epoch + "/tensor_phase1" + ".pt")

                path_fig_epoch = path_fig_phase1 + 'epoch_' + str(ep)
                if not os.path.exists(path_fig_epoch):
                    os.mkdir(path_fig_epoch)

                number_fig = len(testing_output[0, 0, :, 0, 0])  # number of levels of depth

                for channel in channels:
                    for i in range(number_fig):
                        path_fig_channel = path_fig_epoch + '/' + str(channel)
                        if not os.path.exists(path_fig_channel):
                            os.mkdir(path_fig_channel)
                        cmap = plt.get_cmap('Greens')
                        plt.imshow(testing_output[0, channel, i, :, :], cmap=cmap)
                        plt.colorbar()
                        plt.savefig(path_fig_channel + "/profondity_level_" + str(i) + ".png")
                        plt.close()

# PHASE 2
# COMPLETION NETWORK is FIXED and DISCRIMINATORS are trained form scratch for T_d (=epoch2) iterations

model_discriminator = Discriminator(loc_input_shape=(num_channel,) + ld_input_size,
                                    glo_input_shape=(num_channel,) + cn_input_size)
optimizer_discriminator = Adam(model_discriminator.parameters(), lr=lr_d)
loss_discriminator = nn.BCELoss()
for ep in range(epoch2):
    for training_x in train_dataset:
        # fake forward
        hole_area_fake = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
            hole_area=hole_area_fake)
        fake = torch.zeros((len(training_x), 1))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask  # mask the training tensor with
        input_completion = torch.cat((training_x_masked, mask), dim=1)
        output_completion = model_completion(input_completion.float())
        input_global_discriminator_fake = output_completion.detach()
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_fake = loss_discriminator(output_fake, fake)

        # real forward
        hole_area_real = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        real = torch.ones((len(training_x), 1))
        input_global_discriminator_real = training_x
        input_local_discriminator_real = crop(training_x, hole_area_real)
        output_real = model_discriminator((input_local_discriminator_real, input_global_discriminator_real))
        loss_real = loss_discriminator(output_real, real)

        loss = (loss_real + loss_fake) / 2.0

        print(f"[PHASE2 : EPOCH]: {ep + 1}, [LOSS]: {loss.item():.12f}")
        display.clear_output(wait=True)

        loss.backward()
        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()

# PHASE 3
# both the completion network and content discriminators are trained jointly until the end of training

for ep in range(epoch3):
    for training_x in train_dataset:
        # fake forward
        hole_area_fake = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
            hole_area=hole_area_fake)
        fake = torch.zeros((len(training_x), 1))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask
        input_completion = torch.cat((training_x_masked, mask), dim=1)
        output_completion = model_completion(input_completion.float())

        output_completion = output_completion[:, :, :, :-1, :]
        output_completion = torch.cat((output_completion, torch.zeros(1, 4, 30, 65, 1)), -1)

        input_global_discriminator_fake = output_completion.detach()
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_fake = loss_discriminator(output_fake, fake)

        # real forward
        hole_area_real = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        real = torch.ones((len(training_x), 1))
        input_global_discriminator_real = training_x
        input_local_discriminator_real = crop(training_x, hole_area_real)
        output_real = model_discriminator((input_local_discriminator_real, input_global_discriminator_real))
        loss_real = loss_discriminator(output_real, real)

        loss_d = (loss_real + loss_fake) * alpha / 2.0

        # backward discriminator
        loss_d.backward()
        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()

        # forward completion
        loss_c1 = completion_network_loss(training_x, output_completion, mask)
        input_global_discriminator_fake = output_completion
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_c2 = loss_discriminator(output_fake, real)

        loss_c = (loss_c1 + alpha * loss_c2) / 2.0

        # backward completion
        loss_c.backward()
        optimizer_completion.step()
        optimizer_completion.zero_grad()

        print(
            f"[PHASE3 : EPOCH]: {ep}, [LOSS COMPLETION]: {loss_c.item():.12f}, [LOSS DISCRIMINATOR]: {loss_d.item():.12f}")
        display.clear_output(wait=True)
