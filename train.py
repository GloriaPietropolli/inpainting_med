"""
Implementation of the training routine for the 3D CNN with GAN

- train_dataset : list/array of 4D (or 5D ?) tensor in form (bs, input_channels, D_in, H_in, W_in)
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from IPython import display
from discriminator import Discriminator
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import MV_pixel
from utils import generate_input_mask, generate_hole_area, crop

num_channel = 4

path = 'result/'  # result directory
train_dataset = [torch.zeros(1, num_channel, 1, 1)]
test_dataset = [torch.zeros(1, num_channel, 1)]

# compute the mean of the channel of the training set
mean_value_pixel = MV_pixel(train_dataset)

# transform the mean_value_pixel (an array of length 3) into a tensor of the same shape as the input's ones
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, num_channel, 1, 1, 1))

# definitions of the hyperparameters
alpha = 4e-4
lr_c = 1e-3
lr_d = 1e-3
alpha = torch.tensor(alpha)
epoch1 = 1000  # number of step for the first phase of training
epoch2 = 1000  # number of step for the second phase of training
epoch3 = 1000  # number of step for the third phase of training
hole_min_d, hole_max_d = 10, 100
hole_min_h, hole_max_h = 10, 100
hole_min_w, hole_max_w = 10, 100
cn_input_size = 160
ld_input_size = 96

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
        output = model_completion(input)

        loss_completion = completion_network_loss(training_x, output, mask)  # MSE

        print(f"[PHASE1 : EPOCH]: {ep}, [LOSS]: {loss_completion.item():.6f}")
        display.clear_output(wait=True)

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()

# PHASE 2
# COMPLETION NETWORK is FIXED and DISCRIMINATORS are trained form scratch for T_d (=epoch2) iterations

model_discriminator = Discriminator(loc_input_shape=(num_channel, ld_input_size, ld_input_size, ld_input_size),
                                    glo_input_shape=(num_channel, cn_input_size, cn_input_size, cn_input_size))
optimizer_discriminator = Adam(model_discriminator.parameters(), lr=lr_d)
loss_discriminator = nn.BCELoss()
for ep in range(epoch2):
    for training_x in train_dataset:
        # fake forward
        hole_area_fake = generate_hole_area((ld_input_size, ld_input_size, ld_input_size),
                                            (training_x[4], training_x[3], training_x[2]))
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
            hole_area=hole_area_fake)
        fake = torch.zeros((len(training_x), 1))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask  # mask the training tensor with
        input_completion = torch.cat((training_x_masked, mask), dim=1)
        output_completion = model_completion(input_completion)
        input_global_discriminator_fake = output_completion.detach()
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_fake = loss_discriminator(output_fake, fake)

        # real forward
        hole_area_real = generate_hole_area((ld_input_size, ld_input_size, ld_input_size),
                                            (training_x[4], training_x[3], training_x[2]))
        real = torch.ones((len(training_x), 1))
        input_global_discriminator_real = training_x
        input_local_discriminator_real = crop(training_x, hole_area_real)
        output_real = model_discriminator((input_local_discriminator_real, input_global_discriminator_real))
        loss_real = loss_discriminator(output_real, real)

        loss = (loss_real + loss_fake) / 2.0

        print(f"[PHASE2 : EPOCH]: {ep}, [LOSS]: {loss.item():.6f}")
        display.clear_output(wait=True)

        loss.backward()
        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()

# PHASE 3
# both the completion network and content discriminators are trained jointly until the end of training

for ep in range(epoch3):
    for training_x in train_dataset:
        # fake forward
        hole_area_fake = generate_hole_area((ld_input_size, ld_input_size, ld_input_size),
                                            (training_x[4], training_x[3], training_x[2]))
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
            hole_area=hole_area_fake)
        fake = torch.zeros((len(training_x), 1))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask
        input_completion = torch.cat((training_x_masked, mask), dim=1)
        output_completion = model_completion(input_completion)
        input_global_discriminator_fake = output_completion.detach()
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_fake = loss_discriminator(output_fake, fake)

        # real forward
        hole_area_real = generate_hole_area((ld_input_size, ld_input_size, ld_input_size),
                                            (training_x[4], training_x[3], training_x[2]))
        real = torch.ones((len(training_x), 1))
        input_global_discriminator_real = training_x
        input_local_discriminator_real = crop(training_x, hole_area_real)
        output_real = model_discriminator((input_local_discriminator_real, input_global_discriminator_real))
        loss_real = loss_discriminator(output_real, real)

        loss_d = (loss_real + loss_fake) * alpha / 2.0

        # backward discriminator
        loss.backward()
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
            f"[PHASE3 : EPOCH]: {ep}, [LOSS COMPLETION]: {loss_c.item():.6f}, [LOSS DISCRIMINATOR]: {loss_d.item():.6f}")
        display.clear_output(wait=True)
