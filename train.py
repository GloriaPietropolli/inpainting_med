"""
Implementation of the training routine for the 3D CNN with GAN

- train_dataset : list/array of 4D (or 5D ?) tensor in form (input_channels, D_in, H_in, W_in)
"""

import os
import torch
from torch.optim import Adam
from IPython import display
from discriminator import Discriminator
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import MV_pixel
from utils import generate_input_mask

path = 'result/'  # result directory
train_dataset = [torch.zeros(1, 1, 1)]
test_dataset = [torch.zeros(1, 1, 1)]

# compute the mean of the channel of the training set
mean_value_pixel = MV_pixel(train_dataset)

# transform the mean_value_pixel (an array of length 3) into a tensor of the same shape as the input's ones
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 3, 1, 1, 1))

# definitions of the hyperparameters
alpha = 4e-4
lr = 1e-3
alpha = torch.tensor(alpha)
epoch1 = 1000  # number of step for the first phase of training
epoch2 = 1000  # number of step for the second phase of training
epoch3 = 1000  # number of step for the third phase of training
hole_min_d, hole_max_d = 10, 100
hole_min_h, hole_max_h = 10, 100
hole_min_w, hole_max_w = 10, 100


# PHASE 1
# COMPLETION NETWORK is trained with the MSE loss for T_c iterations
model_completion = CompletionN()
optimizer_completion = Adam(model_completion.parameters(), lr=lr)
for ep in range(epoch1):
    for training_x in train_dataset:
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask # mask the training tensor with
        # pixel containing the mean value
        input = torch.cat((training_x_masked, mask), dim=1)
        output = model_completion(input)

        loss = completion_network_loss(training_x, output, mask)

        print(f"[PHASE1 : EPOCH]: {ep}, [LOSS]: {loss.item():.6f}")
        display.clear_output(wait=True)

        optimizer_completion.zero_grad()
        loss.backward()
        optimizer_completion.step()


