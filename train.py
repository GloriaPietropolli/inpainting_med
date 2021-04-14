"""
Implementation of the training routine for the 3D CNN with GAN

- train_dataset : list/array of 4D (or 5D ?) tensor in form (input_channels, D_in, H_in, W_in)
"""

import os
from torch.optim import Adam
from discriminator import Discriminator
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import MV_pixel
from utils import *

path = 'result/'  # result directory
train_dataset = [torch.zeros(1, 1, 1)]
test_dataset = [torch.zeros(1, 1, 1)]

# compute the mean of the channel of the training set
mean_value_pixel = MV_pixel(train_dataset)

# transform the mean_value_pixel (an array of length 3) into a tensor of the same shape as the input's ones
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 3, 1, 1, 1))
