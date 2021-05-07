"""
definition of the local discriminator (LocDiscriminator) and of the global discriminator (GloDiscriminator)
for the implementation of the GAN
"""

import torch.nn as nn
from layers import Flatten, Concatenate

in_channels = 4  # number of channels of the input 3d spaces


class LocDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LocDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)  # represents the local context around the completed region
        self.channel = input_shape[0]
        self.depth = input_shape[1]
        self.height = input_shape[2]
        self.width = input_shape[3]

        self.conv1 = nn.Conv3d(self.channel, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm3d(64)
        self.af1 = nn.ReLU()

        self.conv2 = nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm3d(128)
        self.af2 = nn.ReLU()

        self.conv3 = nn.Conv3d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm3d(256)
        self.af3 = nn.ReLU()

        self.conv4 = nn.Conv3d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm3d(512)
        self.af4 = nn.ReLU()

        self.conv5 = nn.Conv3d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm3d(512)
        self.af5 = nn.ReLU()

        input_linear_layer = 16 * (self.height // 6) * (self.width // 6) * (self.depth // 8)
        # sto cambiando per cosa divido self-h, self.w e self.d controlla articolo se i nmeri posso sceglirli come vgl
        self.linear = nn.Linear(input_linear_layer, 1024)
        self.af_final = nn.ReLU()
        self.flatten = Flatten()

    def forward(self, x):
        x = self.bn1(self.af1(self.conv1(x)))
        x = self.bn2(self.af2(self.conv2(x)))
        x = self.bn3(self.af3(self.conv3(x)))
        x = self.bn4(self.af4(self.conv4(x)))
        x = self.bn5(self.af5(self.conv5(x)))
        x = self.af_final(self.linear(self.flatten(x)))
        return x


class GloDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(GloDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.channel = input_shape[0]
        self.depth = input_shape[1]
        self.height = input_shape[2]
        self.width = input_shape[3]

        self.conv1 = nn.Conv3d(self.channel, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm3d(64)
        self.af1 = nn.ReLU()

        self.conv2 = nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm3d(128)
        self.af2 = nn.ReLU()

        self.conv3 = nn.Conv3d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm3d(256)
        self.af3 = nn.ReLU()

        self.conv4 = nn.Conv3d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm3d(512)
        self.af4 = nn.ReLU()

        self.conv5 = nn.Conv3d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm3d(512)
        self.af5 = nn.ReLU()

        self.conv6 = nn.Conv3d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm3d(512)
        self.af6 = nn.ReLU()

        input_linear_layer = 8 * (self.height // 8) * (self.width // 9) * (self.depth // 7)
        self.flatten = Flatten()
        self.linear = nn.Linear(input_linear_layer, 1024)
        self.af_final = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.af1(self.conv1(x)))
        x = self.bn2(self.af2(self.conv2(x)))
        x = self.bn3(self.af3(self.conv3(x)))
        x = self.bn4(self.af4(self.conv4(x)))
        x = self.bn5(self.af5(self.conv5(x)))
        x = self.bn6(self.af6(self.conv6(x)))
        x = self.af_final(self.linear(self.flatten(x)))
        return x


class Discriminator(nn.Module):
    def __init__(self, loc_input_shape, glo_input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = [loc_input_shape, glo_input_shape]
        self.output_shape = (1,)
        self.model_local_discriminator = LocDiscriminator(loc_input_shape)
        self.model_global_discriminator = GloDiscriminator(glo_input_shape)

        input_linear_layer = self.model_local_discriminator.output_shape[-1] + \
                             self.model_global_discriminator.output_shape[-1]
        self.concatenate = Concatenate(dim=-1)
        self.linear = nn.Linear(input_linear_layer, 1)
        self.af = nn.Sigmoid()

    def forward(self, x):
        x_local_discriminator, x_global_discriminator = x
        x_local_discriminator = self.model_local_discriminator(x_local_discriminator)
        x_global_discriminator = self.model_global_discriminator(x_global_discriminator)
        discriminator_output = self.af(self.linear(self.concatenate([x_local_discriminator, x_global_discriminator])))
        return discriminator_output
