"""
definition of the COMPLETION NETWORK (fully CNN) that compose the architecture
"""
import torch.nn as nn


in_channels = 4  # number of channels of the input 3d spaces
out_channels = in_channels - 1  # same channels of the input except for the mask channel


class CompletionN(nn.Module):
    def __init__(self):
        super(CompletionN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(64)
        self.af1 = nn.ReLU()

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.af2 = nn.ReLU()

        self.conv3 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.af3 = nn.ReLU()

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.af4 = nn.ReLU()

        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.af5 = nn.ReLU()

        self.conv6 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm3d(256)
        self.af6 = nn.ReLU()

        self.conv7 = nn.Conv3d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm3d(256)
        self.af7 = nn.ReLU()

        self.conv8 = nn.Conv3d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm3d(256)
        self.af8 = nn.ReLU()

        self.conv9 = nn.Conv3d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm3d(256)
        self.af9 = nn.ReLU()

        self.conv10 = nn.Conv3d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm3d(256)
        self.af10 = nn.ReLU()

        self.conv11 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm3d(256)
        self.af11 = nn.ReLU()

        self.conv12 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm3d(256)
        self.af12 = nn.ReLU()

        self.deconv13 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm3d(128)
        self.af13 = nn.ReLU()

        self.conv14 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm3d(128)
        self.af14 = nn.ReLU()

        self.deconv15 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm3d(64)
        self.af15 = nn.ReLU()

        self.conv16 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm3d(32)
        self.af16 = nn.ReLU()

        self.conv17 = nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.af17 = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(self.af1(self.conv1(x)))
        x = self.bn2(self.af2(self.conv2(x)))
        x = self.bn3(self.af3(self.conv3(x)))
        x = self.bn4(self.af4(self.conv4(x)))
        x = self.bn5(self.af5(self.conv5(x)))
        x = self.bn6(self.af6(self.conv6(x)))
        x = self.bn7(self.af7(self.conv7(x)))
        x = self.bn8(self.af8(self.conv8(x)))
        x = self.bn9(self.af9(self.conv9(x)))
        x = self.bn10(self.af10(self.conv10(x)))
        x = self.bn11(self.af11(self.conv11(x)))
        x = self.bn12(self.af12(self.conv12(x)))
        x = self.bn13(self.af13(self.deconv13(x)))
        x = self.bn14(self.af14(self.conv14(x)))
        x = self.bn15(self.af15(self.deconv15(x)))
        x = self.bn16(self.af16(self.conv16(x)))
        x = self.act17(self.conv17(x))
        return x








