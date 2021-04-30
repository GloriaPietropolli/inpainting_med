"""
Function that compute the mean value of the pixels of the training set
I : train_dataset i.e. a list of 4D tensor
O : channel_total_mean i.e. a numpy array containing the mean along the channel values of the input training set
Implementation for a problem with 3 channel to estimate
"""
import torch
import numpy as np


def MV_pixel(train_dataset):
    channel_total_mean = np.zeros(shape=(4,))
    for train_tensor in train_dataset:
        tensor_mean = np.array(train_tensor.mean(axis=(0, 2, 3, 4)))  # mean value of the different channel
        channel_total_mean = channel_total_mean + tensor_mean
    channel_total_mean = channel_total_mean / len(train_dataset)
    return channel_total_mean
