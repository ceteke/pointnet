from torch import nn
import torch
import numpy as np
from sklearn.utils import shuffle as skshuffle
from data.utils import batchify

class BasePointNet(nn.Module):
    def __init__(self, n, dtype, cuda, device_id):
        '''
        :param dtype: 1 for (x, y, z) coordinates 2 for (x, y, z, rgb)
        :param cuda: Boolean indicating training device
        :param device_id: int indicating the CUDA device to put the model
        '''
        nn.Module.__init__(self)

        assert dtype in (1, 2), "dtype parameter can be either 1 or 2"
        self.cuda = cuda

        if self.cuda:
            self.device_id = 0 if device_id is None else device_id

        self.dtype = dtype
        self.input_channels = 3 if dtype == 1 else 4
        self.n = n

    def forward(self, input):
        raise NotImplemented

    def loss(self, input, target):
        raise NotImplemented

    def set_optimizer(self, optimizer):
        raise NotImplemented

    def fit(self, **kwargs):
        raise NotImplemented