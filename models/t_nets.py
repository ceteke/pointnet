from torch import nn, matmul
from .base import Flatten
import torch

class TransormationNet(nn.Module):
  def __init__(self, K, n, channels):
    nn.Module.__init__(self)
    self.n = n
    self.K = K
    self.channels = channels
    self.kernel_size = 3 if channels == 1 else 1
    self.net = nn.Sequential(
      nn.Conv2d(channels, 64, (1, self.kernel_size)),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.Conv2d(64, 128, 1),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.Conv2d(128, 1024, 1),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.MaxPool2d((self.n, 1), 1),
      Flatten(),
      nn.Linear(self.n, 512),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      nn.Linear(512, 256),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      nn.Linear(256, K**2),
    )

  def forward(self, input):
    t_out = self.net(input).view(-1,self.K,self.K)
    input = torch.squeeze(input)
    if self.channels != 1:
      input = torch.transpose(input, 1, 2)
    result = matmul(input, t_out)
    if self.channels != 1:
      result = torch.transpose(result, 2, 1).contiguous()
      return result.view(-1,self.channels,self.n,1)
    return result.view(-1,1,self.n,3)