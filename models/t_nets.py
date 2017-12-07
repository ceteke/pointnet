from torch import nn, matmul
from .base import BasePointNet, Flatten
import torch

class TransormationNet(BasePointNet):
  def __init__(self, K, n, cuda, device_id=None):
    BasePointNet.__init__(self, n, cuda, device_id)
    self.K = K
    self.net = nn.Sequential(
      nn.Conv2d(1, 64, (1, 3)),
      nn.ReLU(True),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 128, (1, 1)),
      nn.ReLU(True),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 1024, (1, 1)),
      nn.ReLU(True),
      nn.MaxPool2d((self.n, 1), 1),
      Flatten(),
      nn.Linear(1024, 512),
      nn.ReLU(True),
      nn.BatchNorm1d(512),
      nn.Linear(512, 256),
      nn.ReLU(True),
      nn.BatchNorm1d(256),
      nn.Linear(256, K**2),
    )

    if self._cuda:
      self.cuda(self.device_id)

    self.is_first = True

  def forward(self, input):
    t_out = self.net(input).view(-1,self.K,self.K)
    input = torch.squeeze(input)
    result = matmul(input, t_out)
    return result