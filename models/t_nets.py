from torch import nn, matmul
from .base import BasePointNet, Flatten
import torch

class TransormationNet(BasePointNet):
  def __init__(self, K, n, cuda, device_id=None):
    BasePointNet.__init__(self, n, cuda, device_id)
    self.K = K
    self.net = nn.Sequential(
      nn.Conv1d(3, 64, 1),
      nn.ReLU(True),
      nn.Conv1d(64, 128, 1),
      nn.ReLU(True),
      nn.Conv1d(128, 1024, 1),
      nn.ReLU(True),
      nn.MaxPool1d(self.n, 1),
      Flatten(),
      nn.Linear(1024, 512),
      nn.ReLU(True),
      nn.Linear(512, 256),
      nn.ReLU(True),
      nn.Linear(256, K**2),
    )

    if self._cuda:
      self.cuda(self.device_id)

  def forward(self, input):
    t_out = self.net(input).view(-1,self.K,self.K)
    input = torch.squeeze(input)
    result = matmul(torch.transpose(input, 1, 2), t_out)
    return torch.transpose(result, 1, 2)