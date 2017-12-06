from torch import nn
from torch.nn import functional as F
from .base import BasePointNet

class VanillaPointNet(BasePointNet):
    def __init__(self, n, num_class, dtype, cuda, device_id=None):
        BasePointNet.__init__(self, n, dtype, cuda, device_id)
        self.num_class = num_class
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (1, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 128, (1, 3)),
            nn.ReLU(True),
            nn.Conv2d(128, 1024, (1, 3)),
            nn.ReLU(True),
            nn.MaxPool2d((self.n, 1), 1),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, num_class),
            nn.LogSoftmax()
        )

    def loss(self, input, target):
        return F.nll_loss(input, target)

    def forward(self, input):
        return self.net(input)
