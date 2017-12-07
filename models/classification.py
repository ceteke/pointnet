from torch import nn
from torch import optim
from torch.nn import functional as F
from .base import BasePointNet, Flatten, View2d
from .t_nets import TransormationNet

class VanillaPointNetClassifier(BasePointNet):
    def __init__(self, n, num_class, cuda, device_id=None):
        BasePointNet.__init__(self, n, cuda, device_id)
        self.num_class = num_class
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 128, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 1024, (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((self.n, 1), 1),
            Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, num_class),
            nn.LogSoftmax()
        )

        self.optimizer = optim.Adam(self.parameters())

    def build(self):
        if self._cuda:
            self.cuda(self.device_id)

    def loss(self, input, target):
        return F.nll_loss(input, target)

    def forward(self, input):
        return self.net(input)

class PointNetClassifier(BasePointNet):
    def __init__(self, n, num_class, cuda, device_id=None):
        BasePointNet.__init__(self, n, cuda, device_id)
        self.num_class = num_class
        self.net = nn.Sequential(
            TransormationNet(3, self.n, self._cuda, self.device_id),
            View2d(1, 1024, 3),
            nn.Conv2d(1, 64, (1, 3)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1024, (1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d((self.n, 1), 1),
            Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_class),
            nn.LogSoftmax()
        )

        self.optimizer = optim.Adam(self.parameters())

    def build(self):
        if self._cuda:
            self.cuda(self.device_id)

    def loss(self, input, target):
        return F.nll_loss(input, target)

    def forward(self, input):
        return self.net(input)