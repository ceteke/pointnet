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
    def __init__(self, n, lr, wd, dropout, num_class, cuda, device_id=None):
        BasePointNet.__init__(self, n, lr, wd, cuda, device_id)
        self.num_class = num_class
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.MaxPool1d(self.n, 1),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, num_class),
            nn.LogSoftmax()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def build(self):
        if self._cuda:
            self.cuda(self.device_id)

    def loss(self, input, target):
        return F.nll_loss(input, target)

    def forward(self, input):
        return self.net(input)