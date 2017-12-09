from torch import nn
from torch import optim
from torch.nn import functional as F
from .base import BasePointNet, Flatten, View2d
from .t_nets import TransormationNet
from torch.optim.lr_scheduler import StepLR
import torch

class VanillaPointNetClassifier(BasePointNet):
    def __init__(self, n, num_class, cuda, device_id=None):
        BasePointNet.__init__(self, n, cuda, device_id)
        self.num_class = num_class
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 1024, 1),
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
        optimizer = optim.Adam(self.parameters())
        self.optimizer = StepLR(optimizer, 20, 0.5)

    def loss(self, input, target):
        return F.nll_loss(input, target)

    def forward(self, input):
        return self.net(input)

class PointNetClassifier(BasePointNet):
    def __init__(self, n, lr, wd, dropout, lambd, num_class, cuda, device_id=None):
        BasePointNet.__init__(self, n, lr, wd, cuda, device_id)
        self.num_class = num_class
        self.lamd = lambd
        self.net = nn.Sequential(
            TransormationNet(3, self.n, 1),
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            TransormationNet(64, self.n, 64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.MaxPool2d((self.n, 1), 1),
            Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, num_class),
            nn.LogSoftmax()
        )

        self.optimizer = optim.Adam(self.parameters())
        self.scheduler = StepLR(self.optimizer, 20, 0.5)

    def loss(self, input, target):
        feature_transform_matrix = self.net[7].t_out
        eye = torch.autograd.Variable(torch.eye(64)) # requires_grad=False by default
        if self._cuda:
            eye = eye.cuda(self.device_id)
        transfer_reg = self.lamd * torch.sum(
            torch.pow(
                torch.norm(
                    eye.sub(torch.matmul(feature_transform_matrix, torch.transpose(feature_transform_matrix,1,2))) + 1e-8
                ), 2))
        return F.nll_loss(input, target) + transfer_reg

    def forward(self, input):
        return self.net(input)