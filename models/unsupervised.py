from torch import nn
import torch
from data.utils import batchify
from torch.autograd import Variable
from torch import FloatTensor
from sklearn.utils import shuffle as skshuffle
from torch.optim.lr_scheduler import StepLR

class VanillaPointAE(nn.Module):
  def __init__(self, n):
    nn.Module.__init__(self)
    self.n = n
    self.encoder = nn.Sequential(
      nn.Conv1d(3, 128, 1),
      nn.ReLU(),
      nn.BatchNorm1d(128),
      nn.Conv1d(128, 128, 1),
      nn.ReLU(),
      nn.BatchNorm1d(128),
      nn.Conv1d(128, 256, 1),
      nn.ReLU(),
      nn.BatchNorm1d(256),
      nn.Conv1d(256, 512, 1),
      nn.ReLU(),
      nn.BatchNorm1d(512),
      nn.MaxPool1d(self.n, 1),
    )

    self.decoder = nn.Sequential(
      nn.Linear(512, self.n),
      nn.ReLU(),
      nn.BatchNorm1d(self.n),
      nn.Linear(self.n, 2048),
      nn.ReLU(),
      nn.BatchNorm1d(2048),
      nn.Linear(2048, self.n * 3),
    )

    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    self.scheduler = StepLR(self.optimizer, 20, 0.5)
    self._cuda = True
    self.device_id = 0

  def build(self):
    self.cuda()

  def loss(self, reconstructed, input):
    return torch.sqrt(torch.sum(torch.pow(input-reconstructed, 2)))

  def forward(self, x):
    latent = torch.squeeze(self.encoder(x))
    reconstructed = self.decoder(latent)
    return reconstructed.view(-1, 3, self.n)

  def get_representation(self, X, batch_size):
    self.eval()
    representation_batches = []
    X_tensor = FloatTensor(X.tolist())
    for data in batchify(X_tensor, batch_size):
      x_b = torch.autograd.Variable(data, volatile=True)
      if self._cuda:
        x_b = x_b.cuda(self.device_id)
      rep = self.encoder(x_b)
      representation_batches.append(rep)
    representation = torch.squeeze(torch.cat(representation_batches))
    return representation.cpu().data

  def fit(self, X_train, batch_size):
    self.train()
    if self. scheduler is not None:
      self.scheduler.step()
    losses = 0
    X_train = skshuffle(X_train)
    X_train_tensor = FloatTensor(X_train.tolist())
    for x_batch in batchify(X_train_tensor, batch_size):
      x_b = Variable(x_batch)
      if self._cuda:
        x_b = x_b.cuda(self.device_id)
      self.optimizer.zero_grad()
      recons = self(x_b)
      ce = self.loss(recons, x_b)
      ce.backward()
      losses += ce.data[0]
      self.optimizer.step()
    return losses / len(X_train)

  def score(self, X, batch_size):
    self.eval()
    X_tensor = FloatTensor(X.tolist())
    total_loss = 0.0
    for x_batch in batchify(X_tensor, batch_size):
      x_b = Variable(x_batch)
      if self._cuda:
        x_b = x_b.cuda(self.device_id)
      output = self(x_b)
      total_loss += self.loss(output, x_b).data[0]
    return total_loss / len(X)