from torch import nn
import torch
from torch.nn import functional as F
from data.utils import batchify
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
from sklearn.utils import shuffle as skshuffle

class VanillaPointVAE(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self.conv1 = nn.Conv2d(1, 64, (1, 3))
    self.conv2 = nn.Conv2d(64, 64, 1)
    self.conv3 = nn.Conv2d(64, 128, 1)
    self.conv4 = nn.Conv2d(128, 1024, 1)

    self.pool = nn.MaxPool2d((1024, 1), 1, return_indices=True)

    self.fc11 = nn.Linear(1024, 100)
    self.fc12 = nn.Linear(1024, 100)
    self.fc2 = nn.Linear(100, 1024)

    self.unpool = nn.MaxUnpool2d((1024, 1), 1)

    self.convt1 = nn.ConvTranspose2d(1024, 128, 1)
    self.convt2 = nn.ConvTranspose2d(128, 64, 1)
    self.convt3 = nn.ConvTranspose2d(64, 64, 1)
    self.convt4 = nn.ConvTranspose2d(64, 1, (1, 3))

    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    self.scheduler = None
    self._cuda = True
    self.device_id = 0

  def build(self):
    self.cuda()

  def go_up(self, layer, rec, skip, activation=True):
    reconstruction = skip + rec
    reconstruction = layer(reconstruction)
    if activation:
      return F.relu(reconstruction)
    return reconstruction

  def loss(self, reconstructed, input, mu, sgm):
    rec_loss = F.mse_loss(reconstructed, input, size_average=False)
    kl_div = -0.5 * torch.sum(1 + torch.log(sgm.pow(2)) - mu.pow(2) - sgm.pow(2))
    return rec_loss + kl_div

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x_skip1 = x
    x = self.conv2(x)
    x = F.relu(x)
    x_skip2 = x
    x = self.conv3(x)
    x = F.relu(x)
    x_skip3 = x
    x = self.conv4(x)
    x = F.relu(x)
    x_skip4 = x

    pooled, pool_idx = self.pool(x)
    pooled = pooled.view(pooled.size(0), -1) # Flatten

    mu = self.fc11(pooled)
    log_sgm_sq = self.fc12(pooled)

    sgm = torch.exp(0.5 * log_sgm_sq)
    eps = torch.autograd.Variable(sgm.data.new(sgm.size()).normal_())

    latent = mu + sgm * eps

    latent = self.fc2(latent)
    latent = F.relu(latent)
    latent = latent.view(-1,1024,1,1)

    reconstruction = self.unpool(latent, pool_idx)

    reconstruction = self.go_up(self.convt1, reconstruction, x_skip4)
    reconstruction = self.go_up(self.convt2, reconstruction, x_skip3)
    reconstruction = self.go_up(self.convt3, reconstruction, x_skip2)
    reconstruction = self.go_up(self.convt4, reconstruction, x_skip1, activation=False)

    return reconstruction, mu, sgm

  def fit(self, X_train, batch_size):
    self.train()
    if self. scheduler is not None:
      self.scheduler.step()
    losses = []
    X_train = skshuffle(X_train)
    X_train = X_train.reshape((-1, 1, 1024, 3))
    X_train_tensor = FloatTensor(X_train.tolist())
    for x_batch in batchify(X_train_tensor, batch_size):
      x_b = Variable(x_batch)
      if self._cuda:
        x_b = x_b.cuda(self.device_id)
      self.optimizer.zero_grad()
      logits, mu, sigma = self(x_b)
      ce = self.loss(logits, x_b, mu, sigma)
      losses.append(ce.data[0])
      ce.backward()
      self.optimizer.step()
    return losses

  def score(self, X, batch_size):
    self.eval()
    X = X.reshape(-1, 1, 1024, 3)
    X_tensor = FloatTensor(X.tolist())
    total_loss = 0.0
    for x_batch in batchify(X_tensor, batch_size):
      x_b = Variable(x_batch)
      if self._cuda:
        x_b = x_b.cuda(self.device_id)
      output, mu, sigma = self(x_b)
      total_loss += self.loss(output, x_b, mu, sigma)
    return total_loss / len(X)