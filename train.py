from models.classification import PointNetClassifier
from data.datasets import ModelNet10
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--did', type=int, default=0, help='Device ID')
parser.add_argument('--n_point', type=int, default=1024, help='Number of points [default: 1024]')
parser.add_argument('--n_class', type=int, default=10, help='Number of classes [default: 10]')
parser.add_argument('--epochs', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--wd', type=float, default=0.0, help='Weight decay [default: 1e-5]')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate [default: 0.3]')
FLAGS = parser.parse_args()

model = PointNetClassifier(n=FLAGS.n_point,lr=FLAGS.lr, wd=FLAGS.wd, dropout=FLAGS.dropout, num_class=FLAGS.n_class,
                           cuda=True, device_id=FLAGS.did)
model.build()

print("Reading dataset")
dataset = ModelNet10('model10_train.pk', 'model10_test.pk', FLAGS.n_point)

X_train, y_train, X_test, y_test = dataset.process()

all_losses = []
train_acc = []
test_acc = []
for e in range(FLAGS.epochs):
  print("Epoch {}:".format(e+1), flush=True)
  epoch_losses = model.fit(X_train, y_train, FLAGS.batch_size)
  print("\tMean Loss:", np.mean(epoch_losses), flush=True)
  all_losses += epoch_losses
  tr_acc = model.score(X_train, y_train, 64)
  print("\tTraining Accuracy", tr_acc, flush=True)
  ts_acc = model.score(X_test, y_test, 64)
  print("\tTest Accuracy", ts_acc, flush=True)
  train_acc.append(tr_acc)
  test_acc.append(ts_acc)
  print("===========", flush=True)

print("All losses", flush=True)
print(all_losses, flush=True)
print("All Train Acc", flush=True)
print(train_acc, flush=True)
print("All Test Acc", flush=True)
print(test_acc, flush=True)