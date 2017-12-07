import pickle
import numpy as np
from .utils import normalize_unit_sphere

class DatasetBase(object):
  def __init__(self, train_file, test_file):
    self.train_pk = pickle.load(open(train_file, 'rb'))
    self.test_pk = pickle.load(open(test_file, 'rb'))

class ModelNet(DatasetBase):
  def __init__(self, train_file, test_file, n_samples):
    DatasetBase.__init__(self, train_file, test_file)
    self.labels = None
    self.n_samples = n_samples

  def process(self):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for k, v in self.train_pk.items():
      label = self.labels[k]
      for d in v:
        d = np.array(d)
        if d.shape[0] != self.n_samples:
          d = np.concatenate((d, np.zeros((self.n_samples - d.shape[0], 3))))
        X_train.append(normalize_unit_sphere(d))
      y_train += [label] * len(v)

    for k, v in self.test_pk.items():
      label = self.labels[k]
      for d in v:
        d = np.array(d)
        if d.shape[0] != self.n_samples:
          d = np.concatenate((d, np.zeros((self.n_samples - d.shape[0], 3))))
        X_test.append(normalize_unit_sphere(d))
      y_test += [label] * len(v)

    X_train = np.array(X_train).reshape((-1, 3, self.n_samples))
    X_test = np.array(X_test).reshape((-1, 3, self.n_samples))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

class ModelNet10(ModelNet):
  def __init__(self, train_file, test_file, n_samples):
    ModelNet.__init__(self, train_file, test_file, n_samples)
    self.labels = {'bathtub': 0,
                   'bed': 1,
                   'chair': 2,
                   'desk': 3,
                   'dresser': 4,
                   'monitor': 5,
                   'night_stand': 6,
                   'sofa': 7,
                   'table': 8,
                   'toilet': 9,
                   }
