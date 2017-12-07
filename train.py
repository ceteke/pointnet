from models.classification import VanillaPointNetClassifier, PointNetClassifier
from data.datasets import ModelNet10

model = PointNetClassifier(1024, 10, True)
model.build()

print("Reading dataset")
dataset = ModelNet10('model10_train.pk', 'model10_test.pk', 1024)

X_train, y_train, X_test, y_test = dataset.process()

for e in range(100):
  print('fitting')
  model.fit(X_train, y_train, 64)
  print('train acc')
  model.eval()
  print(model.score(X_train, y_train, 200))
  print('test acc')
  print(model.score(X_test, y_test, 200))