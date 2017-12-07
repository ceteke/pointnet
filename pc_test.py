import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle
from sklearn import preprocessing

points = pickle.load(open('model10_test.pk', 'rb'))

points = np.array(points['bathtub'])[0]
print(points.shape)

with open('test.pcd', 'w') as f:
  for p in points:
    f.write('{} {} {}\n'.format(p[0],p[1],p[2]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=0.5)

plt.show()
