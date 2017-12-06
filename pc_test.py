import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

with open('bathtub_0001.off', 'r') as f:
    points = f.readlines()

for i, p in enumerate(points):
    points[i] = [float(coor) for coor in p.strip().split(' ')]

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=0.5)

plt.show()
