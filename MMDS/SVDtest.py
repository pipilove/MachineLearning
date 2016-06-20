import numpy as np
import scipy as sp
from scipy import linalg, spatial

x = np.array([[5, 0, 0, 0, 0], [0, 2, 0, 0, 4]])
x = np.array([[5, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 0, 0, 0, 4]])
V = np.array([[-0.57, -0.11, -0.57, -0.11, -0.57], [-0.09, 0.7, -0.09, 0.7, -0.09]])

sim = x.dot(V.T)
print(sim)
print(spatial.distance.cdist(sim[0].reshape((1, 2)), sim[1].reshape((1, 2)), metric='cosine'))
print('similarity: ', 1 - spatial.distance.pdist(sim, metric='cosine'))
