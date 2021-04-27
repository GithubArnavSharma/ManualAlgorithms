import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

class ManualKMeans:
    def __init__(self, clusters, iters=2000, centroids=None, labels_=None):
        self.clusters = clusters
        self.iters = iters
        self.centroids = centroids
        self.labels_ = labels_
    
    def fit(self, X):
        X = np.array(X)
        self.centroids = [[None for x in range(len(X[0]))] for i in range(self.clusters)]
        
        for i in range(self.clusters):
            for j in range(len(X[0])):
                random_point = X[np.random.randint(0, len(X)), j]
                self.centroids[i][j] = random_point
                
        for x in range(self.iters):
            self.labels_ = []
            for i in range(len(X)):
                dists = []
                for j in range(len(self.centroids)):
                    dist = np.mean([(X[i][z] - self.centroids[j][z])**2 for z in range(len(X[i]))])
                    dists.append(dist)
                closest_index = dists.index(min(dists))
                self.labels_.append(closest_index)

            for i in range(len(self.centroids)):
                X_cluster = np.array([X[j] for j in range(len(X)) if self.labels_[j] == i])
                if len(X_cluster) == 0:
                    dists = []
                    for j in range(len(X)):
                        dist = np.mean([(X[j][z] - self.centroids[i][z])**2 for z in range(len(X[j]))])
                        dists.append(dist)
                    X_cluster = np.array([X[dists.index(max(dists))]])
                    
                avg_vector = [np.mean(X_cluster[:, j]) for j in range(len(X_cluster[0]))]
                self.centroids[i] = avg_vector

X, _ = make_blobs(n_samples=200, n_features=3, centers=5, cluster_std=0.7, random_state=0)

fig = plt.figure()
ax = plt.axes(projection='3d')

model = ManualKMeans(clusters=5)
model.fit(X)

ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=model.labels_)
centroids = np.array(model.centroids)
ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=250, marker='x', linewidths=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_ylim(1, 11)
ax.set_zlim(-11.5, 9.5)

plt.show()
