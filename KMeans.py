#Import neccessary modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

#Class to implement the unsupervised KMeans algorithm in Python from scratch using numpy
class ManualKMeans:
    #Defining features of the ManualKMeans class
    def __init__(self, clusters, iters=2000, centroids=None, labels_=None):
        self.clusters = clusters #Amount of clusters specified to split the input data into
        self.iters = iters #Amount of iterations/epochs to go through the training process
        self.centroids = centroids #List of the vectors of every single centroid
        self.labels_ = labels_ #A cluster label given to every single data point 
    
    #Function to implement the K Means algorithm given input data. Results of function can be seen through self.labels_
    def fit(self, X):
        X = np.array(X)
        self.centroids = [[None for x in range(len(X[0]))] for i in range(self.clusters)] #Define empty centroids values to later be changed
        
        #Initialize the original centroids vectors to later be changed 
        for i in range(self.clusters):
            for j in range(len(X[0])):
                #Get a random number from a random input data vector, and add it to self.centroids
                random_point = X[np.random.randint(0, len(X)), j]
                self.centroids[i][j] = random_point
                
        for x in range(self.iters): 
            self.labels_ = []
            for i in range(len(X)):
                dists = []
                #Calculate the distance from every single input data point to every single centroid
                for j in range(len(self.centroids)):
                    dist = np.mean([(X[i][z] - self.centroids[j][z])**2 for z in range(len(X[i]))]) #||x(i) - self.centroids(j)||**2
                    dists.append(dist)
                #Whichever centroid is closest to the data point will be the centroid index assigned to that data point
                closest_index = dists.index(min(dists))
                self.labels_.append(closest_index)

            for i in range(len(self.centroids)):
                #Get a list of every single data point that was assigned to self.centroids(i)
                X_cluster = np.array([X[j] for j in range(len(X)) if self.labels_[j] == i])
                #If the centroid is not close to any data point, change its location to the data point of maximum distance away from it
                if len(X_cluster) == 0:
                    dists = []
                    for j in range(len(X)):
                        dist = np.mean([(X[j][z] - self.centroids[i][z])**2 for z in range(len(X[j]))])
                        dists.append(dist)
                    X_cluster = np.array([X[dists.index(max(dists))]])
                    
                #Change the centroid vector to be the average of all the data vectors it is closest to
                avg_vector = [np.mean(X_cluster[:, j]) for j in range(len(X_cluster[0]))]
                self.centroids[i] = avg_vector

#Prepare input data points, which will have 5 centers/clusters
X, _ = make_blobs(n_samples=200, n_features=3, centers=5, cluster_std=0.7, random_state=0)

fig = plt.figure()
ax = plt.axes(projection='3d')

#Use the ManualKMeans class to fit the unsupervised algorithm model to X
model = ManualKMeans(clusters=5)
model.fit(X)

#Scatter the data points and the model's results, as well as the centroids 
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=model.labels_)
centroids = np.array(model.centroids)
ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=250, marker='x', linewidths=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_ylim(1, 11)
ax.set_zlim(-11.5, 9.5)

plt.show() #Results show that the KMeans algorithm effectively is able to cluster data points 
