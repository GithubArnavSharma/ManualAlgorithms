import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split

class ManualLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, param_vectors=None, unique_y=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.param_vectors = param_vectors
        self.unique_y = unique_y

    def sigmoid(self, input_num):
        sig_num = 1/(1+np.exp(-input_num))
        return sig_num

    def fit(self, X, y):
        X = [np.insert(x, 0, 1) for x in X]
        y = np.ravel(y)
        
        self.unique_y = np.unique(y)
        classes = len(self.unique_y)

        self.param_vectors = [np.random.randn(len(X[0]), 1) for i in range(classes)]

        for n in range(classes):
            y_modified = []
            for y_var in y:
                if y_var != self.unique_y[n]:
                    y_modified.append(0)
                else:
                    y_modified.append(1)
                    
            for x in range(self.max_iter):
                for j in range(len(self.param_vectors[n])):
                    hyps = np.dot(X, self.param_vectors[n])

                    gradients = []
                    for i in range(len(hyps)):
                        gradient = (self.sigmoid(hyps[i]) - y_modified[i]) * X[i][j]
                        gradients.append(gradient)
                    cost_der = self.learning_rate * np.mean(gradients)

                    self.param_vectors[n][j] -= cost_der

    def predict(self, X, use_prob=False):
        X = [np.insert(x, 0, 1) for x in X]

        preds = []
        for x in X:
            probs = []
            for n in range(len(self.unique_y)):
                hyp = np.dot(x, self.param_vectors[n])
                prob = self.sigmoid(hyp)
                probs.append(prob)

            prediction = self.unique_y[probs.index(max(probs))]
            if not use_prob:
                preds.append(prediction)
            else:
                preds.append(probs)
            
        return preds

    def score(self, X, y):
        y = np.ravel(y)
        y_pred = self.predict(X)

        correct = sum([1 for i in range(len(y_pred)) if y_pred[i] == y[i]])
        total = len(y)
        return correct / total

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = ManualLogisticRegression(max_iter=250)
model.fit(X_train, y_train)

print(model.score(X_test, y_test)) #91%

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = ManualLogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

print(model.score(X_test, y_test)) #90%

X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.9)

model = ManualLogisticRegression(max_iter=1500)
model.fit(X, y)

xx, yy = np.mgrid[min(list(X[:, 0]))-1:max(list(X[:, 0]))+1:0.01, min(list(X[:, 1]))-1:max(list(X[:, 1]))+1:0.01]
grid = np.c_[xx.ravel(), yy.ravel()]

probs = np.array(model.predict(grid, use_prob=True))[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, probs, 25, cmap="RdBu")
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="RdBu", edgecolor="white", linewidth=1)

plt.show()
