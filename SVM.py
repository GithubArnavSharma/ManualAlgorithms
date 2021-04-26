import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_blobs
from sklearn.model_selection import train_test_split

class SupportVectorMachine:
    def __init__(self, C=1, learning_rate=0.01, iters=1000, param_vectors=None, unique_y=None):
        self.C = C
        self.learning_rate = learning_rate
        self.iters = iters
        self.param_vectors = param_vectors
        self.unique_y = unique_y

    def fit(self, X, y):
        X = [np.insert(x, 0, 1) for x in X]
        y = np.ravel(y)
        
        self.unique_y = np.unique(y)
        classes = len(self.unique_y)
        
        self.param_vectors = np.zeros((classes, len(X[0]), 1))

        for n in range(classes):
            y_modified = []
            for y_var in y:
                if y_var != self.unique_y[n]:
                    y_modified.append(np.array([-1]))
                else:
                    y_modified.append(np.array([1]))

            for x in range(self.iters):
                weights = self.param_vectors[n]
                for i in range(len(y)):
                    hinge_loss = 1 - (y_modified[i] * np.dot(X[i], weights))
                    if max(0, hinge_loss) != 0:
                        gradient = (self.C * y_modified[i] * X[i])
                        gradient = np.array([gradient]).reshape(-1, 1)
                        weights = weights - gradient
                weights = weights / len(y)

                self.param_vectors[n] = self.param_vectors[n] - (self.learning_rate * weights)

    def predict(self, X):
        X = [np.insert(x, 0, 1) for x in X]

        preds = []
        for x in X:
            values = []
            for n in range(len(self.unique_y)):
                hyp = np.dot(x, self.param_vectors[n])
                values.append(hyp)

            prediction = self.unique_y[values.index(max(values))]
            preds.append(prediction)
            
        return preds

    def score(self, X, y):
        y = np.ravel(y)
        y_pred = self.predict(X)
        
        correct = sum([1 for i in range(len(y_pred)) if y_pred[i] == y[i]])
        total = len(y)
        return correct / total

def accuracy_model(data, iters):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = SupportVectorMachine(iters=iters)
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)

datasets = [(load_iris(return_X_y=True), 5000), (load_breast_cancer(return_X_y=True), 5000), (load_wine(return_X_y=True), 5000)]

for data in datasets:
    print(accuracy_model(data[0], data[1]))

        
                    
            

        
