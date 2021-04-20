#Import neccessary libaries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split

#Class to use numpy for Logistic Regression. Logistic Regression is Linear Regression with a sigmoid function to turn it into a classifier
class ManualLogisticRegression:
    #Specify variables that will be used in future functions within the class
    def __init__(self, learning_rate=0.01, max_iter=1000, param_vectors=None, unique_y=None):
        self.learning_rate = learning_rate #learning rate of the logistic regression fitting 
        self.max_iter = max_iter #Amount of iterations/epochs to pass through every class
        self.param_vectors = param_vectors #List storing parameter vectors for each class
        self.unique_y = unique_y #List storing all of the unique values of y in the same order

    #Function which takes an input number and applies the sigmoid/logistic function on it
    def sigmoid(self, input_num):
        sig_num = 1/(1+np.exp(-input_num))
        return sig_num

    #Function which tunes parameters in self.param_vectors(Logistic Regression is not just for binary cases) using an input X and y 
    def fit(self, X, y):
        X = [np.insert(x, 0, 1) for x in X] #Insert a 1 in the beginning of every array in X as x(0) = 1
        y = np.ravel(y)
        
        self.unique_y = np.unique(y)
        classes = len(self.unique_y)

        self.param_vectors = [np.random.randn(len(X[0]), 1) for i in range(classes)] #Randomly initialize parameters to calculate P(y=class | X | parameters)
        
        for n in range(classes): #Tune each parameter from each class
            #Turn each individual class into a binary problem. Class which is not the input class is 0, class which is the input class is 1
            y_modified = []
            for y_var in y:
                if y_var != self.unique_y[n]:
                    y_modified.append(0)
                else:
                    y_modified.append(1)
                
            for x in range(self.max_iter): #Go through all the iterations/epochs
                for j in range(len(self.param_vectors[n])): #Go through each parameter from the specified parameter vector 
                    hyps = np.dot(X, self.param_vectors[n]) #Multiply all arrays in X with the specified parameter vector

                    #Calculate the gradient of the predicted array, the actual y(binary array), and the specific feature of the x array
                    gradients = []
                    for i in range(len(hyps)):
                        gradient = (self.sigmoid(hyps[i]) - y_modified[i]) * X[i][j]
                        gradients.append(gradient)
                        
                    #Calculate the cost and subtract the parameter by that cost
                    cost_der = self.learning_rate * np.mean(gradients)
                    self.param_vectors[n][j] -= cost_der

    #Function which takes input X and uses the parameter vectors to find the class with the highest probability. If use_prob is specified, the output is the probability array
    def predict(self, X, use_prob=False):
        X = [np.insert(x, 0, 1) for x in X] #Insert a 1 in the beginning of every array in X as x(0) = 1

        preds = []
        for x in X: #Go through each array from X
            probs = []
            #Calculate P(y=class | X | parameters) for each unique class
            for n in range(len(self.unique_y)):
                hyp = np.dot(x, self.param_vectors[n])
                prob = self.sigmoid(hyp)
                probs.append(prob)

            #Match the highest probability with the unique y and append to preds array according to the use_prob bool
            prediction = self.unique_y[probs.index(max(probs))]
            if not use_prob:
                preds.append(prediction)
            else:
                preds.append(probs)
            
        return preds

    #Function to calculate the accuracy of the parameters with a specified X and y(Probability represented from 0-1)
    def score(self, X, y):
        y = np.ravel(y)
        y_pred = self.predict(X)

        correct = sum([1 for i in range(len(y_pred)) if y_pred[i] == y[i]]) #Find the total number of correct class predictions 
        total = len(y)
        return correct / total 

#The first test of Logistic Regression uses the breast cancer binary dataset 
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = ManualLogisticRegression(max_iter=250)
model.fit(X_train, y_train)

print(model.score(X_test, y_test)) #The model gets a 91% on the test set

#The second test of Logistic Regression uses the iris leaf non-binary dataset
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = ManualLogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

print(model.score(X_test, y_test)) #The model gets 90% on the test set

#The third test of Logistic Regression uses a 2 dimensional binary dataset to visualize the decision boundary of the Logistic Regression model
X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.8, random_state=0)

model = ManualLogisticRegression(max_iter=750)
model.fit(X, y)

#Store the minimum and maximum of x1 and x2 to fill in the graph correctly
min_x1, min_x2 = min(list(X[:, 0]))-1, min(list(X[:, 1]))-1
max_x1, max_x2 = max(list(X[:, 0]))+1, max(list(X[:, 1]))+1

#Create a grid of points going through the minimums and maximums and use the model to predict the classes for all those points
xx, yy = np.mgrid[min_x1:max_x1:0.01, min_x2:max_x2:0.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = np.array(model.predict(grid, use_prob=True))[:, 1].reshape(xx.shape)

#Add the decision boundary coloring and the individual points onto the figure
plt.contourf(xx, yy, probs, 25, cmap="RdBu")
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="RdBu", edgecolor="white", linewidth=1)
plt.show()
