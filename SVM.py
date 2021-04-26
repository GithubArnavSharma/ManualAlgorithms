#Import neccessary modules
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

#Implementation of a basic Support Vector Machine(without kernels) in Python from scratch using numpy
class SupportVectorMachine:
    #Defining features of the Support Vector Machine
    def __init__(self, C=1, learning_rate=0.01, iters=5000, param_vectors=None, unique_y=None):
        self.C = C #Regularization parameter  Higher C --> higher variance and lower bias  Lower C --> lower variance and higher bias
        self.learning_rate = learning_rate #Learning rate for the model
        self.iters = iters #Amount of iterations/epochs to go through per class
        self.param_vectors = param_vectors #The storing of parameter vectors per class
        self.unique_y = unique_y #The storing of all unique instances in the y provided for training

    #Function to optimize parameters for the predicting of new inputs
    def fit(self, X, y):
        X = [np.insert(x, 0, 1) for x in X] #Insert 1 as the first instance of every array in X, as x(0) is 1 due to theta(0) being the bias 
        y = np.ravel(y)
        
        self.unique_y = np.unique(y)
        classes = len(self.unique_y)
        
        self.param_vectors = np.zeros((classes, len(X[0]), 1)) #Parameters will be initialized as 0 before training

        for n in range(classes): #Run through every class in the loop for the optimization of each parameter for the class
            #Since SVMs are meant for binary classification, transform y into -1 when y(i) =/= current class, 1 when y(i) = current_class
            y_modified = []
            for y_var in y:
                if y_var != self.unique_y[n]:
                    y_modified.append(np.array([-1]))
                else:
                    y_modified.append(np.array([1]))

            for x in range(self.iters): 
                weights = self.param_vectors[n] #The vector will be modified in the weights variable before modifying self.param_vectors
                for i in range(len(y)):
                    #theta-transpose-x(i) is supposed to be > 1 if y(i) = 1 and < -1 if y(i) = 0
                    hinge_loss = 1 - (y_modified[i] * np.dot(X[i], weights))
                    if max(0, hinge_loss) != 0: #If the hinge loss(maximum margin loss) is greater than 0:
                        #Calculate the gradient of the cost function and update the weight variable 
                        gradient = (self.C * y_modified[i] * X[i])
                        gradient = np.array([gradient]).reshape(-1, 1)
                        weights = weights - gradient
                weights = weights / len(y) #Average out the weight variable

                self.param_vectors[n] = self.param_vectors[n] - (self.learning_rate * weights) #Update the parameter vector, also using self.learning_rate

    #Function to predict new inputs
    def predict(self, X):
        #Insert 1 as x(0) for every single array in X so theta(0) continues to be the bias 
        X = [np.insert(x, 0, 1) for x in X]

        #Goes through every array in X, calculates self.param_vectors(n)-transpose-x(i), and the maximum result will be corresponding to the correct class
        preds = []
        for x in X:
            values = []
            for n in range(len(self.unique_y)):
                hyp = np.dot(x, self.param_vectors[n]) #theta-transpose-x(i)  theta = self.param_vectors(n)
                values.append(hyp)

            prediction = self.unique_y[values.index(max(values))]
            preds.append(prediction)
            
        return preds

    #Function to score the model with new X and y inputs
    def score(self, X, y):
        y = np.ravel(y)
        y_pred = self.predict(X)
        
        correct = sum([1 for i in range(len(y_pred)) if y_pred[i] == y[i]]) #Counter of all correct predictions
        total = len(y)
        return correct / total

#Function that inputs data in (X, y) format and outputs the models validation score 
def accuracy_model(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = SupportVectorMachine()
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)

datasets = [load_iris(return_X_y=True), load_breast_cancer(return_X_y=True), load_wine(return_X_y=True)]

for data in datasets:
    print(accuracy_model(data))

'''
Results:
Iris dataset: ~92% validation accuracy
Breast cancer dataset: ~89.5% validation accuracy
Wine dataset: ~77.7% validaton accuracy
'''
