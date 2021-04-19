#Import all neccessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

#ManualLinearRegression class, which will use machine learning to possess the ability to use Linear Regression on any dimension X and y
class ManualLinearRegression:
    #All variables within the initializer do not need to be specified unless neccessary by whoever is using it
    def __init__(self, learning_rate=0.01, max_iter=1000, param_vector=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.param_vector = param_vector
        
    #Function which can input X and y variables needed for training and update the parameters so future predictions can be made
    def fit(self, X, y):
        #Since θ(0) will be the y_intercept, it will be multiplied by x(0) which would be 1, so 1 needs to be inserted at the beginning of each data point
        X = [np.insert(x, 0, 1) for x in X]
        
        #Randomly initialize parameter θ(0) - θ(max) to be multipled by x(0) - x(max)
        self.param_vector = np.random.randn(len(X[0]), 1)
        
        #Go through the same process for each iteration
        for x in range(self.max_iter):
            #Each parameter from param_vector will be updated seperately, to their corresponding feature in x
            for j in range(len(self.param_vector)):
                #Compute the hypothesis' by multiplying each array of X to the parameter vector
                hyps = np.dot(X, self.param_vector)
                
                gradients = []
                for i in range(len(hyps)):
                    #The gradient of the loss is calculated by multiplying the difference between the hypothesis and actual y by the corresponding feature
                    gradient = (hyps[i] - y[i]) * X[i][j]
                    gradients.append(gradient)
                #Find the cost by multiplying the learning rate by the average of all the gradients
                cost_der = self.learning_rate * np.mean(gradients)
                
                #Subtract the individual parameter by the calculated cost
                self.param_vector[j] -= cost_der
        
    #Function that can input a unique X and output a prediction using the parameters learned from .fit
    def predict(self, X):
        #For each arrat of X, insert 1 beforehand to be the x(0) multiplied by θ(0)
        X = [np.insert(x, 0, 1) for x in X]
        #Make the prediction by multiplying each array of X by the parameters calculated
        prediction = np.dot(X, self.param_vector)
        return prediction

    #An r2 score is a scoring used for regression problems. This function inputs an X and y and computes the r2_score(0-1) of the actual y and the linear regression predictions
    def score(self, X, y):
        r2score = r2_score(y, self.predict(X))
        return r2score

#This first example will use 3 dimensional vectors(X is 2 dimensional and y is 1 dimensional) that have a regression pattern to test the ManualLinearRegression class
X1, X2 = np.linspace(0, 10, 200), np.random.randint(0, 10, 200)
X = [[X1[i], X2[i]] for i in range(len(X1))]
y = np.linspace(0, 5, 200)

#Scatter the values of X and y onto a 3 dimensional figure
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], y[i])

#Use the ManualLinearRegression class to generate parameters for the input X and y data 
model = ManualLinearRegression()
model.fit(X, y)

#With the parameters, X, and y, the rectangular plane denoting what the Linear Regression learned can be generated
X_mesh, Y_mesh = np.meshgrid(X, y)
params = model.param_vector
Z = params[0] + params[1]*X_mesh + params[2]*Y_mesh

#Plot the plane onto the figure 
ax.plot_surface(X_mesh, Y_mesh, Z)
ax.set(xlabel='x1', ylabel='x2', zlabel='y', title='Multiple Linear Regression')

#Show the figure 
plt.show()

#The second test will involve using the .score part of ManualLinearRegression to test the ability to generate comprehensive parametters  
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Fit the model on the training data 
model_train = ManualLinearRegression()
model_train.fit(X_train, y_train)
#Get the score for the testing data(the score was 0.9960475699753332, signalling that the model was able to learn well)
print(model_train.score(X_test, y_test))
