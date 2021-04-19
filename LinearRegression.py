import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

class ManualLinearRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, param_vector=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.param_vector = param_vector
        
    def fit(self, X, y):
        X = [np.insert(x, 0, 1) for x in X]
        
        self.param_vector = np.random.randn(len(X[0]), 1)
        
        for x in range(self.max_iter):
            for j in range(len(self.param_vector)):
                hyps = np.dot(X, self.param_vector)
                
                gradients = []
                for i in range(len(hyps)):
                    gradient = (hyps[i] - y[i]) * X[i][j]
                    gradients.append(gradient)
                cost_der = self.learning_rate * np.mean(gradients)
                
                self.param_vector[j] -= cost_der
        
    def predict(self, X):
        X = [np.insert(x, 0, 1) for x in X]
        prediction = np.dot(X, self.param_vector)
        return prediction

    def score(self, X, y):
        r2score = r2_score(y, self.predict(X))
        return r2score


X1, X2 = np.linspace(0, 10, 200), np.random.randint(0, 10, 200)
X = [[X1[i], X2[i]] for i in range(len(X1))]
y = np.linspace(0, 5, 200)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], y[i])

model = ManualLinearRegression()
model.fit(X, y)

X_mesh, Y_mesh = np.meshgrid(X, y)
params = model.param_vector
Z = params[0] + params[1]*X_mesh + params[2]*Y_mesh

ax.plot_surface(X_mesh, Y_mesh, Z)
ax.set(xlabel='x1', ylabel='x2', zlabel='y', title='Multiple Linear Regression')

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model_train = ManualLinearRegression()
model_train.fit(X_train, y_train)
print(model_train.score(X_test, y_test))


