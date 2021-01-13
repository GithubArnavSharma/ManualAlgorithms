#Import numpy for the Manual KNeighbors
import numpy as np
#Import the iris dataset and train_test_split for testing of model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#Function that can find the distance between two vectors 
def distance_vector(vector1, vector2):
    distance_n = []
    for n in range(len(vector1)):
        distance_n.append((vector1[n]-vector2[n])**2)
    return np.sqrt(sum(distance_n))

#The KNeighbors Class
class ManualKNeighbors:
    #Define the variables needed(X, y, and n_neighbors)
    def __init__(self, n, x=None, y=None):
        self.x = x
        self.y = y
        self.n = n
        
    #The fit function will just be defining self.x and self.y
    def fit(self, the_x, the_y):
        self.x = the_x
        self.y = the_y

    #The function for predicting y based on x
    def predict(self, x_pred):
        #Find the distance between the pred vector and the data x 
        all_distances = [distance_vector(x_pred, self.x[i]) for i in range(len(self.x))]
        #Get the n_neighbors closest vectors from self.x
        the_mins = []
        for i in range(self.n):
            min_index = np.argmin(all_distances)
            the_mins.append(self.y[min_index])
            all_distances.pop(min_index)

        #Get the most popular class in the the_mins list
        amount = 0
        the_value = 0
        for i in range(len(the_mins)):
            if the_mins.count(the_mins[i]) > amount:
                amount = the_mins.count(the_mins[i])
                the_value = the_mins[i]
        
        #Return the most popular class
        return the_value

    #Function that can predict multiple x_preds
    def predict_multiple(self, x_preds):
        return [self.predict(x_vector) for x_vector in x_preds]

    #Function that can determine the accuracy of the model based on how much it got correct
    def accuracy(self, x_test, y_test):
        predictions = [self.predict(test_vector) for test_vector in x_test]
        amount_correct = [predictions[i] == y_test[i] for i in range(len(predictions))].count(True)
        return amount_correct/len(predictions)

#Load the data
data = load_iris()
X = data.data
y = data.target
#Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Test the model accuracy on X_test and y_test based on training it from X_train and y_train
model = ManualKNeighbors(1)
model.fit(X_train, y_train)
print(model.accuracy([X_test[0]], [y_test[0]])) #Accuracy is around 97%
