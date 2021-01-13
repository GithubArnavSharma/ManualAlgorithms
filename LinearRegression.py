import numpy as np

#Function for getting r(important for calculating the slope of regression line)
def get_r(x_array, y_array):
	numerator = 0
	dem1 = 0
	dem2 = 0
	for i in range(len(x_array)):
		numerator += (x_array[i] - np.mean(x_array)) * (y_array[i] - np.mean(y_array))
		dem1 += (x_array[i] - np.mean(x_array))**2
		dem2 += (y_array[i] - np.mean(y_array))**2
	denom = np.sqrt(dem1 * dem2)
	return numerator / denom

#Function for calulcating the slope of the regression line
def calculate_slope(x_array, y_array):
	return get_r(x_array, y_array) * (np.std(y_array)/np.std(x_array))

#Calculating the y_intercept of the regression line
def y_intercept(x_array, y_array):
	slope = calculate_slope(x_array, y_array)
	return np.mean(y_array) - slope * np.mean(x_array)

#Using the y=mx+b equation and the other functions I programmed to predict y based on x
def predict_y(x_array, y_array, theX):
	slope = calculate_slope(x_array, y_array)
	y_inter = y_intercept(x_array, y_array)
	return slope*theX + y_inter

#Create the Linear Regression class using the predict_y function. Class inspired based on sklearn's LinearRegression function
class ManualLinearRegression:
	def __init__(self, xArray = None, yArray = None):
		self.xArray = xArray
		self.yArray = yArray
	def fit(x_array, y_array):
		self.xArray = x_array
		self.yArray = y_array
	def predict(theNum):
		return predict_y(self.xArray, self.yArray, theNum)
  
#Test out manually made Linear Regression
x_array = [17,13,12,15,16,14,16,16,18,19]
y_array = [94,73,59,80,93,85,66,79,77,91]

lr = ManualLinearRegression()
lr.fit(x_array, y_array)
print(lr.predict(21))
