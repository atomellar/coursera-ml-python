import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warm_up, c_math as math_util


print("=============================================")
print("Coursera Machine Learning - Python Version")
print("EXERCISE 1 - Linear Regression")
print("=============================================")

print("Warming Excercise")
identity_matrix = warm_up.create_identity_matrix(5)
print("Identity matrix 5x5 is \n {}".format(identity_matrix))
print("===================")

print("Load data")
data = pd.read_csv("./ex1data1.txt", delimiter=',', header=None)
X = data.iloc[:, 0].to_numpy()
X = X.reshape(X.shape[0], 1)
y = data.iloc[:, 1].to_numpy()
y = y.reshape(y.shape[0], 1)

print("Plot Data")
plt.scatter(X, y, c='r', marker='x', label='Data')
plt.xlabel('Population in 10,000s')
plt.ylabel('Profit in $10,000s')
print("===================")

print("Cost Function")
ones = np.ones_like(X)
X = np.concatenate((ones, X), axis=1)
theta = np.zeros((2, 1))

test_theta = np.array([-1, 2])
test_theta = test_theta.reshape(test_theta.shape[0], 1)
test_cost = math_util.compute_cost(X, y, test_theta)
print("Compute Cost for theta [-1; 2] = {}".format(test_cost))
print("Expected Compute Cost for theta [-1; 2] ~54.24")
print("===================")

print("Gradient Descent")
iter_count = 1500
alpha = 0.01

theta, theta_history = math_util.gradient_descent(X, y, theta, alpha, iter_count)
print("Theta after {} loops, alpha {} is {}".format(iter_count, alpha, theta))
print("Expected theta ~[-3.6303; 1.1664]")
print("===================")

# Predict values for population sizes of 35,000 and 70,000
# print(np.array([1, 35000]).reshape(1, 2))
# print(theta)
predict35 = np.array([1, 35000]).reshape(1, 2).dot(theta)
print("35,000 has price {}".format(predict35))
predict70 = np.array([1, 70000]).reshape(1, 2).dot(theta)
print("70,000 has price {}".format(predict70))

print("===================")
print(theta_history[0:5, :])
print(theta_history.shape)
# Plot linear regression
# print(theta_history)
# plt.plot(X[:, [1]], np.dot(X[:], theta), '-', label='Linear Regression')
# plt.legend()
# plt.show()