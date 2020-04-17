import numpy as np

def compute_cost(X, y, theta):
    m = X.shape[0]
    hypo = np.dot(X, theta)
    return np.sum(np.power(np.subtract(hypo, y), 2)) / 2 / m

def gradient_descent(X, y, theta, alpha, iter_count):
    m = X.shape[0]
    theta_count = theta.shape[0]
    # print(X[:, 0].reshape(X.shape[0], 1))
    for i in range(iter_count):
        hypo = np.dot(X, theta)
        temp_theta = theta
        for j in range(theta_count):
            cond = np.multiply(np.subtract(hypo, y), X[:, j].reshape(X.shape[0], 1))
            temp_theta[j] = theta[j] - (alpha / m) * np.sum(cond)
        theta = temp_theta
    return theta