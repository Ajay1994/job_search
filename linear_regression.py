"""
Linear Regression:
f_x = dot(X, weight)
error = (y - y_hat) 
cost = mse(error) + lambda * l2-norm(weight)
gradient = (1/n) * -1 * np.dot(X.T, error) + 2 * lambda * weight
"""
import numpy as np
import random


def forward(X, W):
    return np.matmul(X, W)

def loss(Y, y_hat, W, Lambda):
    error = (Y - y_hat)
    return error, np.square(error).mean() + Lambda * np.linalg.norm(W, 2)

def gradient(error, W):
    return (-2/n) * np.matmul(X.T, error) + 2 * Lambda * W


def fit_regression(W, X, Y, alpha, Lambda, max_iter):
    for iter in range(max_iter):
        y_hat = forward(X, W)
        _error, _loss = loss(Y, y_hat, W, Lambda)

        grad = gradient(_error, W)
        W = W - alpha * grad
        if iter % 100 == 0: print(_loss)


n, d, p = 1000, 8, 1
alpha, Lambda, max_iter = 0.0001, 0.01, 1000
X = np.random.random([n, d])
W = np.random.random([d, p])
Y = np.random.random([n, p])

# Incorporate Bias term: concatenate X by a 1 column, and increase the number of rows in W by one
X = np.concatenate((X, np.ones((n, 1))), axis=1)
W = np.concatenate((W, np.random.random((1, p))), axis=0)

fit_regression(W, X, Y, alpha, Lambda, max_iter)




# class LinearRegression():
#     def __init__(self):
#         self.data_n = 10000
#         self.n_iters = 1000
#         self.loss = []

#         noise = np.random.normal(0, 1, self.data_n)
#         x1 = np.random.randn(self.data_n, 1)
#         x2 = np.random.randn(self.data_n, 1)
#         x3 = np.random.randn(self.data_n, 1)

#         self.Y = 100 * x1 + 50 * x2 + 25 * x3 + 10 + noise
#         self.X = np.concatenate([x1, x2, x3], axis = 1)

#         self.weight = np.ones(self.X.shape[1], 1)
#         self.bias = 0

#     def fit(self, alpha = 0.05):
#         print(f"Initialized Weights are: {self.weight}")
#         for _ in range(self.n_iters):

#             #Generate prediction
#             y_hat = np.dot(self.X, self.weight) + self.bias

#             #Calcuate error
#             error = y_hat - self.Y

#             #Calculate MSE
#             mse = np.square(error).mean()
#             self.loss.append(mse)

#             #Calculate gradients


        

