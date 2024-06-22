from __future__ import print_function, division
import numpy as np
import math
from mlfromscratch.utils import normalize, polynomial_features

# L1 Regularization class (for Lasso Regression)
class L1Regularization:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, 1)

    def grad(self, w):
        return self.alpha * np.sign(w)

# L2 Regularization class (for Ridge Regression)
class L2Regularization:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 * np.dot(w, w)

    def grad(self, w):
        return self.alpha * w

# Combined L1 and L2 Regularization class (for ElasticNet)
class L1L2Regularization:
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1 = self.l1_ratio * np.linalg.norm(w, 1)
        l2 = (1 - self.l1_ratio) * 0.5 * np.dot(w, w)
        return self.alpha * (l1 + l2)

    def grad(self, w):
        l1 = self.l1_ratio * np.sign(w)
        l2 = (1 - self.l1_ratio) * w
        return self.alpha * (l1 + l2)

# Base regression model class
class Regression:
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, n_features)
        self.bias = 0

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(X.shape[1])

        for i in range(self.n_iterations):
            y_pred = X.dot(self.weights)
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.weights))
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.weights)
            self.weights -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.weights)

# Linear Regression class
class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super().__init__(n_iterations, learning_rate)
    
    def fit(self, X, y):
        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)
            self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        else:
            super().fit(X, y)

# Lasso Regression class
class LassoRegression(Regression):
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = L1Regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X)

# Ridge Regression class
class RidgeRegression(Regression):
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        self.regularization = L2Regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)

# Polynomial Regression class
class PolynomialRegression(Regression):
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        self.degree = degree
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super().fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super().predict(X)

# Polynomial Ridge Regression class
class PolynomialRidgeRegression(Regression):
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = L2Regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X)

# ElasticNet Regression class
class ElasticNet(Regression):
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = L1L2Regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X)
