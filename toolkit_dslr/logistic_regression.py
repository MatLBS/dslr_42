import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        """Cross-entropy loss"""
        m = len(y)
        return - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def fit(self, X, y):
        """Train model using gradient descent"""
        m, n = X.shape
        self.weights = np.zeros(n)

        for _ in range(self.iterations):
            h = self.sigmoid(np.dot(X, self.weights) + self.bias)

            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.cost_history.append(self.cost(h, y))

        return self.weights, self.bias

    def predict(self, X):
        """Make predictions"""
        return (self.sigmoid(np.dot(X, self.weights)
                             + self.bias) >= 0.5).astype(int)

    def predict_arguments(self, X, weights, bias):
        """Make predictions"""
        return self.sigmoid(np.dot(X, weights) + bias)

    def sgd(self, X, y, batch_size=1):
        m, n = X.shape
        self.weights = np.zeros(n + 1) 
        X_bias = np.c_[np.ones((m, 1)), X]
        self.cost_history = []

        for epoch in range(self.iterations):
            indices = np.random.permutation(m)
            X_shuffled = X_bias[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                h_batch = self.sigmoid(np.dot(X_batch, self.weights))
                dw = (1/batch_size) * np.dot(X_batch.T, (h_batch - y_batch))

                self.weights -= self.lr * dw

            h_all = self.sigmoid(np.dot(X_bias, self.weights))
            cost = self.cost(h_all, y)
            self.cost_history.append(cost)