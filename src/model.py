import numpy as np
from numpy import ndarray
from constants import path
from functions import binary_cross_entropy, gradients, normalize, sigmoid

# =======================================================================================


class Model():
    def __init__(self, learning_rate: float, batch_size: int, l2: float):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2 = l2

    def initialize(self, columns_number):
        self.W = np.random.rand(columns_number)
        self.b = np.random.rand()

    def evaluate(self, X: ndarray) -> ndarray:
        Z = np.dot(X, self.W) + self.b
        Z = sigmoid(Z)
        return Z

    def gradient(self, X: ndarray, Y: ndarray, Y_label: ndarray):
        return gradients(X, Y, Y_label, self.W, self.l2)

    def update(self, dW: ndarray, db: float):
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def train(self, X: ndarray, Y_labels: ndarray, iterations: int = 10):
        self.initialize(X.shape[1])
        X = normalize(X)
        losses = []

        for _ in range(iterations):
            for b in range(X.shape[0]//self.batch_size):
                b_X = X[b*self.batch_size:b*self.batch_size+self.batch_size, :]
                b_Y_labels = Y_labels[b*self.batch_size:b *
                                      self.batch_size+self.batch_size]
                Y = self.evaluate(b_X)
                losses.append(binary_cross_entropy(
                    Y, b_Y_labels, self.W, self.l2))
                (dW, db) = self.gradient(b_X, Y, b_Y_labels)
                self.update(dW, db)

        return losses

# ==============================================================================================
