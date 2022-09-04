from imp import init_builtin
from random import random
from turtle import forward
import numpy as np
from numpy import ndarray
from constants import path
from functions import binary_cross_entropy, gradients, normalize, sigmoid


class Model():
    def __init__(self, learning_rate: float, l2: float):
        self.learning_rate = learning_rate
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
        #X = normalize(X)
        losses = []
        for _ in range(iterations):
            Y = self.evaluate(X)
            losses.append(binary_cross_entropy(Y, Y_labels, self.W, self.l2))
            (dW, db) = self.gradient(X, Y, Y_labels)
            self.update(dW, db)
        return losses
