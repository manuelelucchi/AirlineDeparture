import math
from random import random
from turtle import forward
import numpy as np
from numpy import ndarray
from constants import columns_number
from functions import binary_cross_entropy, gradients, sigmoid


class Model():
    def __init__(self, batch_size=20, learning_rate: float = 0.1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.W = np.random.rand(columns_number, batch_size)
        self.b = np.random.rand()

    def forward(self, X: ndarray):
        Z = np.dot(X, self.W) + self.b
        Z = sigmoid(Z)
        return Z

    def backward(self, X: ndarray, Y: ndarray, Y_hat: ndarray):
        return gradients(X, Y, Y_hat)

    def update(self, dW: ndarray, db: float):
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def train(self, X: ndarray, Y_hat: ndarray, iterations: int = 10):
        for i in range(iterations):
            for b in range(X.shape[0]):
                b_X = 0
                b_Y_hat = 0
                Y = self.forward(b_X)
                l = binary_cross_entropy(Y, b_Y_hat)
                print("Loss {}|Iteration {}|Batch {}", l, i, b)
                (dW, db) = self.backward(b_X, b_Y_hat)
                self.update(dW, db)

    def save(self):
        pass

    def eval(self, X: ndarray) -> int:
        return round(forward(self, X))
