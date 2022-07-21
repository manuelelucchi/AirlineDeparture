from random import random
from turtle import forward
import numpy as np
from numpy import ndarray
from constants import columns_number
from constants import path
from functions import binary_cross_entropy, gradients, sigmoid


class Model():
    def __init__(self, batch_size=20, learning_rate: float = 0.01):
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.W = np.random.rand(columns_number)
        self.b = np.random.rand()

    def forward(self, X: ndarray):
        Z = np.dot(X, self.W) + self.b
        Z = sigmoid(Z)
        return Z

    def backward(self, X: ndarray, Y: ndarray, Y_label: ndarray):
        return gradients(X, Y, Y_label)

    def update(self, dW: ndarray, db: float):
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        Model.load(self)

    def train(self, X: ndarray, Y_label: ndarray, iterations: int = 10):
        for i in range(iterations):
            for b in range(X.shape[0]//self.batch_size):
                b_X = X[b*self.batch_size:b*self.batch_size+self.batch_size, :]
                b_Y_label = Y_label[b*self.batch_size:b *
                                    self.batch_size+self.batch_size]
                Y = self.forward(b_X)
                (dW, db) = self.backward(b_X, Y, b_Y_label)
                self.update(dW, db)
                if b % 10000 == 0:
                    l = binary_cross_entropy(Y, b_Y_label)
                    print("Loss {}|Iteration {}|Batch {}".format(l, i, b))

            print("Iteration {}".format(i))

    def save(self):
        with open(path + '/model.txt', 'w') as f:
            for n in self.W:
                f.write(str(n) + '\n')
            f.write(str(self.b) + '\n')

    def load(self):
        with open(path + '/model.txt', 'r') as f:
            counter:int = 0
            lines = f.readlines()
            for l in lines:
                if counter == columns_number:
                    self.b = float(l)
                else:
                    self.W[counter] == float(l)
                    counter += 1

    def eval(self, X: ndarray) -> int:
        return round(self.forward(X))
