import numpy as np


def sigmoid(x: float) -> float:
    '''
    Calculates the sigmoid of the given data
    '''
    g = 1.0 / (1.0 + np.exp(-x))
    return g


def binary_cross_entropy(y: float, y_hat: float) -> float:
    '''
    Calculates the binary cross entropy loss of the calculated y and the given y_hat
    '''
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss


def normalize(X: np.matrix) -> np.matrix:
    '''
    Normalizes the data based on the number of features
    '''
    # _ -> number of training examples
    # n -> number of features
    _, n = X.shape

    for _ in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)

    return X


def gradients(X, y, y_hat):
    '''
    Calculates the gradient w.r.t weights and bias
    '''

    # m-> number of training examples.
    m = X.shape[0]

    # Gradient of loss w.r.t weights.
    dw = (1/m)*np.dot(X.T, (y_hat - y))

    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y_hat - y))

    return dw, db
