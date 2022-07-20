import numpy as np
from numpy import ndarray


def sigmoid(x: ndarray) -> ndarray:
    '''
    Calculates the sigmoid of the given data
    '''
    g = 1.0 / (1.0 + np.exp(-x))
    return g


def binary_cross_entropy(y: ndarray, y_label: ndarray):
    '''
    Calculates the binary cross entropy loss of the calculated y and the given y_label
    '''
    loss = -np.mean(y_label*(np.log(y)) - (1-y_label)*np.log(1-y))
    return loss


def normalize(X: ndarray) -> ndarray:
    '''
    Normalizes the data based on the number of features
    '''
    # _ -> number of training examples
    # n -> number of features
    _, n = X.shape

    for _ in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)

    return X


def gradients(X: ndarray, y: ndarray, y_label: ndarray):
    '''
    Calculates the gradient w.r.t weights and bias
    '''

    # m-> number of training examples.
    m = X.shape[0]

    # Gradient of loss w.r.t weights.
    dw = (1/m)*np.dot(X.T, (y - y_label))

    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y - y_label))

    return dw, db
