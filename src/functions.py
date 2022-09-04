import numpy as np
from numpy import ndarray


def sigmoid(x: ndarray) -> ndarray:
    '''
    Calculates the sigmoid of the given data
    '''
    g = 1.0 / (1.0 + np.exp(-x))
    return g


def binary_cross_entropy(y: ndarray, y_label: ndarray, w: ndarray, l2: ndarray):
    '''
    Calculates the binary cross entropy loss of the calculated y and the given y_label
    '''
    loss = -np.mean(y_label*(np.log(y)) + (1-y_label)
                    * np.log(1-y)) + regularize(w, l2)
    return loss


def regularize(W: ndarray, l2: float) -> float:
    '''
    Calculates the regularization term for the loss
    '''
    return (l2 / 2) * np.sum(np.square(W))


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


def gradients(X: ndarray, Y: ndarray, Y_label: ndarray, W: ndarray, l2: float):
    '''
    Calculates the gradient w.r.t weights and bias
    '''

    # m-> number of training examples.
    m = X.shape[0]

    # Gradient of loss w.r.t weights with regularization
    dw = (1/m)*np.dot(X.T, (Y - Y_label)) + l2 * W

    # Gradient of loss w.r.t bias with regularization
    db = (1/m)*np.sum((Y - Y_label))

    return dw, db
