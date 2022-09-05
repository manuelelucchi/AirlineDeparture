import sys
from functions import binary_cross_entropy
from model import Model
from preprocess import preprocess
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from numpy import zeros

# =============================================================================

file = open('data/results.txt', 'a')


def print_and_save(s: str):
    file.write(s + '\n')
    print(s)


def custom_train_eval(iterations=100, lr=1, batch_size=20, l2=0.01) -> float:
    model = Model(learning_rate=lr, batch_size=batch_size, l2=l2)
    train_losses = model.train(train_data,
                               train_labels, iterations=iterations)
    res = model.evaluate(test_data)
    test_loss = binary_cross_entropy(
        res, test_labels, zeros([res.shape[0]]), 0)

    print_and_save("For Custom, LR: {}, Batch Size: {}, L2: {}, IT: {}".format(
        lr, batch_size, l2, iterations))
    print_and_save("The last train loss is: {}".format(train_losses[-1]))
    print_and_save("The average test loss is: {}".format(test_loss))
    print_and_save("=====================================================")


def sklearn_train_eval() -> float:
    model = LogisticRegression()
    model.fit(train_data, train_labels)
    res = list(map(lambda x: x[1], model.predict_proba(
        test_data)))
    loss = log_loss(test_labels, res)
    print_and_save(
        "For Sklearn, IT: {}, the average test loss is: {}".format(100, loss))
    print_and_save("=====================================================")


# =============================================================================

train_data, train_labels, test_data, test_labels = preprocess(
    "DIVERTED", 6000000, 10000, usePyspark=False)

# =============================================================================

# Learning Rate

custom_train_eval(iterations=100, lr=0.1,
                  batch_size=train_data.shape[0], l2=0)

custom_train_eval(iterations=100, lr=0.01,
                  batch_size=train_data.shape[0], l2=0)

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=train_data.shape[0], l2=0)

custom_train_eval(iterations=100, lr=0.0001,
                  batch_size=train_data.shape[0], l2=0)

# =============================================================================

# Batch Size

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=1, l2=0)

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=20, l2=0)

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=1000, l2=0)

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=train_data.shape[0], l2=0)

# =============================================================================

# L2 Regularization

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=20, l2=0)

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=20, l2=0.1)

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=20, l2=0.01)

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=20, l2=0.001)


# =============================================================================

# L2 Regularization

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=20, l2=0.01)

custom_train_eval(iterations=200, lr=0.001,
                  batch_size=20, l2=0.01)

custom_train_eval(iterations=1000, lr=0.001,
                  batch_size=20, l2=0.01)

# =============================================================================

sklearn_train_eval()
