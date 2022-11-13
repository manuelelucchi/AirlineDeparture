from functions import binary_cross_entropy, normalize
from logistic_regression import LogisticRegression
from preprocess import preprocess
import read
from pyspark.sql import SparkSession
import sklearn.linear_model as sk
from sklearn.metrics import log_loss
from numpy import zeros
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================


def print_and_save(s: str):
    file = open('data/results.txt', 'a')
    file.write(s + '\n')
    print(s)


def logistic_train_eval(iterations=100, lr=1, batch_size=20, l2=0.01) -> float:
    model = LogisticRegression(learning_rate=lr, batch_size=batch_size, l2=l2)
    (train_losses, gradients) = model.train(train_data,
                                            train_labels, iterations=iterations)
    res = model.evaluate(normalize(test_data))
    test_loss = binary_cross_entropy(
        res, test_labels, zeros([res.shape[0]]), 0)
    print_and_save("For Custom, LR: {}, Batch Size: {}, L2: {}, IT: {}".format(
        lr, batch_size, l2, iterations))
    print_and_save("The last train loss is: {}".format(train_losses[-1]))
    print_and_save("The average test loss is: {}".format(test_loss))

    name = "IT={}_LR={}_BatchSize={}_L2={}".format(
        iterations, lr, batch_size, l2)

    plot_loss_gradient(iterations, train_losses, gradients, name)

    make_roc(test_labels, res, name)

    print_and_save("=====================================================")


def sklearn_linear_train_eval() -> float:
    model = sk.LinearRegression()
    model.fit(train_data, train_labels)
    res = list(model.predict(
        test_data))
    loss = log_loss(test_labels, res)
    print_and_save(
        "For Sklearn Linear, IT: {}, the average test loss is: {}".format(100, loss))
    print_and_save("=====================================================")


def sklearn_logistic_train_eval() -> float:
    model = sk.LogisticRegression()
    model.fit(train_data, train_labels)
    res = list(map(lambda x: x[1], model.predict_proba(
        test_data)))
    loss = log_loss(test_labels, res)
    print_and_save(
        "For Sklearn Logistic, IT: {}, the average test loss is: {}".format(100, loss))
    print_and_save("=====================================================")


def plot_loss_gradient(iterations, train_losses, gradients, name):
    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss/Gradient')
    ax.set_title(name)
    ax.plot(range(iterations), train_losses, label='Loss')
    ax.plot(range(iterations), gradients, label='Gradient')
    ax.grid()
    ax.legend()

    fig.savefig("./data/{}.png".format(name))
    fig.clear()
    plt.close()


def make_roc(labels, results, name):
    labels_and_results = sorted(
        list(zip(labels, map(lambda x: x, results))), key=lambda x: x[1])

    labels_by_weights = np.array([k for (k, _) in labels_and_results])

    length = labels_by_weights.size

    true_positives = labels_by_weights.cumsum()

    num_positive = true_positives[-1]

    false_positives = np.arange(1.0, length + 1, 1.) - true_positives

    true_positives_rate = true_positives / num_positive
    false_positives_rate = false_positives / (length - num_positive)

    fig, ax = plt.subplots()
    ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    plt.plot(false_positives_rate, true_positives_rate,
             color='#8cbfd0', linestyle='-', linewidth=3.)
    plt.plot((0., 1.), (0., 1.), linestyle='--',
             color='#d6ebf2', linewidth=2.)

    plt.savefig('./data/{}_roc.png'.format(name))
    fig.clear()
    plt.close()


# To summarize, there is a bias-variance trade-off associated with the choice of k in k-fold cross-validation.
# Typically, given these considerations, one performs k-fold cross-validation using k = 5 or k = 10, as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance.

# ============================================================================


# If true, the balancing will be done before resulting in a great performances gain
earlyBalance = True
problem_to_solve = 'CANCELLED'  # The alternative is 'DIVERTED'
usePyspark = False  # If true, uses PySpark, otherwise Pandas
# If false, only #records_per_file records will be sampled from the most recent year csv
sample_from_all_files = True
# Warning, this number should be smaller or equal than the number of positive cases in the sampled dataset.
records_for_balancing = 10000
# Since the sampling is random, the number could vary, thus we don't recomend to use values >10000
worker_nodes = "*"
# Tests have been performed using the value 1 and the *, which means thath Spark automatically set the number of nodes based on the enviroment characteristics

read.spark = SparkSession.builder \
    .appName("Airline Departure") \
    .master('local[' + worker_nodes + ']') \
    .getOrCreate()
# =============================================================================
data = preprocess(
    problem_to_solve, 10, sample_from_all_files, records_for_balancing, usePyspark=usePyspark)

# =============================================================================

"""

# Learning Rate

logistic_train_eval(iterations=100, lr=0.1,
                    batch_size=train_data.shape[0], l2=0)

logistic_train_eval(iterations=100, lr=0.01,
                    batch_size=train_data.shape[0], l2=0)

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=train_data.shape[0], l2=0)

logistic_train_eval(iterations=100, lr=0.0001,
                    batch_size=train_data.shape[0], l2=0)

# =============================================================================

# Batch Size

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=1, l2=0)

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=20, l2=0)

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=1000, l2=0)

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=train_data.shape[0], l2=0)

# =============================================================================

# L2 Regularization

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=20, l2=0)

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=20, l2=0.1)

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=20, l2=0.01)

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=20, l2=0.001)


# =============================================================================

# Iterations

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=20, l2=0.01)

logistic_train_eval(iterations=200, lr=0.001,
                    batch_size=20, l2=0.01)

logistic_train_eval(iterations=1000, lr=0.001,
                    batch_size=20, l2=0.01)

# =============================================================================

sklearn_train_eval()

# =============================================================================

"""

sklearn_linear_train_eval()

sklearn_logistic_train_eval()

logistic_train_eval(iterations=100, lr=0.001,
                    batch_size=20, l2=0.01)
