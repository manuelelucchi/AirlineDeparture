from functions import binary_cross_entropy, normalize
from model import Model
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from numpy import zeros
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================

file = open('data/results.txt', 'a')


def print_and_save(s: str):
    file.write(s + '\n')
    print(s)


def custom_train_eval(iterations=100, lr=1, batch_size=20, l2=0.01) -> float:
    model = Model(learning_rate=lr, batch_size=batch_size, l2=l2)
    (train_losses, gradients) = model.train(train_data,
                                            train_labels, iterations=iterations)
    res = model.evaluate(normalize(test_data))
    test_loss = binary_cross_entropy(
        res, test_labels, zeros([res.shape[0]]), 0)
    print_and_save("For Custom, LR: {}, Batch Size: {}, L2: {}, IT: {}".format(
        lr, batch_size, l2, iterations))
    print_and_save("The last train loss is: {}".format(train_losses[-1]))
    print_and_save("The average test loss is: {}".format(test_loss))
    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss/Gradient')
    ax.set_title('IT={} LR={} Batch Size={} L2={}'.format(iterations,
                                                          lr, batch_size, l2))
    ax.plot(range(iterations), train_losses, label='Loss')
    ax.plot(range(iterations), gradients, label='Gradient')
    ax.grid()
    ax.legend()
    name = "IT={}_LR={}_BatchSize={}_L2={}".format(
        iterations, lr, batch_size, l2)
    fig.savefig("./data/{}.png".format(name))
    fig.clear()
    plt.close()

    make_roc(test_labels, res, name)

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

# ============================================================================


# If true, the balancing will be done before resulting in a great performances gain
earlyBalance = False
problem_to_solve = 'CANCELLED'  # The alternative is 'DIVERTED'
usePyspark = False  # If true, uses PySpark, otherwise Pandas
# If false, only #records_per_file records will be sampled from the most recent year csv
sample_from_all_files = True
records_per_file = 500000
# Warning, this number should be smaller or equal than the number of positive cases in the sampled dataset.
records_for_balancing = 10000
# Since the sampling is random, the number could vary, thus we don't recomend to use values >10000

# =============================================================================
train_data, train_labels, test_data, test_labels = preprocess(
    problem_to_solve, sample_from_all_files, records_per_file, records_for_balancing, usePyspark=usePyspark, earlyBalance=earlyBalance)

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

# Iterations

custom_train_eval(iterations=100, lr=0.001,
                  batch_size=20, l2=0.01)

custom_train_eval(iterations=200, lr=0.001,
                  batch_size=20, l2=0.01)

custom_train_eval(iterations=1000, lr=0.001,
                  batch_size=20, l2=0.01)

# =============================================================================

sklearn_train_eval()

# =============================================================================
