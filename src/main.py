from functions import binary_cross_entropy
from model import Model
from preprocess import preprocess
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from numpy import zeros
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
    res = model.evaluate(test_data)
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
    fig.savefig("./data/IT={}_LR={}_BatchSize={}_L2={}.png".format(iterations,
                                                                   lr, batch_size, l2))
    fig.clear()
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
    "DIVERTED", False, 6000000, 10000, usePyspark=False)

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

"""
labelsAndScores = OHEValidationData.map(lambda lp:
                                            (lp.label, getP(lp.features,
                                                            model0.weights,
                                                            model0.intercept)))
labelsAndWeights = labelsAndScores.collect()
labelsAndWeights.sort(key=lambda kv: kv[1], reverse=True)
labelsByWeight = np.array([k for (k, v) in labelsAndWeights])

length = labelsByWeight.size
truePositives = labelsByWeight.cumsum()
numPositive = truePositives[-1]
falsePositives = np.arange(1.0, length + 1, 1.) - truePositives

truePositiveRate = truePositives / numPositive
falsePositiveRate = falsePositives / (length - numPositive)

# Generate layout and plot data
fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(falsePositiveRate, truePositiveRate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
plt.show()
"""
