import sys
from functions import binary_cross_entropy
from model import Model
from preprocess import preprocess
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

train_data, train_labels, test_data, test_labels = preprocess()

print('Train Data {}, Train Labels {}, Test Data {}, Test Labels {}'.format(
    train_data.shape, train_labels.shape, test_data.shape, len(test_labels)))

n_features = train_data.shape[0]


def custom_train_eval(lr=1, l2=0.01, iterations=100) -> float:
    model = Model(batch_size=train_data.shape[1], learning_rate=lr, l2=l2)
    model.train(train_data.to_numpy(),
                train_labels.to_numpy(), iterations=iterations)
    res = model.forward(test_data.values)
    return binary_cross_entropy(res, test_labels.values)


def sklearn_train_eval() -> float:
    model = LogisticRegression()
    model.fit(train_data.to_numpy(), train_labels.to_numpy())
    res = list(map(lambda x: x[1], model.predict_proba(
        test_data.values)))
    return log_loss(test_labels.values, res)


def experiment(n, func):
    res: float = func()
    print("Experiment {} completed with result: {}".format(n, res))


lr1 = 0.01
lr2 = 0.001
lr3 = 0.0001
lr4 = 0.00001
it1 = 100
it2 = 500
it3 = 1000

experiment(1, lambda: custom_train_eval(lr=lr1, iterations=it1))
experiment(2, lambda: custom_train_eval(lr=lr2, iterations=it1))
experiment(3, lambda: custom_train_eval(lr=lr3, iterations=it1))
experiment(4, lambda: custom_train_eval(lr=lr4, iterations=it1))
experiment(5, lambda: custom_train_eval(lr=lr3, iterations=it2))
experiment(6, lambda: custom_train_eval(lr=lr3, iterations=it3))
experiment("Sklearn", lambda: sklearn_train_eval())
