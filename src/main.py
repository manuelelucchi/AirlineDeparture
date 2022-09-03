import sys
from functions import binary_cross_entropy
from model import Model
from preprocess import preprocess
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

lr_1 = 0.01
lr_2 = 0.001
lr_3 = 0.0001
lr_4 = 0.00001

l2_1 = 0.01

it_1 = 100
it_2 = 500
it_3 = 1000


def experiments(forIndex: str):
    print("Loading data for experiment: {}".format(forIndex))
    train_data, train_labels, test_data, test_labels = preprocess(forIndex)
    print("Done")

    def custom_train_eval(lr=1, l2=0.01, iterations=100) -> float:
        model = Model(batch_size=train_data.shape[1], learning_rate=lr, l2=l2)
        train_losses = model.train(train_data,
                                   train_labels, iterations=iterations)
        print("For Custom, LR: {}, L2: {}, IT: {}, the last train loss is: {}".format(
            lr, l2, iterations, train_losses[-1]))
        res = model.evaluate(test_data)
        test_loss = binary_cross_entropy(res, test_labels)
        print("For Custom, LR: {}, L2: {}, IT: {}, the average test loss is: {}".format(
            lr, l2, iterations, test_loss))

    def sklearn_train_eval() -> float:
        model = LogisticRegression()
        model.fit(train_data, train_labels)
        res = list(map(lambda x: x[1], model.predict_proba(
            test_data)))
        loss = log_loss(test_labels, res)
        print("For Sklearn, IT: {}, the average loss is: {}".format(100, loss))

    custom_train_eval(lr=lr_1, iterations=it_1)
    custom_train_eval(lr=lr_2, iterations=it_1)
    custom_train_eval(lr=lr_3, iterations=it_1)
    custom_train_eval(lr=lr_4, iterations=it_1)
    custom_train_eval(lr=lr_3, iterations=it_2)
    custom_train_eval(lr=lr_3, iterations=it_3)
    sklearn_train_eval()


experiments(forIndex='CANCELLED')
experiments(forIndex='DIVERTED')
