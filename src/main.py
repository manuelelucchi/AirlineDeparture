import sys
from functions import binary_cross_entropy
from model import Model
from preprocess import preprocess
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

train_data, train_labels, test_data, test_labels = preprocess()

# print('Train Data {}, Train Labels {}, Test Data {}, Test Labels {}'.format(
#    train_data.shape, train_labels.shape, test_data.shape, len(test_labels)))

if sys.argv[2] == "custom":
    model = Model(batch_size=train_data.size, learning_rate=1)
    model.train(train_data,
                train_labels, iterations=100)
else:
    model = LogisticRegression()
    model.fit(train_data, train_labels)

if sys.argv[2] == "custom":
    res = model.forward(test_data)
    loss = binary_cross_entropy(res, test_labels)
else:
    res = list(map(lambda x: x[1], model.predict_proba(
        test_data.values)))
    loss = log_loss(test_labels, res)


predictions = zip(res, test_labels)
