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

if sys.argv[2] == "custom":
    model = Model(batch_size=train_data.shape[0], learning_rate=1)
    model.train(train_data.to_numpy(),
                train_labels.to_numpy(), iterations=100)
else:
    model = LogisticRegression()
    model.fit(train_data.to_numpy(), train_labels.to_numpy())

if sys.argv[2] == "custom":
    res = model.forward(test_data.values)
    loss = binary_cross_entropy(res, test_labels.values)
else:
    res = list(map(lambda x: x[1], model.predict_proba(
        test_data.values)))
    loss = log_loss(test_labels.values, res)


predictions = zip(res, test_labels.values)
