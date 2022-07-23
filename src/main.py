from turtle import forward

from numpy import positive
from model import Model
from preprocess import preprocess
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression

train_data, train_labels, test_data, test_labels = preprocess()

print('Train Data {}, Train Labels {}, Test Data {}, Test Labels {}'.format(
    train_data.shape, train_labels.shape, test_data.shape, len(test_labels)))

if True:
    model = Model(batch_size=20, learning_rate=0.01)
    model.train(train_data.to_numpy(), train_labels.to_numpy(), iterations=100)
else:
    model = LogisticRegression()

    model.fit(train_data.to_numpy(), train_labels.to_numpy())

predictions = []

for test_sample, test_label in zip(test_data.values, test_labels.values):
    if True:
        res = model.forward(test_sample)
    else:
        res = model.predict(test_sample.reshape([1, 8]))
    predictions.append([res, test_label])

correct = len(list(filter(lambda x: round(x[1]) == x[0], predictions)))
canceled = len(list(filter(lambda x: round(x[1]) == 1, predictions)))


print("{} correct predictions in {} total: {}%".format(
    correct, len(predictions), correct/len(predictions)*100))

print("There were {} real canceled flights".format(canceled))
