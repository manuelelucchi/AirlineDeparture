from model import Model
from preprocess import preprocess

train_data, train_labels, test_data, test_labels = preprocess()

print('Train Data {}, Train Labels {}, Test Data {}, Test Labels {}'.format(
    train_data.shape, train_labels.shape, test_data.shape, len(test_labels)))

model = Model(batch_size=1000, learning_rate=0.01)

model.train(train_data.to_numpy(), train_labels.to_numpy(), iterations=10)

predictions = []

for test_sample, test_label in zip(test_data.values, test_labels.values):
    res = model.eval(test_sample)
    predictions.append([test_label, res == test_label])

print("{} correct predictions in {} total".format(
    len(list(filter(lambda x: x[1] == True, predictions))), len(predictions)))

print("{} flights were canceled, {} flights were predicted canceled".format(
    len(list(filter(lambda x: x[0] == True, predictions))), 0))
