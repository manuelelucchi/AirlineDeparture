from model import Model
from preprocess import preprocess

train_data, train_labels, test_data, test_labels = preprocess()

model = Model(batch_size=20, learning_rate=0.1)

model.train(train_data.to_numpy(), train_labels.to_numpy(), iterations=10)

predictions = []

for test_sample, test_label in zip(test_data, test_labels):
    res = model.eval(test_data.iloc[0].to_numpy())
    predictions.append(res == res == test_label)

print("{} correct predictions in {} total".format(
    len(list(filter(lambda x: x == True, predictions))), len(predictions)))
