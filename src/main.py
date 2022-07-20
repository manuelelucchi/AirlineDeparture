from model import Model
from preprocess import preprocess

train_data, train_labels, test_data, test_labels = preprocess()

model = Model(batch_size=20, learning_rate=0.1)

model.train(train_data.to_numpy(), train_labels.to_numpy(), iterations=10)

res = model.eval(test_data[0].to_numpy())

if res == test_labels[0]:
  print("Correct prediction")
else:
  print("Wrong prediction")
