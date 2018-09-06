import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


def predict(row, weights):
  activation = row.T.dot(weights)

  return 1.0 if activation >= 0.0 else 0.0


def train_weights(x, y, learning_rate, epochs):
  weights = np.zeros(x.shape[1])
  total_err = 0.0

  epochs_iterable = tqdm(range(epochs))

  for epoch in epochs_iterable:
    for i, row in enumerate(x):
      target = y[i]
      pred = predict(row, weights)
      err = target - pred
      total_err += err**2
      weights = weights + (learning_rate * err * row)

  return weights


def sample_train():
  dataset = np.array([[1,2.7810836,2.550537003,0],
    [1,1.465489372,2.362125076,0],
    [1,3.396561688,4.400293529,0],
    [1,1.38807019,1.850220317,0],
    [1,3.06407232,3.005305973,0],
    [1,7.627531214,2.759262235,1],
    [1,5.332441248,2.088626775,1],
    [1,6.922596716,1.77106367,1],
    [1,8.675418651,-0.242068655,1],
    [1,7.673756466,3.508563011,1]])

  x = dataset[:,:-1]
  y = dataset[:,-1]

  weights = np.array([-0.1, 0.20653640140000007, -0.23418117710000003])

  print('weights: ', weights)

  weights = train_weights(x, y, 0.1, 5)

  print('weights: ', weights)

  for i, row in enumerate(x):
    prediction = predict(row, weights)
    print('Expected=%d, Predicted=%d' % (y[i], prediction))


def main():
  sample_train()

  df = pd.read_csv('data/sonar.csv', header=None)
  df.columns = [*df.columns[0:-1], 'y']
  df['bias'] = 1

  y = (df['y'] == 'M').astype(float).values
  x = df[[*list(range(60)), 'bias']].values

  weights = train_weights(x, y, 0.01, 10000)

  preds = []

  for row in x:
    preds.append(predict(row, weights))

  preds = np.array(preds)

  print('MSE: ', mean_squared_error(y, preds))
  print('Accuracy Score: ', accuracy_score(y, preds))

  # import pdb; pdb.set_trace()


if __name__ == "__main__":
  main()
