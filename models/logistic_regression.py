import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

def sigmoid(scores):
  return 1 / (1 + np.exp(-scores))


def log_likelihood(x, y, b):
  scores = x.dot(b)
  first = y * scores
  second = np.log(1 + np.exp(scores))

  return np.sum(first - second)


def gradient(x, y, y_hat):
  return x.T.dot(y - y_hat) / len(y)


def get_weights(x, y, steps=300000, learning_rate=0.01, add_intercept=False):
  if add_intercept:
    intercept = np.ones((x.shape[0], 1))
    x = np.hstack((intercept, x))

  b = np.zeros(x.shape[1])

  iterable = tqdm(range(steps), 'Log likelihood: ')

  for i in iterable:
    scores = x.dot(b)
    y_hat = sigmoid(scores)
    grad = gradient(x, y, y_hat)
    b += learning_rate * grad

    if i % 10000 == 0:
      iterable.set_description('Log Likelihood: {}'.format(log_likelihood(x, y, b)))

  return b


def main():
  np.random.seed(12)
  num_observations = 5000

  x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
  x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

  x = np.vstack((x1, x2)).astype(np.float32)
  y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

  weights = get_weights(x, y, add_intercept=True)

  print(weights)

if __name__ == "__main__":
  main()
