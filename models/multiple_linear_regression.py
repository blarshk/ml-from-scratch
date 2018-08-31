import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

def cost_function(x, y, b):
  return np.sum((x.dot(b) - y) ** 2) / (2 * len(y))

def main():
  df = pd.read_csv('data/student_scores.csv')

  y = df['Writing'].values
  m = len(y)
  x = np.array([np.ones(m), df['Math'].values, df['Reading'].values]).T

  b = np.zeros(3)

  learning_rate = 0.0001
  epochs = 10000

  print(cost_function(x, y, b))

  costs = np.zeros(epochs)

  for epoch in range(epochs):
    h = x.dot(b)
    loss = h - y
    gradient = x.T.dot(loss) / m
    b = b - learning_rate * gradient
    cost = cost_function(x, y, b)
    costs[epoch] = cost

  print("b: ", b)
  print(costs[-1])
  
  preds = x.dot(b)

  print(np.sqrt(mean_squared_error(y, preds)))

if __name__ == "__main__":
  main()
