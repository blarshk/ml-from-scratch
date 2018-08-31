import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

def main():
  df = pd.read_csv('data/headbrain.csv')

  print(df.columns)

  x = df['Head Size(cm^3)']
  y = df['Brain Weight(grams)']

  x_mean = np.mean(x)
  y_mean = np.mean(y)

  numer = np.sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])
  denom = np.sum([(x[i] - x_mean) ** 2 for i in range(len(x))])

  b1 = numer / denom
  b0 = y_mean - (b1 * x_mean)

  print("b1: ", b1)
  print("b0: ", b0)

  predictor = lambda x: b0 + (b1 * x)

  preds = [predictor(xi) for xi in x]

  mse = mean_squared_error(y, preds)

  print(mse)
  print(np.sqrt(mse))

if __name__ == "__main__":
  main()