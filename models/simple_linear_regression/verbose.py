import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

def main():
  headbrain_dataframe = pd.read_csv('data/headbrain.csv')

  print(headbrain_dataframe.columns)

  head_sizes = headbrain_dataframe['Head Size(cm^3)']
  brain_weights = headbrain_dataframe['Brain Weight(grams)']

  head_sizes_mean = np.mean(head_sizes)
  brain_weights_mean = np.mean(brain_weights)

  numer = np.sum([(head_sizes[i] - head_sizes_mean) * (brain_weights[i] - brain_weights_mean) for i in range(len(head_sizes))])
  denom = np.sum([(head_sizes[i] - head_sizes_mean) ** 2 for i in range(len(head_sizes))])

  slope = numer / denom
  intercept = brain_weights_mean - (slope * head_sizes_mean)

  print("slope: ", slope)
  print("intercept: ", intercept)

  brain_weight_predictor = lambda head_size: intercept + (slope * head_size)

  brain_weight_predictions = [brain_weight_predictor(head_size) for head_size in head_sizes]

  brain_weight_prediction_mean_squared_error = mean_squared_error(brain_weights, brain_weight_predictions)

  print(brain_weight_prediction_mean_squared_error)
  print(np.sqrt(brain_weight_prediction_mean_squared_error))

if __name__ == "__main__":
  main()
