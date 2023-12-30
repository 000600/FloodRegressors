# Imports
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('flood.csv')
df = pd.DataFrame(df)

# Initialize x and y lists
x = []
y = list(df.pop("FloodProbability"))

# Add dataset to x and y lists
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0])): # Loop through all columns
    rows.append(df.iloc[row][point])
  x.append(rows)

# Divide the x and y values into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

# Create and train model
model = XGBRegressor(n_estimators = 100, learning_rate = 0.001, early_stopping_rounds = 5)
model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose = 1) # Predicts the popularity rating of music

# View mean squared error of the model
predictions = model.predict(x_test)
mse = mean_squared_error(predictions, y_test)
print(f"\nTest Mean Squared Error (MSE): {mse}")

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = model.predict([x_test[index]])[0]

print(f"Model's Prediction on a Sample Input: {prediction}")
print(f"Actual Label on the Same Input: {y_test[index]}\n")