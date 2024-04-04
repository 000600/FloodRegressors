# Imports
import tensorflow as tf
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('flood.csv')
df = pd.DataFrame(df)

# Initialize x and y sets
y = df['FloodProbability']
x = df.drop(labels = ['FloodProbability'], axis = 1)

# Divide the x and y values into three sets: train, test, and validation
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

# Get input shape
input_shape = x.shape[1]

# Create Adam optimizer
opt = Adam(learning_rate = 0.001)

# Create model
model = Sequential()

# Add an initial batch norm layer so that all the values are in a reasonable range for the network to process
model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu', input_shape = [input_shape])) # Input layer
model.add(Dropout(0.4))

# Hidden layers
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.4))

# Output layer
model.add(Dense(1)) # Linear activation because the model is a regression algorithm

# Compile model
model.compile(optimizer = opt, loss = 'mse', metrics = ['mse'])
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 10
history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_test, y_test), callbacks = [early_stopping])

file = 'FloodNN%03depochs.h5' % (epochs)
model.save(file)

# Visualize training and validation mean squared error
history_dict = history.history
train_mse = history_dict['mse']
val_mse = history_dict['val_mse']

plt.plot(train_mse, label = 'Training MSE')
plt.plot(val_mse, label = 'Validation MSE')
plt.title('Validation and Training MSE Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# View model's mean squared error on the test set
predictions = model.predict(x_test, verbose = 0)
mse = mean_squared_error(predictions, y_test)
print("\nTest Mean Squared Error (MSE):", mse)

# Compare model's prediction on an input to the actual label corresponding to that input (change the index to view a different input and output set)
index = 0

print(f"Model's Prediction on a Sample Input: {predictions[index]}")
print(f"Actual Label on the Same Input: {y_test.iat[index]}\n")
