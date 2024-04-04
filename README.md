# Flood Regressors
## The Neural Network
This neural network (found in the **floodNN.py** file) assesses the probability of a flood occurring in a certain region based on a variety of factors. Since the model is a regression algorithm that predicts probabilities, it uses a mean squared error loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001 in addition to multiple dropout layers and early stopping to prevent overfitting. The model has an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 32 input neurons and a ReLU activation function)
- 1 Dropout layer (with a dropout rate of 0.4)
- 2 Hidden layers (one with 16 neurons and the other with 8 neurons, both with a ReLU activation function)
- 1 Dropout layer (with a dropout rate of 0.4)
- 1 Output layer (with 1 output neuron and no activation function)

I found that the model had a mean squared error of around 0.001 after some hyperparameter tuning, although I am sure lower error values are possible. Feel free to further tune the hyperparameters or build upon the model!

## The XGB Regressor
An XGBoost Regressor model is also included in the **floodXGB.py**. The XGBoost Regressor has 100 estimators, a learning rate of 0.001, and early stopping based on validation sets. The classifier predicts the likelihood of a flood based on the same inputs as the model in the **floodNN.py** file. Although the number of estimators is lower than usual, I found that it achieved similar results.

As with the neural network, feel free to tune the hyperparameters or build upon the classifier!

## The Dataset
https://www.kaggle.com/datasets/brijlaldhankour/flood-prediction-factors/data

## Libraries
These neural networks and XGBoost Regressor were created with the help of the Tensorflow, Scikit-Learn, and XGBoost libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- XGBoost's Website: https://xgboost.readthedocs.io/en/stable/#
- XGBoost's Installation Instructions: https://xgboost.readthedocs.io/en/stable/install.html

## Disclaimer
Please note that I do not endorse, recommend, or encourage the use of my work here in any actual risk assessments or applications.
