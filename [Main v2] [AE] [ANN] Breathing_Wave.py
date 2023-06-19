import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/yokahealthcare/Anasa-GAN/master/dataset/breathing_waveform_data.csv").iloc[:, :-1] # get rid of last column ("notes")

zeros_val = df[df.iloc[:, 1:].eq(0).any(axis=1)]

df = df[~df.isin(zeros_val)].dropna()

normal_df = df[df['labels'] == "normal"]
quick_df = df[df['labels'] == "quick"]
hold_df = df[df['labels'] == "hold"]
deep_df = df[df['labels'] == "deep"]
deep_quick_df = df[df['labels'] == "deep_quick"]

X = normal_df.iloc[:, :-1]
Y = normal_df.iloc[:, -1]

# Data (X)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Label (Y)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode class values as integers [0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,2]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
hot_y = np_utils.to_categorical(encoded_Y)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Activation
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from sklearn.base import BaseEstimator

def AE():
    model = Sequential()
    # Encoder layers
    model.add(Dense(64, input_shape=(85,)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dense(8))
    model.add(LeakyReLU())

    # Decoder layers
    model.add(Dense(16, input_shape=(8,)))
    model.add(LeakyReLU())
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dense(85))
    model.add(Activation(activations.sigmoid))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

    return model

import multiprocessing

cpu_count = multiprocessing.cpu_count()

print(f"Number of CPU cores: {cpu_count}")

from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
# from scikeras.wrappers import KerasRegressor
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_absolute_error

model_kr = KerasRegressor(model=AE)

# Define the hyperparameters and their values to search
param_grid = {
    'epochs': [10, 20],
    'batch_size': [32, 64]
}

# Create the grid search object
grid_search = GridSearchCV(estimator=model_kr, param_grid=param_grid, cv=10, verbose=5, refit=True, n_jobs=cpu_count)

# Fit the grid search object to your data
grid_search.fit(X, X)  # Assuming X_train is your training data

# Print the best hyperparameters and the corresponding mean squared error
print("Best Hyperparameters:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)