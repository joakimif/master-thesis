import numpy as np
import pandas as pd

import os
import sys
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Nadam, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from parse_args import *

def regression_model():
	model = Sequential()
	model.add(Dense(5, input_dim=1, activation='relu'))
	#model.add(Dense(10, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	return model


df = pd.read_csv('../datasets/scores.csv')

df['age'] = df['age'].apply(lambda x: type(x) == str and x.split('-')[0] or x)
df['edu'] = df['edu'].apply(lambda x: type(x) == str and x.split('-')[0] or x)
df['gender'] = df['gender'].apply(lambda x: x-1)
df['melanch'] = df['melanch'].apply(lambda x: x-1)
df['inpatient'] = df['inpatient'].apply(lambda x: x-1)
df['work'] = df['work'].apply(lambda x: x-1)
df['id'] = df.index

df['afftype'].fillna(0, inplace=True)
df['melanch'].fillna(0, inplace=True)
df['madrs2'].fillna(0, inplace=True)
df['madrs1'].fillna(0, inplace=True)
df['work'].fillna(1, inplace=True)

df['afftype'].replace([2.0, 3.0], 1.0, inplace=True)

X_columns = ['gender', 'age', 'edu', 'work', 'madrs1', 'madrs2']
y_col = 'afftype'

results = []

for X_col in X_columns:
    X = df[X_col]
    y = df[y_col]

    X = X.values.astype('float32')
    y = y.values.astype('float32')

    seed = random.randint(1, 999) # same seed for X and y to make them index the same rows as before
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if do_load:
        regressor = load_model(f'../models/kerasregressor_{X_col}.h5')
    else:
        regressor = KerasRegressor(build_fn=regression_model, epochs=epochs, batch_size=batch_size, verbose=1)
        h = regressor.fit(X_train, y_train)
        regressor.model.save(f'../models/kerasregressor_{X_col}.h5')

    predictions = regressor.predict(X_test).round()
    predictions = list(zip(predictions, y_test))
    prediction_df = pd.DataFrame(predictions, columns=['Predicted', 'Actual'])
    
    results.append({'df': prediction_df, 'name': X_col, 'h': h})

    pd.DataFrame(h.history).plot()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(f'../img/kerasregressor_{X_col}.png')
    plt.clf()

for res in results:
    print(f'Predict {y_col} based on ' + res['name'] + ':')
    print('----------------')
    print(res['df'])
    print('----------------')