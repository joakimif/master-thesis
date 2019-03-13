import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def regression_model():
	model = Sequential()
	model.add(Dense(8, input_dim=8, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

df = pd.read_csv('../datasets/scores.csv')

df['age'] = df['age'].apply(lambda x: type(x) == str and x.split('-')[0] or x)
df['edu'] = df['edu'].apply(lambda x: type(x) == str and x.split('-')[0] or x)
df['gender'] = df['gender'].apply(lambda x: x-1)
df['melanch'] = df['melanch'].apply(lambda x: x-1)
df['inpatient'] = df['inpatient'].apply(lambda x: x-1)
df['work'] = df['work'].apply(lambda x: x-1)

df['afftype'].fillna(0, inplace=True)
df['melanch'].fillna(0, inplace=True)
df['inpatient'].fillna(0, inplace=True)
df['madrs2'].fillna(0, inplace=True)
df['madrs1'].fillna(0, inplace=True)
df['work'].fillna(1, inplace=True)

df['afftype'].replace([2.0, 3.0], 1.0, inplace=True)

X = df[['gender', 'melanch', 'afftype', 'inpatient', 'age', 'edu', 'work', 'madrs1']]
y = df[['madrs2']]

X = X.values.astype('float32')
y = y.values.astype('float32')

print(X.shape)
print(y.shape)

estimator = KerasRegressor(build_fn=regression_model, epochs=100, batch_size=5, verbose=1)
estimator.fit(X, y)

prediction = estimator.predict(X)

print(pd.DataFrame(prediction).describe())
