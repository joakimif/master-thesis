import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from parse_args import *

DATASET_DIR = '../datasets'
SEGMENT_LENGTH = segment_length
STEP = step
EPOCHS = epochs
BATCH_SIZE = batch_size

""" Create segments and labels """

scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores_2.csv'))
scores['madrs2'].fillna(0, inplace=True)

segments = []
labels = []

for person in scores['number']:
    p = scores[scores['number'] == person]
    filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv') # ../datasets/[control or condition]/[person].csv
    df_activity = pd.read_csv(filepath)

    for i in range(0, len(df_activity) - SEGMENT_LENGTH, STEP):
        segment = df_activity['activity'].values[i : i + SEGMENT_LENGTH]
        
        segments.append([segment])
        labels.append(p['madrs2'].values[0])

segments = np.asarray(segments)
segments = segments.reshape(-1, SEGMENT_LENGTH, 1)

input_shape = segments.shape[1]
segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
labels = np.asarray(labels).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.4)

""" Create model """

model = Sequential()
model.add(Reshape((SEGMENT_LENGTH, 1), input_shape=(input_shape,)))
model.add(Conv1D(128, 2, activation='relu', input_shape=(SEGMENT_LENGTH, 1), kernel_initializer='VarianceScaling'))

model.add(MaxPooling1D(pool_size=2, strides=1))

model.add(Conv1D(64, 2, activation='relu', kernel_initializer='VarianceScaling'))

model.add(GlobalAveragePooling1D())

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1, activation='linear', kernel_initializer='VarianceScaling'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

""" Train model """

h = model.fit(X_train,
                y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[
                    # EarlyStopping(monitor='mean_squared_error', patience=2),
                ],
                validation_data=(X_test, y_test),
                verbose=1)

print(model.evaluate(X_test, y_test))

actual = pd.DataFrame(y_test)
predictions = [x[0] for x in model.predict(X_test)]
predictions = pd.DataFrame(predictions, index=y_test.index)

actual.plot()
predictions.plot()

plt.legend(['predicted_score', 'actual_score'])
plt.ylabel('MADRS Score')

plt.savefig('graph.png')
