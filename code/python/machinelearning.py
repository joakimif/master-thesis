import numpy as np
import pandas as pd
import seaborn as sns

import os
import random
import time
import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from IPython.display import display

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

pd.options.mode.chained_assignment = None

PROJECT_DIR = '../'
DATASET_DIR = 'datasets'
SAVE_DIR = 'data'

try:
    os.chdir(PROJECT_DIR)
except:
    print('No such directory')
    pass

CATEGORIES = ['CONDITION', 'CONTROL']
BIPOLAR = ['normal', 'bipolar II', 'unipolar', 'bipolar I']

def make_confusion_matrix(validations, predictions):
    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=BIPOLAR,
                yticklabels=BIPOLAR,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig('confusion_matrix.png')

def average_str(string):
    if(type(string) == str):
        num1, num2 = string.split('-')
        num1 = int(num1)
        num2 = int(num2)

        return (num1 + num2) // 2

    else:
        return string

def time_between(now, start, end):
    if start <= end:
        return start <= now < end
    else:
        return start <= now or now < end

def extract_time_of_day(timestamp):
    ts = time.strptime(timestamp.split(' ')[1], '%H:%M:%S')

    for label in part_of_days.keys():
        times = part_of_days[label]
        start = time.strptime(times[0], '%H:%M:%S')
        end = time.strptime(times[1], '%H:%M:%S')

        if time_between(ts, start, end):
            return label

    return None

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores.csv'))

scores['afftype'].fillna(0, inplace=True)

scores.loc[scores['melanch'] == 2, 'melanch'] = 0

scores['inpatient'] %= 2 # 1 = inpatient; 0 = not
scores['marriage'] %= 2 # 1 = married; 0 = not
scores['work'] %= 2 # 1 = working; 0 = not

scores['gender'] -= 1 # 0: male; 1: female

scores['age'] = scores['age'].apply(average_str)
scores['edu'] = scores['edu'].apply(average_str)

segments = []
labels = []

N_FEATURES = 1

SEG_LEN = 60
step = 60

for person in scores['number']:
    p = scores[scores['number'] == person]
    filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv')
    df_activity = pd.read_csv(filepath)

    for i in range(0, len(df_activity) - SEG_LEN, step):
        segment = df_activity['activity'].values[i : i + step]

        segments.append([segment])
        labels.append(p['afftype'].values[0])

labels = to_categorical(np.asarray(labels), 4)
segments = np.asarray(segments).reshape(-1, SEG_LEN, N_FEATURES)

num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
num_classes = 4
input_shape = num_time_periods * num_sensors

segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
labels = labels.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.4)

""" Start machine learning """ 

K.clear_session()

model = Sequential()
model.add(Reshape((SEG_LEN, num_sensors), input_shape=(input_shape,)))
model.add(Conv1D(100, 10, activation='relu', input_shape=(SEG_LEN, num_sensors)))
model.add(Conv1D(100, 10, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(80, 10, activation='relu'))
model.add(Conv1D(80, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 40
EPOCHS = 50

history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[
                        ModelCheckpoint(
                            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                            monitor='val_loss', save_best_only=True),
                        EarlyStopping(monitor='acc', patience=3)
                    ],
                    validation_split=0.2,
                    verbose=1)

y_pred_test = model.predict(X_test)

max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

print(classification_report(max_y_test, max_y_pred_test))

make_confusion_matrix(max_y_test, max_y_pred_test)