import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys
import random
import time
import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

verbose = len(sys.argv) > 1 and sys.argv[1] == '-v'

if verbose:
    verbose = 1
    print('Verbose mode.')
else:
    verbose = 0

CATEGORIES = ['CONDITION', 'CONTROL']
LABELS = ['normal', 'bipolar']

def make_confusion_matrix(validations, predictions, save_image_location=None, print_stdout=False):
    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if save_image_location:
        plt.savefig(save_image_location)

    if print_stdout:
        print('Confusion matrix:\n', matrix)

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

def is_at_night(timestamp):
    return time_between(time.strptime(timestamp.split(' ')[1], '%H:%M:%S'),
                        time.strptime('22:00:00', '%H:%M:%S'),
                        time.strptime('06:00:00', '%H:%M:%S'))

def is_daytime(timestamp):
    return time_between(time.strptime(timestamp.split(' ')[1], '%H:%M:%S'),
                        time.strptime('06:00:00', '%H:%M:%S'),
                        time.strptime('21:00:00', '%H:%M:%S'))

def create_segments_and_labels(n_features, segment_length, step, filter_timestamp=None):
    scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores.csv'))
    scores['afftype'].fillna(0, inplace=True)
    
    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]

            append = False

            if not filter_timestamp:
                append = True

            elif filter_timestamp == 'day' and is_daytime(df_activity['timestamp'].values[i]):
                append = True
                
            elif filter_timestamp == 'night' and is_at_night(df_activity['timestamp'].values[i]):
                append = True
            
            if append:
                segments.append([segment])

                if p['afftype'].values[0] == 0:
                    labels.append(0)
                else:
                    labels.append(1)

    labels = to_categorical(np.asarray(labels), 2)
    segments = np.asarray(segments).reshape(-1, segment_length, n_features)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    labels = labels.astype('float32')

    if verbose:
        print('\nINPUT DATA\n------------\n')
        print(f'Segments length:', len(segments), ':: Labels length:', len(labels))
        print(f'num_time_periods: {num_time_periods}, num_sensors: {num_sensors}, input_shape: {input_shape}')
        print('------------\n')
    
    return segments, labels, num_sensors, input_shape

def create_model(segment_length, num_sensors, input_shape, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    K.clear_session()

    model = Sequential()
    model.add(Reshape((segment_length, num_sensors), input_shape=(input_shape,)))
    model.add(Conv1D(100, 10, activation='relu', input_shape=(segment_length, num_sensors)))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if verbose:
        print(model.summary())

    return model

def train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.2):
    return model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    verbose=verbose)

def predict(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    print(classification_report(max_y_test, max_y_pred_test))

    return max_y_test, max_y_pred_test