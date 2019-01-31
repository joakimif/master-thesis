import numpy as np
import pandas as pd

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

#scores = df[df['number'].str.contains('condition')]

scores['afftype'].fillna(0, inplace=True)

scores.loc[scores['melanch'] == 2, 'melanch'] = 0

scores['inpatient'] %= 2 # 1 = inpatient; 0 = not
scores['marriage'] %= 2 # 1 = married; 0 = not
scores['work'] %= 2 # 1 = working; 0 = not

scores['gender'] -= 1 # 0: male; 1: female
#scores['afftype'] -= 1 # 0: bipolar II; 1: unipolar depressive; 2: bipolar I

scores['age'] = scores['age'].apply(average_str)
scores['edu'] = scores['edu'].apply(average_str)


segments = []
labels = []

N_FEATURES = 3

SEG_LEN = 60
step = 60

for person in scores['number']:
    p = scores[scores['number'] == person]
    filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv')
    df_activity = pd.read_csv(filepath)

    for i in range(0, len(df_activity) - SEG_LEN, step):
        segment = df_activity['activity'].values[i : i + step]

        segments.append([segment, segment, segment])
        labels.append(p['afftype'].values[0])

labels = to_categorical(np.asarray(labels), 4)
segments = np.asarray(segments).reshape(-1, SEG_LEN, N_FEATURES)


# In[7]:


num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
num_classes = 4
input_shape = num_time_periods * num_sensors


# In[8]:


display(num_time_periods, SEG_LEN, num_sensors, input_shape, segments.shape, labels.shape)


# In[9]:


segments = segments.reshape(segments.shape[0], input_shape)

segments = segments.astype('float32')
labels = labels.astype('float32')


# In[10]:


display(num_time_periods, SEG_LEN, num_sensors, input_shape, segments.shape, labels.shape)


# In[11]:


segments[1]


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.4)


# In[15]:


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


callbacks_list = [
    ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='acc', patience=1)
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 16
EPOCHS = 1

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

print(type(history))