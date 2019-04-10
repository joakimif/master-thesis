import os
import random

import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical

def create_segments_and_labels_loo(dataset_dir, segment_length, step, n_output_classes=2):
    scores = pd.read_csv(os.path.join(dataset_dir, 'scores.csv'))
    scores['afftype'].fillna(0, inplace=True)
    
    segments = []
    labels = []

    left_out_segments = []
    left_out_correct = []

    r = random.randint(0, len(scores['number']))

    for i, person in enumerate(scores['number']):
        p = scores[scores['number'] == person]
        filepath = os.path.join(dataset_dir, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        if i == r:
            left_out_correct = p['afftype'].values[0] == 0 and 0 or 1

        for j in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[j : j + segment_length]
            
            if i == r:
                left_out_segments.append([segment])
            else:
                segments.append([segment])
                labels.append(p['afftype'].values[0] == 0 and 0 or 1)

    labels = np.asarray(labels).astype('float32')
    labels = to_categorical(labels, n_output_classes)

    segments = np.asarray(segments).reshape(-1, segment_length, 1)
    left_out_segments = np.asarray(left_out_segments).reshape(-1, segment_length, 1)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    left_out_segments = left_out_segments.reshape(left_out_segments.shape[0], input_shape).astype('float32')
    
    return segments, labels, left_out_segments, left_out_correct, input_shape


def create_segments_and_labels(dataset_dir, segment_length, step, n_output_classes=2):
    scores = pd.read_csv(os.path.join(dataset_dir, 'scores.csv'))
    scores['afftype'].fillna(0, inplace=True)
    
    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(dataset_dir, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]
            segments.append([segment])

            if p['afftype'].values[0] == 0:
                labels.append(0)
            else:
                labels.append(1)

    labels = np.asarray(labels).astype('float32')
    labels = to_categorical(labels, n_output_classes)
    
    segments = np.asarray(segments).reshape(-1, segment_length, 1)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    
    return segments, labels, input_shape

def create_segments_and_labels_prediction(dataset_dir, segment_length, step):
    scores = pd.read_csv(os.path.join(dataset_dir, 'scores.csv'))
    scores['madrs2'].fillna(0, inplace=True)

    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(dataset_dir, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]
            
            segments.append([segment])
            labels.append(p['madrs2'].values[0])

    segments = np.asarray(segments)
    segments = segments.reshape(-1, segment_length, 1)

    input_shape = segments.shape[1]
    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    labels = np.asarray(labels).astype('float32')

    return segments, labels, input_shape