import numpy as np
import pandas as pd

from cnn.models import PredictionModel

step = 60
segment_length = 2880
optimizer = 'adam'
learning_rate = 0.0001
batch_size = 16
epochs = 10

verbose = 1
DATASET_DIR = '../datasets'

scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores.csv'))
scores['madrs2'].fillna(0, inplace=True)

segments = []
labels = []

for person in scores['number']:
    p = scores[scores['number'] == person]
    filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv')
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

# X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=834567654)

model = PredictionModel(input_shape=input_shape, segment_length=segment_length, step=step, learning_rate=learning_rate, optimizer=optimizer, verbose=verbose)

