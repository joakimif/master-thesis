import os
import sys

import numpy as np
import pandas as pd

from cnn.models import PredictionModel
from cnn.utils import create_segments_and_labels_prediction

from sklearn.model_selection import train_test_split

DATASET_DIR = '../datasets'

step = 60
segment_length = 2880
optimizer = 'adam'
learning_rate = 0.0001
batch_size = 16
epochs = 10
verbose = 1

segments, labels, input_shape = create_segments_and_labels_prediction(DATASET_DIR, segment_length, step)

X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=834567654)

if '--continue' in sys.argv:
    model = PredictionModel(old_path=sys.argv[sys.argv.index('--continue')+1])
else:
    model = PredictionModel(input_shape=input_shape, segment_length=segment_length, step=step, learning_rate=learning_rate, optimizer=optimizer, verbose=verbose)

model.longterm('val_loss')
model.enable_tensorboard()
model.fit(X_train, y_train, batch_size, epochs, validation_split=0.4)
model.graph_history()