import os
import sys
import random

import numpy as np
import pandas as pd

from tensorflow.keras import backend as K

from cnn.models import ClassificationModel
from cnn.utils import create_segments_and_labels_loo

from parse_args import *

DATASET_DIR = '../datasets'

print('Participant,Correct,Prediction,Votes,Total')

start_i = 0

loo_results_filepath = 'leave_one_out_predictions.txt'
loo_filepath = 'leave_one_out.txt'

if os.path.isfile(loo_filepath):
    with open(loo_filepath, 'r') as f:
        start_i = int(f.read())+1

for i in range(start_i, 55):
    print(f'Leaving out {i}')

    segments, labels, left_out_segments, left_out_group, input_shape = create_segments_and_labels_loo(DATASET_DIR, segment_length, step, leave_out_id=i)
    model = ClassificationModel(input_shape=input_shape, segment_length=segment_length, step=step, optimizer=optimizer, verbose=verbose, dropout=0.5, n_output_classes=2)
    model.fit(segments, labels, batch_size, epochs)
    prediction = model.majority_voting_prediction(left_out_segments)

    with open(loo_results_filepath, 'a') as f:
        f.write(f'{i+1},{left_out_group},{prediction[0]},{prediction[1]},{prediction[2]}')

    with open(loo_filepath, 'w') as f:
        f.write(str(i))