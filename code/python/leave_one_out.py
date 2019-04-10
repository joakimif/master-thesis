import os
import sys
import random

import numpy as np
import pandas as pd

from cnn.models import ClassificationModel
from cnn.utils import create_segments_and_labels_loo

from parse_args import *

DATASET_DIR = '../datasets'

segments, labels, left_out_segments, left_out_group, input_shape = create_segments_and_labels_loo(DATASET_DIR, segment_length, step)

model = ClassificationModel(input_shape=input_shape, segment_length=segment_length, step=step, optimizer=optimizer, verbose=verbose, dropout=0.5, n_output_classes=2)

model.fit(segments, labels, batch_size, epochs)

prediction = model.majority_voting_prediction(left_out_segments)
print(f'Prediction: {prediction[0]} (votes: {prediction[1]}/{prediction[2]})')
print('Correct:', left_out_group)