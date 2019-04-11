import numpy as np
import pandas as pd

from heatmap import heatmap

df = pd.read_csv('leave_one_out_predictions.txt', names=['Participant','Correct','Prediction','Votes','Total','Confidence'])

true_positives = len(df.query('Correct == 1 and Prediction == 1'))
false_positives = len(df.query('Correct == 0 and Prediction == 1'))
true_negatives = len(df.query('Correct == 0 and Prediction == 0'))
false_negatives = len(df.query('Correct == 1 and Prediction == 0'))

matrix = np.array([[true_positives, false_negatives], [false_positives, true_negatives]])

heatmap(matrix, xticklabels=['Condition', 'Control'], yticklabels=['Condition', 'Control'], filename='leave_one_out.pdf')