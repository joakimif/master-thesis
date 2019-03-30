from cnn.models import PredictionModel

step = 60
segment_length = 2880
learning_rate = 0.0001
optimizer = 'adam'
verbose = 1

input_shape = (1,)

model = PredictionModel(segment_length=segment_length, step=step, learning_rate=learning_rate, optimizer=optimizer, verbose=verbose)

