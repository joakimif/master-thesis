import os
import json
import time
import datetime

import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from matplotlib import pyplot as plt


RESULTS_DIR = '../results'

def timestamp():
    return datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")

class Conv1DModel():
    history = None
    callbacks = []
    initial_epoch = 0
    model = Sequential()

    def __init__(self, input_shape=None, segment_length=None, step=None, n_output_classes=None, loss_function=None, metrics=None, optimizer='adam', identifier='Conv1D', verbose=0, old_path=None, path=None):
        if old_path:
            self.init_old(old_path)
        else:
            self.init_new(input_shape, segment_length, step, n_output_classes, loss_function, metrics, optimizer, identifier, verbose, path)

    def init_new(self, input_shape, segment_length, step, n_output_classes, loss_function, metrics, optimizer, identifier, verbose, path):
        assert input_shape != None
        assert segment_length != None
        assert step != None
        assert n_output_classes != None
        assert loss_function != None
        assert metrics != None
        assert optimizer != None

        self.input_shape = input_shape
        self.segment_length = segment_length
        self.step = step
        self.n_output_classes = n_output_classes
        self.loss_function = loss_function
        self.metrics = metrics
        self.optimizer = optimizer
        self.verbose = verbose
        
        if not path:
            self.directory = f'{RESULTS_DIR}/{identifier}_{timestamp()}'
        else:
            assert not os.path.isdir(path), f'Cannot set {path} as session directory because it already exists.'
            self.directory = path

        os.mkdir(self.directory)
        os.mkdir(f'{self.directory}/checkpoints')
        os.mkdir(f'{self.directory}/logs')
        os.mkdir(f'{self.directory}/img')

        self.save_settings()

    def init_old(self, old_path):
        with open(f'{old_path}/settings.json', 'r') as f:
            settings = json.loads(f.read())

        for key, value in settings.items():
            setattr(self, key, value)

        assert self.input_shape != None
        assert self.segment_length != None
        assert self.n_output_classes != None
        assert self.loss_function != None
        assert self.metrics != None
        assert self.optimizer != None
        assert self.directory != None
        
        model_path = f'{self.directory}/checkpoints/best_model.h5'
        
        assert os.path.isfile(model_path), f'Model at path {old_path} cannot to be restored.'

        self.model = load_model()

    def save_settings(self):
        settings = {
            'input_shape': self.input_shape,
            'segment_length': self.segment_length,
            'n_output_classes': self.n_output_classes,
            'loss_function': self.loss_function,
            'metrics': self.metrics,
            'optimizer': self.optimizer,
            'verbose': self.verbose,
            'directory': self.directory
        }

        with open(f'{self.directory}/settings.json', 'w') as f:
            f.write(json.dumps(settings))

    def add(self, layer):
        self.model.add(layer)

    def compile(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)

        if self.verbose:
            self.model.summary()

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def enable_tensorboard(self):
        self.add_callback(TensorBoard(log_dir=f'{self.directory}/logs'))

        if self.verbose:
            print('TensorBoard activated. Remember to run `tensorboard --logdir=/full_path_to_your_logs`.')

    def enable_checkpoints(self, monitor, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        filename = save_best_only and 'best_model' or '{epoch:02d}-{val_loss:.2f}'
        filename = save_weights_only and f'{filename}.hdf5' or f'{filename}.h5'

        filepath = f'{self.directory}/checkpoints/{filename}'

        self.add_callback(ModelCheckpoint(filepath, monitor, self.verbose, save_best_only, save_weights_only, mode, period))
        
    def longterm(self, monitor):
        self.enable_checkpoints(monitor, save_best_only=True)

    def fit(self, X, y, batch_size, epochs, validation_split=0.0, validation_data=None):
        self.history = self.model.fit(X, y, batch_size, epochs, 
                                    callbacks=self.callbacks, 
                                    validation_split=validation_split, 
                                    validation_data=validation_data, 
                                    initial_epoch=self.initial_epoch,
                                    verbose=self.verbose)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_test):
        return self.model.predict(X_test)


class ClassificationModel(Conv1DModel):
    def __init__(self, input_shape=None, segment_length=None, step=None, n_output_classes=None, optimizer='adam', dropout=None, verbose=0, old_path=None):
        if old_path:
            self.restore(old_path)
        else:
            self.create(input_shape, segment_length, step, n_output_classes, optimizer, dropout, verbose)

    def restore(self, old_path):
        super().__init__(old_path=old_path)

    def create(self, input_shape, segment_length, step, n_output_classes, optimizer, dropout, verbose):
        identifier = f'Conv1D_{n_output_classes}'

        super().__init__(input_shape, segment_length, step, n_output_classes, 
                    optimizer=optimizer, 
                    loss_function='categorical_crossentropy', 
                    metrics=['accuracy'], 
                    identifier=identifier,
                    verbose=verbose)

        self.add(Reshape((segment_length, 1), input_shape=(input_shape,)))
        self.add(Conv1D(100, 10, activation='relu', input_shape=(segment_length, 1)))
        self.add(Conv1D(100, 10, activation='relu'))
        self.add(MaxPooling1D(2))
        self.add(Conv1D(160, 10, activation='relu'))
        self.add(Conv1D(160, 10, activation='relu'))
        self.add(GlobalAveragePooling1D())

        if dropout:
            self.add(Dropout(dropout))
        
        self.add(Dense(output_classes, activation='softmax'))

        self.compile()


class PredictionModel(Conv1DModel):
    def __init__(self, input_shape=None, segment_length=None, step=None, optimizer='adam', learning_rate=None, verbose=0, old_path=None):
        if old_path:
            self.restore(old_path)
        else:
            self.create(input_shape, segment_length, step, optimizer, learning_rate, verbose)

    def restore(self, old_path):
        super().__init__(old_path=old_path)

    def create(self, input_shape, segment_length, step, optimizer, learning_rate, verbose):
        identifier = f'Conv1D_pred'

        super().__init__(input_shape, segment_length, step,
                    n_output_classes=1, 
                    loss_function='mean_squared_error', 
                    metrics=['mse'],
                    identifier=identifier,
                    verbose=verbose)
 
        self.add(Reshape((segment_length, 1), input_shape=(input_shape,)))
        self.add(Conv1D(128, 2, activation='relu', input_shape=(segment_length, 1)))
        self.add(MaxPooling1D(pool_size=2, strides=1))
        self.add(Conv1D(64, 2, activation='relu'))
        self.add(GlobalAveragePooling1D())
        self.add(Flatten())
        self.add(Dense(10, activation='relu'))
        self.add(Dense(1, activation='linear'))

        if optimizer == 'sgd':
            if learning_rate:
                self.optimizer = SGD(lr=learning_rate, nesterov=True)
            else:
                self.optimizer = SGD(nesterov=True)
        elif optimizer == 'nadam' and learning_rate:
            self.optimizer = Nadam(lr=learning_rate)

        self.compile()


    def graph_history(self, metric='mean_squared_error', title='Training Loss', xlabel='Epoch', ylabel='Mean Squared Error'):
        assert self.history != None, 'Model must be fit before generating history graph.'

        historydf = pd.DataFrame(self.history.history, index=self.history.epoch)
        historydf.xs(metric, axis=1).plot()

        plt.clf()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(f'{self.directory}/img/train_history{timestamp()}.png')